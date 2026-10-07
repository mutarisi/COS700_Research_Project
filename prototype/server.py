"""
Real-time alerting prototype: the server.

  bank clients  --WebSocket /ws/client-->  server  --WebSocket /ws/console-->  monitoring console
  (login page or replay_client.py)         (detector.py)                       (browser)

A client streams one login attempt as key press/release messages. When the
attempt is submitted the server checks the password, scores the typing rhythm
against the account's profile, and publishes the decision to every open
monitoring console. A correct password typed with the wrong rhythm raises an
account-compromise alert; the client is told to do a step-up verification.

Run from the project root:

    python prototype/server.py

then open  http://localhost:8000        (monitoring console)
     and   http://localhost:8000/login  (demo bank login)
"""
import argparse
import asyncio
import json
import os
import time

import tornado.web
import tornado.websocket

import replay_client
from detector import (Detector, WrongPassword, pair_key_events,
                      MIN_ENROL_ATTEMPTS, TARGET_ENROL_ATTEMPTS)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
HISTORY_LIMIT = 500
MAX_EVENTS_PER_ATTEMPT = 200
# Account the console preselects: close to the average account's error rates
# (check_alert_rates.py --by-account); accounts differ a lot, see the README.
DEFAULT_REPLAY_ACCOUNT = "s025"


class Hub:
    """State shared by all connections: the detector, the open consoles, the
    recent decisions, and the replay task started from a console."""

    def __init__(self, detector: Detector, port: int):
        self.detector = detector
        self.port = port
        self.consoles = set()
        self.history = []
        self.replay_task = None
        self._next_id = 1

    def broadcast(self, message: dict) -> None:
        data = json.dumps(message)
        for console in list(self.consoles):
            try:
                console.write_message(data)
            except tornado.websocket.WebSocketClosedError:
                self.consoles.discard(console)

    def publish(self, event: dict) -> None:
        """Record one login decision and push it to the consoles."""
        event["id"] = self._next_id
        event["time"] = time.strftime("%H:%M:%S")
        self._next_id += 1
        self.history.append(event)
        del self.history[:-HISTORY_LIMIT]
        self.broadcast({"type": "event", "event": event})

    def replay_running(self) -> bool:
        return self.replay_task is not None and not self.replay_task.done()

    def start_replay(self, account: str, poll_ms: float, n: int, impostor_share: float) -> None:
        if self.replay_running():
            return
        url = f"ws://127.0.0.1:{self.port}/ws/client"
        self.replay_task = asyncio.ensure_future(
            replay_client.replay(url, self.detector.data, account, poll_ms, n, impostor_share))
        self.replay_task.add_done_callback(self._replay_finished)
        self.broadcast({"type": "replay", "running": True})

    def stop_replay(self) -> None:
        if self.replay_running():
            self.replay_task.cancel()

    def _replay_finished(self, task) -> None:
        if not task.cancelled() and task.exception():
            print(f"Replay failed: {task.exception()!r}")
        self.broadcast({"type": "replay", "running": False})


class ClientSocket(tornado.websocket.WebSocketHandler):
    """A bank client. Messages: start, key (x22), submit; plus accounts and
    enrol_reset from the login page."""

    def initialize(self, hub: Hub):
        self.hub = hub
        self.attempt = None

    def reply(self, message: dict) -> None:
        self.write_message(json.dumps(message))

    def on_message(self, raw):
        try:
            message = json.loads(raw)
            kind = message["type"]
            if kind == "accounts":
                self.reply({"type": "accounts", "accounts": self.hub.detector.accounts(),
                            "enrol_target": TARGET_ENROL_ATTEMPTS})
            elif kind == "enrol_reset":
                self.hub.detector.reset_enrolment(str(message["account"]))
                self.reply({"type": "enrol", "count": 0, "target": TARGET_ENROL_ATTEMPTS, "ready": False})
            elif kind == "start":
                self.attempt = {
                    "account": str(message["account"]),
                    "poll_ms": min(max(float(message.get("poll_ms", 1)), 1.0), 1000.0),
                    "purpose": "enrol" if message.get("purpose") == "enrol" else "login",
                    "source": "replay" if message.get("source") == "replay" else "live",
                    "truth": str(message["truth"])[:40] if message.get("truth") else None,
                    "events": [], "key_messages": 0, "bytes": 0,
                }
            elif self.attempt is None:
                self.reply({"type": "error", "message": "No login attempt in progress."})
            elif kind == "key":
                attempt = self.attempt
                if len(attempt["events"]) < MAX_EVENTS_PER_ATTEMPT:
                    attempt["events"].append((str(message["k"]), str(message["e"]), float(message["t"])))
                attempt["key_messages"] += 1
                attempt["bytes"] += len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
            elif kind == "submit":
                attempt, self.attempt = self.attempt, None
                self.reply(self.decide(attempt))
        except (KeyError, TypeError, ValueError) as error:
            self.reply({"type": "error", "message": f"Bad message: {error}"})

    def decide(self, attempt: dict) -> dict:
        detector = self.hub.detector
        account = attempt["account"]
        try:
            press, release = pair_key_events(attempt["events"])
        except WrongPassword:
            if attempt["purpose"] == "login":
                self.publish(attempt, outcome="wrong_password")
            return {"type": "decision", "outcome": "wrong_password", "alert": False}

        if attempt["purpose"] == "enrol":
            count = detector.add_enrolment(account, press, release)
            return {"type": "enrol", "count": count, "target": TARGET_ENROL_ATTEMPTS,
                    "ready": count >= MIN_ENROL_ATTEMPTS}

        try:
            result = detector.assess(account, press, release, attempt["poll_ms"])
        except KeyError:
            return {"type": "decision", "outcome": "unknown_account", "alert": False}
        outcome = "step_up" if result["alert"] else "accepted"
        self.publish(attempt, outcome=outcome, **result)
        return {"type": "decision", "outcome": outcome, **result}

    def publish(self, attempt: dict, **fields) -> None:
        self.hub.publish({"account": attempt["account"], "poll_ms": attempt["poll_ms"],
                          "source": attempt["source"], "truth": attempt["truth"],
                          "key_messages": attempt["key_messages"], "bytes": attempt["bytes"],
                          **fields})


class ConsoleSocket(tornado.websocket.WebSocketHandler):
    """A monitoring console. Receives every decision; can start/stop a replay."""

    def initialize(self, hub: Hub):
        self.hub = hub

    def open(self):
        self.hub.consoles.add(self)
        self.write_message(json.dumps({"type": "hello", "accounts": self.hub.detector.subjects,
                                       "default_account": DEFAULT_REPLAY_ACCOUNT,
                                       "history": self.hub.history,
                                       "replay_running": self.hub.replay_running()}))

    def on_close(self):
        self.hub.consoles.discard(self)

    def on_message(self, raw):
        try:
            message = json.loads(raw)
            kind = message["type"]
            if kind == "replay_start":
                account = str(message["account"])
                if account not in self.hub.detector.subjects:
                    raise ValueError(f"unknown account {account}")
                self.hub.start_replay(account,
                                      poll_ms=min(max(float(message["poll_ms"]), 1.0), 1000.0),
                                      n=min(max(int(message["n"]), 1), 500),
                                      impostor_share=min(max(float(message["impostor_share"]), 0.0), 1.0))
            elif kind == "replay_stop":
                self.hub.stop_replay()
            elif kind == "clear":
                self.hub.history.clear()
                self.hub.broadcast({"type": "clear"})
        except (KeyError, TypeError, ValueError) as error:
            self.write_message(json.dumps({"type": "error", "message": f"Bad message: {error}"}))


class Page(tornado.web.RequestHandler):
    def initialize(self, filename: str):
        self.filename = filename

    def get(self):
        self.set_header("Cache-Control", "no-store")
        with open(os.path.join(STATIC_DIR, self.filename), encoding="utf-8") as f:
            self.write(f.read())


def make_app(hub: Hub) -> tornado.web.Application:
    return tornado.web.Application([
        (r"/", Page, {"filename": "console.html"}),
        (r"/login", Page, {"filename": "login.html"}),
        (r"/ws/client", ClientSocket, {"hub": hub}),
        (r"/ws/console", ConsoleSocket, {"hub": hub}),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": STATIC_DIR}),
    ])


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="use 0.0.0.0 to let other devices on the network connect")
    args = parser.parse_args()

    print("Loading the CMU dataset...")
    hub = Hub(Detector(), args.port)
    make_app(hub).listen(args.port, address=args.host)
    print(f"Monitoring console: http://localhost:{args.port}")
    print(f"Demo bank login:    http://localhost:{args.port}/login")
    print("Ctrl+C to stop.")
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
