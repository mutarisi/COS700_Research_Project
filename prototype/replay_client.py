"""
Simulated bank clients: replays held-out CMU login attempts to the server.

Each attempt is sent the way a real client would send it: one WebSocket
connection per login, one message per key press and one per key release (the
study's "event-driven" architecture, 22 key messages per password), with the
original timing between messages. The client does the polling: at a polling
interval above 1ms it snaps every timestamp to its own poll grid (random
phase per attempt) before sending, using the study's quantisation.

For one account it mixes genuine attempts (the account holder's held-out
repetitions) with impostor attempts (other subjects typing the same password),
which is the study's test split. The "truth" field it sends is shown on the
monitoring console so the audience can see hits and misses; the server never
uses it for the decision.

The server's "Start replay" button runs replay() in-process. To run the
clients as a separate program (or from another machine):

    python prototype/replay_client.py --account s002 --poll-ms 20 --n 30
"""
import argparse
import asyncio
import json

import numpy as np
import pandas as pd
from tornado.websocket import websocket_connect

from detector import PASSWORD_KEYS, RAW_CSV
from downsample import _reconstruct_timestamps, _quantize
from subject_models import split_subject

MAX_PAUSE_S = 1.0      # cap on the wait between two key messages


async def send_attempt(url: str, account: str, press: np.ndarray, release: np.ndarray,
                       poll_ms: float, rng: np.random.Generator, speed: float = 4.0,
                       truth: str = None, source: str = "replay") -> dict:
    """Send one login attempt (press/release in seconds) and return the server's decision."""
    dt = poll_ms / 1000.0
    if dt > 0.001:
        phase = rng.uniform(0, dt)
        sent_press, sent_release = _quantize(press, dt, phase), _quantize(release, dt, phase)
    else:
        sent_press, sent_release = press, release

    # (true time, reported time, key, kind), in the order the keys really moved
    events = sorted([(press[i], sent_press[i], key, "down") for i, key in enumerate(PASSWORD_KEYS)] +
                    [(release[i], sent_release[i], key, "up") for i, key in enumerate(PASSWORD_KEYS)],
                    key=lambda event: event[0])

    ws = await websocket_connect(url)
    try:
        await ws.write_message(json.dumps({"type": "start", "account": account, "poll_ms": poll_ms,
                                           "source": source, "truth": truth}))
        previous = events[0][0]
        for true_t, sent_t, key, kind in events:
            await asyncio.sleep(min((true_t - previous) / speed, MAX_PAUSE_S))
            previous = true_t
            await ws.write_message(json.dumps({"type": "key", "k": key, "e": kind,
                                               "t": round(float(sent_t) * 1000.0, 3)}))
        await ws.write_message(json.dumps({"type": "submit"}))
        reply = await ws.read_message()
        return json.loads(reply) if reply else {"type": "error", "message": "connection closed"}
    finally:
        ws.close()


async def replay(url: str, data: pd.DataFrame, account: str, poll_ms: float = 1, n: int = 30,
                 impostor_share: float = 0.3, speed: float = 4.0, gap_s: float = 0.6,
                 seed: int = None, on_decision=None) -> None:
    """Send n login attempts for one CMU account, a random impostor_share of them impostors."""
    rng = np.random.default_rng(seed)
    genuine, impostor = split_subject(data, account)
    for _ in range(n):
        if rng.random() < impostor_share:
            row = impostor.iloc[rng.integers(len(impostor))]
            truth = f"impostor ({row['subject']})"
        else:
            row = genuine.iloc[rng.integers(len(genuine))]
            truth = "genuine"
        press, release = _reconstruct_timestamps(row)
        decision = await send_attempt(url, account, press, release, poll_ms, rng, speed, truth)
        if on_decision:
            on_decision(truth, decision)
        await asyncio.sleep(gap_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws/client")
    parser.add_argument("--account", default="s002")
    parser.add_argument("--poll-ms", type=float, default=1)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--impostor-share", type=float, default=0.3)
    parser.add_argument("--speed", type=float, default=4.0, help="4 = replay typing 4x faster than real")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    def show(truth, decision):
        if decision.get("type") != "decision":
            print(f"{truth:<18} -> {decision}")
            return
        verdict = "ALERT" if decision["alert"] else "ok"
        print(f"{truth:<18} score {decision['score']:.3f}  threshold {decision['threshold']:.3f}  -> {verdict}")

    asyncio.run(replay(args.url, pd.read_csv(RAW_CSV), args.account, args.poll_ms, args.n,
                       args.impostor_share, args.speed, seed=args.seed, on_decision=show))


if __name__ == "__main__":
    main()
