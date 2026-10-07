# Real-time alerting prototype

A small working system that raises an account-compromise alert, at login,
when the password is correct but the typing rhythm does not match the account
holder. It is a demonstrator built on the study's detector; it was not used to
produce any result in the report.

```
bank clients  --WebSocket-->  server  --WebSocket-->  monitoring console
(login page, or replayed      (Isolation Forest       (browser: login feed
 CMU attempts)                 + alert threshold)       and alerts)
```

A client streams one login as 22 key press/release messages (the study's
"event-driven" architecture). The client does the polling: above 1 ms it
snaps every timestamp to its own poll grid before sending. On submit the
server checks the password, scores the rhythm, and pushes the decision to the
console, typically within about 10-15 ms.

## Run it

From the project root, with the project's virtual environment active (no extra
packages are needed; `tornado` is already in `requirements.txt`):

```bash
python prototype/server.py
```

Then open:

- http://localhost:8000 - the fraud monitoring console
- http://localhost:8000/login - the demo bank login page

The server reads `data/raw/DSL-StrongPasswordData.csv` directly, so none of
the study's pipeline steps has to be run first.

## Demo 1: replayed logins (nobody types)

On the console choose an account, a polling interval and an impostor share,
then **Start replay**. The server's simulated clients send held-out CMU login
attempts for that account, one at a time: the account holder's own later
repetitions mixed with other subjects typing the same password. This is the
study's test split, so nothing the model was trained on is replayed.

Each login appears in the feed with its rhythm score, the alert threshold and
the decision. Because replayed logins have a known origin, the console also
shows whether each decision was right (caught / missed / false alarm); that
label is never used for the decision.

Things worth showing:

- Run the same account at 1 ms, 30 ms and 100 ms. The alerts barely change up
  to 30 ms and visibly miss more impostors at 100 ms.
- Accounts differ a lot, as the report's between-subject spread says. `s036`
  is an easy account, `s025` (the default) is close to average, and `s002` is
  a hard one whose own logins are often flagged.

The simulated clients can also be run as a separate program, or from another
machine if the server was started with `--host 0.0.0.0`:

```bash
python prototype/replay_client.py --account s025 --poll-ms 20 --n 30
```

## Demo 2: live typing

Nobody enters real credentials. Every account uses the CMU password
`.tie5Roanl`, typed in one go without Backspace, then Enter.

1. Before the demo, open the login page, choose **Enrol a new account**, give
   it a name and type the password 30 times. The account can be used after 15;
   more makes the profile steadier.
2. At the demo, choose **Sign in**, pick that account, and have someone else
   type the password once. The console shows the login and, if the rhythm does
   not match, the alert. Then type it yourself.

Treat this part as an illustration, not as evidence:

- With 15-50 enrolment attempts instead of the study's 200, roughly one in
  four of the account holder's own logins is flagged (measured by enrolling
  CMU subjects on that few repetitions).
- Someone typing an unfamiliar password for the first time is a much easier
  impostor than a practised attacker.
- Browsers and keyboards already limit timing to a few milliseconds, so the
  "1 ms" option means "as fine as the browser reports".

Live enrolments are saved in `prototype/profiles/`, which is not tracked in
git because keystroke timing is biometric data. Delete a file there, or use
**Start enrolment over**, to remove an account.

## How the alert threshold is set

The study reports EER, whose threshold is chosen afterwards using impostor
scores. A live system has no impostor data, so the prototype takes the
threshold from the account holder's own enrolment attempts: the 90th
percentile of their anomaly scores. Coarser polling raises every score, so
the enrolment attempts are first quantised to the client's polling interval
and each interval gets its own threshold.

`check_alert_rates.py` measures what this rule gives over all 51 CMU accounts
(mean %, Isolation Forest, one training seed, three phase-jitter seeds):

| Polling | Missed impostors | False alarms | False alarms if the threshold is left at its 1 ms value |
|---|---|---|---|
| 1 ms | 11.5 | 11.4 | 11.4 |
| 20 ms | 13.8 | 10.4 | 15.3 |
| 30 ms | 15.4 | 10.1 | 21.1 |
| 60 ms | 21.1 | 9.1 | 51.4 |
| 100 ms | 28.6 | 7.9 | 58.6 |

```bash
python prototype/check_alert_rates.py                # the table above (~40 s)
python prototype/check_alert_rates.py --by-account   # plus every account's rates
```

These figures are indicative and are not part of the study's hypothesis tests.
At roughly one error in ten, the alert is a risk signal that triggers a
step-up check, not a decision to block.

## What it does not do

- It alerts at the login event only; it does not monitor a session.
- It knows one fixed password.
- Impostors are people, not bots.
- It runs on one machine over plain WebSocket: no encryption, no real
  accounts, no cloud deployment, and it measures no infrastructure cost.
- Only the Isolation Forest is used, as the report recommends; the LSTM
  Autoencoder is too slow to train during a live enrolment.

## Files

| File | Purpose |
|---|---|
| `server.py` | Web server: client and console WebSocket endpoints, decision per login |
| `detector.py` | Key events -> features -> Isolation Forest score -> alert threshold; live enrolment |
| `replay_client.py` | Simulated bank clients replaying held-out CMU attempts |
| `check_alert_rates.py` | Error rates of the alert rule across polling intervals |
| `test_prototype.py` | Unit tests for the detector |
| `static/` | The console and login pages |

```bash
python -m unittest discover prototype -v
```
