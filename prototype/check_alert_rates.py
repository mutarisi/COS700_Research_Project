"""
What error rates does the prototype's alert rule give?

The study reports EER, whose threshold is picked afterwards with impostor
scores. The prototype cannot do that, so it uses a threshold taken from the
account holder's own enrolment attempts (see detector.py). This script runs
that rule over all 51 CMU accounts with the study's test split and reports:

  missed impostors  - impostor attempts that raised no alert   (FAR)
  false alarms      - genuine attempts that raised an alert    (FRR)

for the threshold calibrated at the client's polling interval (what the
prototype does), and for comparison a threshold left at its 1ms value.

Indicative only: one training seed, a few phase-jitter seeds, Isolation
Forest only. It is not part of the study's hypothesis tests.

Run from the project root:  python prototype/check_alert_rates.py
"""
import argparse

import numpy as np

from detector import Detector
from downsample import downsample_dataframe

POLL_MS = [1, 20, 30, 60, 100]
TEST_SEEDS = [0, 1, 2]      # phase-jitter seeds for the test attempts (calibration uses 1000+)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--by-account", action="store_true",
                        help="also list every account's rates at 1ms and 30ms")
    args = parser.parse_args()

    detector = Detector(percentile=args.percentile)
    subjects = detector.subjects[:args.n_subjects] if args.n_subjects else detector.subjects

    rates = {poll: {"calibrated": [], "fixed": []} for poll in POLL_MS}
    for subject in subjects:
        profile = detector.profile(subject)
        genuine, impostor = detector.replay_pool(subject)
        for poll in POLL_MS:
            seeds = TEST_SEEDS if poll > 1 else [0]
            for seed in seeds:
                g = profile.score(downsample_dataframe(genuine, poll / 1000.0, seed=seed))
                i = profile.score(downsample_dataframe(impostor, poll / 1000.0, seed=seed))
                for name, threshold in (("calibrated", profile.threshold(poll)),
                                        ("fixed", profile.threshold(1))):
                    rates[poll][name].append(((i <= threshold).mean(), (g > threshold).mean()))

    print(f"Alert threshold = {args.percentile:g}th percentile of the account holder's enrolment scores")
    print(f"{len(subjects)} accounts, Isolation Forest, mean over accounts (%)\n")
    print(f"{'polling':>8} | {'threshold calibrated at interval':^34} | {'threshold left at 1ms value':^34}")
    print(f"{'':>8} | {'missed impostors':>16} {'false alarms':>16}  | {'missed impostors':>16} {'false alarms':>16}")
    for poll in POLL_MS:
        cal = np.array(rates[poll]["calibrated"]).mean(axis=0) * 100
        fix = np.array(rates[poll]["fixed"]).mean(axis=0) * 100
        print(f"{poll:>6}ms | {cal[0]:>16.1f} {cal[1]:>16.1f}  | {fix[0]:>16.1f} {fix[1]:>16.1f}")

    if args.by_account:
        # each account contributed len(seeds) rows per interval, in subject order
        at_1ms = np.array(rates[1]["calibrated"]) * 100
        at_30ms = np.array(rates[30]["calibrated"]).reshape(len(subjects), len(TEST_SEEDS), 2).mean(axis=1) * 100
        print()
        print(f"{'account':>8} | {'1ms: missed':>12} {'false alarms':>13} | {'30ms: missed':>12} {'false alarms':>13}")
        for k in np.argsort(at_1ms.sum(axis=1)):
            print(f"{subjects[k]:>8} | {at_1ms[k, 0]:>12.1f} {at_1ms[k, 1]:>13.1f} | "
                  f"{at_30ms[k, 0]:>12.1f} {at_30ms[k, 1]:>13.1f}")


if __name__ == "__main__":
    main()
