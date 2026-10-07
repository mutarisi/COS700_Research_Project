"""
Training-seed robustness run: repeats the whole multi-seed experiment with
several different MODEL TRAINING seeds.

Why this exists
---------------
run_multiseed_experiment.py varies the phase-jitter seed of the TEST data but
trains each subject's models once (seed 42). Training is itself random: the
LSTM Autoencoder starts from random weights and the Isolation Forest builds
random trees. This script retrains every subject's models under several
training seeds, so the hypothesis tests can average over training randomness
as well, and so the spread between training seeds can be reported.

For every training seed it runs the same protocol as
run_multiseed_experiment.py: 18 resolutions x 10 phase-jitter seeds, 51
subjects. Output has one extra column, train_seed.

Designed to be interrupted: results are saved after every subject, and a
restart skips whatever is already on disk.

Run: python3 src/run_training_seeds.py                       # 5 training seeds
     python3 src/run_training_seeds.py --train-seeds 42 1    # choose seeds
     python3 src/run_training_seeds.py --n-subjects 3 --n-seeds 2 --train-seeds 42   # smoke test
"""
import os
import sys
import time
import argparse
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from run_multiseed_experiment import RAW_CSV, RESOLUTIONS_MS, build_test_conditions, evaluate_subject
from subject_models import get_feature_columns

TRAIN_SEEDS = [42, 1, 2, 3, 4]   # 42 is the seed used everywhere else in the project
OUT_DIR = "results/training_seeds"
OUT_CSV = "results/training_seeds_eer_by_subject.csv"
CORE_MS = [1, 20, 30]


def run_one_seed(raw_df: pd.DataFrame, test_df: pd.DataFrame, subjects: list,
                  feature_cols: list, train_seed: int, alpha: float, out_dir: str) -> pd.DataFrame:
    """Evaluate every subject under one training seed, resuming from disk."""
    path = os.path.join(out_dir, f"train_seed_{train_seed}.csv")
    done_df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    done = set(done_df["subject"]) if len(done_df) else set()
    rows = done_df.to_dict("records")

    for i, subj in enumerate(subjects):
        if subj in done:
            continue
        t0 = time.time()
        subj_rows = evaluate_subject(raw_df, test_df, subj, feature_cols, alpha=alpha, seed=train_seed)
        rows.extend({"train_seed": train_seed, **r} for r in subj_rows)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  train seed {train_seed}: {subj} ({i+1}/{len(subjects)}) done "
              f"in {time.time() - t0:.0f}s", flush=True)

    seed_df = pd.DataFrame(rows)
    return seed_df[seed_df["subject"].isin(subjects)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=10,
                         help="Phase-jitter seeds per downsampled resolution (default 10)")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--out", type=str, default=OUT_CSV)
    args = parser.parse_args()

    print("=" * 70)
    print(f"TRAINING-SEED RUN: training seeds {args.train_seeds}, "
          f"{len(RESOLUTIONS_MS)} resolutions x {args.n_seeds} jitter seeds")
    print("=" * 70)

    os.makedirs(args.out_dir, exist_ok=True)
    raw_df = (pd.read_csv(RAW_CSV)
                .sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True))
    feature_cols = get_feature_columns(raw_df)
    test_df = build_test_conditions(raw_df, args.n_seeds)

    subjects = sorted(raw_df["subject"].unique())
    if args.n_subjects:
        subjects = subjects[:args.n_subjects]

    start = time.time()
    parts = []
    for train_seed in args.train_seeds:
        parts.append(run_one_seed(raw_df, test_df, subjects, feature_cols,
                                  train_seed, args.alpha, args.out_dir))
        pd.concat(parts, ignore_index=True).to_csv(args.out, index=False)
        print(f"Training seed {train_seed} complete "
              f"({(time.time() - start) / 60:.0f} min elapsed)\n", flush=True)

    detail_df = pd.concat(parts, ignore_index=True)
    print(f"Saved {args.out}  ({len(detail_df)} rows, "
          f"{detail_df['train_seed'].nunique()} training seeds, "
          f"{detail_df['subject'].nunique()} subjects)")

    core = detail_df[detail_df["resolution_ms"].isin(CORE_MS)]
    summary = (core.groupby(["train_seed", "model", "resolution_ms"])["eer"].mean()
                   .mul(100).unstack("resolution_ms").round(2))
    print("\nMean EER (%) at the core resolutions, per training seed:")
    print(summary.to_string())
