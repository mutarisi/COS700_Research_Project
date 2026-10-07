"""
Multi-seed robustness run: the same Option A protocol as the other scripts,
but every downsampled resolution is regenerated with several independent
phase-jitter seeds instead of one.

Why this exists
---------------
The downsampling protocol draws a random phase offset per repetition, so a
single run at (say) 20ms is one random draw of the test data. Two single-seed
runs of the same resolution differ by ~0.2-0.3 EER points, which is the same
size as some of the effects being tested (e.g. H2). Repeating each resolution
over several jitter seeds and averaging per subject removes that noise, and
the per-subject output is what hypothesis_verdicts.py runs its paired
significance tests on.

Training is unchanged: ONE Isolation Forest + ONE LSTM Autoencoder per
subject, trained once on control_1ms (first 200 reps). Only the test data is
regenerated per (resolution, jitter seed), so extra seeds add scoring time
only, not training time.

Run: python3 src/run_multiseed_experiment.py              # 10 jitter seeds
     python3 src/run_multiseed_experiment.py --n-subjects 3 --n-seeds 2   # smoke test
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from eer import compute_eer
from downsample import downsample_dataframe
from subject_models import (N_TRAIN_REPS, N_IMPOSTOR_REPS, get_feature_columns,
                            fit_subject_models, score_samples)

RAW_CSV = "data/raw/DSL-StrongPasswordData.csv"
OUT_CSV = "results/multiseed_eer_by_subject.csv"

# Same resolutions as breaking_point_analysis.py (1 = undownsampled control)
RESOLUTIONS_MS = [1, 20, 30, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 200, 300, 500]


def build_test_conditions(raw_df: pd.DataFrame, n_seeds: int,
                           resolutions_ms: list = RESOLUTIONS_MS) -> pd.DataFrame:
    """Return the test rows (genuine + impostor pool) of every subject under
    every (resolution, jitter seed) condition, stacked into one dataframe with
    added columns resolution_ms, jitter_seed and role ('genuine'/'impostor')."""
    rep_position = raw_df.groupby("subject").cumcount()
    role = pd.Series(np.where(rep_position >= N_TRAIN_REPS, "genuine",
                              np.where(rep_position < N_IMPOSTOR_REPS, "impostor", "")),
                     index=raw_df.index)
    keep = role != ""

    parts = []
    for res_ms in resolutions_ms:
        # the 1ms control has no jitter, so one copy is enough
        seeds = [0] if res_ms == 1 else range(n_seeds)
        for jitter_seed in seeds:
            df = downsample_dataframe(raw_df, dt=res_ms / 1000.0,
                                       seed=jitter_seed * 1000 + res_ms)
            part = df[keep].copy()
            part["role"] = role[keep]
            part["resolution_ms"] = res_ms
            part["jitter_seed"] = jitter_seed
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def evaluate_subject(raw_df: pd.DataFrame, test_df: pd.DataFrame, subject: str,
                      feature_cols: list, alpha: float = 0.10, seed: int = 42) -> list:
    train_df = raw_df[raw_df["subject"] == subject].iloc[:N_TRAIN_REPS]  # control_1ms ONLY
    models = fit_subject_models(train_df, feature_cols, seed=seed)

    is_genuine = (test_df["subject"] == subject) & (test_df["role"] == "genuine")
    is_impostor = (test_df["subject"] != subject) & (test_df["role"] == "impostor")
    subj_test = test_df[is_genuine | is_impostor]

    # one scoring pass over every condition at once
    scores = score_samples(models, subj_test, alpha=alpha)
    genuine = (subj_test["role"] == "genuine").to_numpy()
    conditions = subj_test.groupby(["resolution_ms", "jitter_seed"], sort=True).indices

    rows = []
    for (res_ms, jitter_seed), idx in conditions.items():
        for model_name, model_scores in scores.items():
            s = model_scores[idx]
            result = compute_eer(s[genuine[idx]], s[~genuine[idx]])
            rows.append({"subject": subject, "resolution_ms": res_ms,
                         "jitter_seed": jitter_seed, "model": model_name,
                         "eer": result["eer"]})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=10,
                         help="Independent phase-jitter seeds per downsampled resolution (default 10)")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--out", type=str, default=OUT_CSV)
    args = parser.parse_args()

    print("=" * 70)
    print(f"MULTI-SEED RUN: {len(RESOLUTIONS_MS)} resolutions x {args.n_seeds} jitter seeds "
          f"(alpha={args.alpha}, Option A protocol)")
    print("=" * 70)

    raw_df = (pd.read_csv(RAW_CSV)
                .sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True))
    feature_cols = get_feature_columns(raw_df)

    print("Generating downsampled test data for every (resolution, seed)...")
    test_df = build_test_conditions(raw_df, args.n_seeds)
    print(f"  {test_df.groupby(['resolution_ms', 'jitter_seed']).ngroups} conditions, "
          f"{len(test_df)} test rows")

    subjects = sorted(raw_df["subject"].unique())
    if args.n_subjects:
        subjects = subjects[:args.n_subjects]

    all_rows = []
    for i, subj in enumerate(subjects):
        all_rows.extend(evaluate_subject(raw_df, test_df, subj, feature_cols, alpha=args.alpha))
        # rewrite after every subject so a crash doesn't lose the run so far
        pd.DataFrame(all_rows).to_csv(args.out, index=False)
        print(f"  {subj} ({i+1}/{len(subjects)}) done", flush=True)

    detail_df = pd.DataFrame(all_rows)
    print(f"\nSaved {args.out}")

    summary = (detail_df.groupby(["resolution_ms", "model"])["eer"].mean()
                        .unstack("model").mul(100).round(2))
    print("\nMean EER (%) by resolution and model, averaged over subjects and jitter seeds:")
    print(summary.to_string())
