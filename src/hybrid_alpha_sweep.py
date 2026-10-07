"""
Sensitivity analysis: does ANY fusion weight make the Hybrid beat Isolation
Forest alone?

The Hybrid's weight (alpha = 0.10, i.e. 10% Isolation Forest / 90% LSTM) was
adopted unchanged from Shaheen & Alomari (2026), who derived it for network
traffic, not keystrokes. At that weight the Hybrid is close to the LSTM
Autoencoder alone and is worse than Isolation Forest alone. This script
sweeps alpha from 0.0 (LSTM Autoencoder only) to 1.0 (Isolation Forest only)
to check whether that conclusion depends on the borrowed weight.

Same Option A protocol and the same trained models as the other scripts
(one Isolation Forest + one LSTM Autoencoder per subject, trained once on
control_1ms); only the fusion weight changes, so the sweep costs one training
run. Evaluated at the three core resolutions, 10 phase-jitter seeds each.

Caveat: the best alpha is read off the TEST data, so its EER is an optimistic
estimate. That is the right direction for the question asked here: if even
the best weight chosen this way does not beat Isolation Forest, no weight
does.

Run: python3 src/hybrid_alpha_sweep.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from eer import compute_eer
from hypothesis_verdicts import bootstrap_ci, signed_rank_p
from run_multiseed_experiment import RAW_CSV, build_test_conditions
from subject_models import (N_TRAIN_REPS, get_feature_columns, fit_subject_models,
                            score_samples, fuse_scores)

RESOLUTIONS_MS = [1, 20, 30]
ALPHAS = [i / 10 for i in range(11)]
PAPER_ALPHA = 0.1


def evaluate_subject(raw_df: pd.DataFrame, test_df: pd.DataFrame, subject: str,
                      feature_cols: list, seed: int = 42) -> list:
    train_df = raw_df[raw_df["subject"] == subject].iloc[:N_TRAIN_REPS]  # control_1ms ONLY
    models = fit_subject_models(train_df, feature_cols, seed=seed)

    is_genuine = (test_df["subject"] == subject) & (test_df["role"] == "genuine")
    is_impostor = (test_df["subject"] != subject) & (test_df["role"] == "impostor")
    subj_test = test_df[is_genuine | is_impostor]

    scores = score_samples(models, subj_test)
    if_scores, lstm_scores = scores["Isolation Forest"], scores["LSTM Autoencoder"]
    genuine = (subj_test["role"] == "genuine").to_numpy()
    conditions = subj_test.groupby(["resolution_ms", "jitter_seed"], sort=True).indices

    rows = []
    for alpha in ALPHAS:
        fused = fuse_scores(models, if_scores, lstm_scores, alpha)
        for (res_ms, jitter_seed), idx in conditions.items():
            result = compute_eer(fused[idx][genuine[idx]], fused[idx][~genuine[idx]])
            rows.append({"subject": subject, "resolution_ms": res_ms,
                         "jitter_seed": jitter_seed, "alpha": alpha, "eer": result["eer"]})
    return rows


def summarise(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Mean EER (%) per alpha and resolution, plus the paired difference from
    Isolation Forest alone (alpha = 1.0) at each resolution."""
    per_subject = (detail_df.groupby(["subject", "alpha", "resolution_ms"])["eer"].mean() * 100
                   ).unstack(["alpha", "resolution_ms"])
    rows = []
    for alpha in ALPHAS:
        for res_ms in RESOLUTIONS_MS:
            e = per_subject[(alpha, res_ms)].to_numpy()
            d = e - per_subject[(1.0, res_ms)].to_numpy()
            lo, hi = bootstrap_ci(d)
            rows.append({"alpha": alpha, "resolution_ms": res_ms, "eer_mean": e.mean(),
                         "diff_vs_if_alone": d.mean(), "diff_ci_low": lo, "diff_ci_high": hi,
                         "p_differs_from_if_alone": signed_rank_p(d)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--out-prefix", type=str, default="results/hybrid_alpha_sweep")
    args = parser.parse_args()

    print("=" * 70)
    print(f"HYBRID ALPHA SWEEP: alpha in {ALPHAS} (0 = LSTM only, 1 = IF only)")
    print("=" * 70)

    raw_df = (pd.read_csv(RAW_CSV)
                .sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True))
    feature_cols = get_feature_columns(raw_df)
    test_df = build_test_conditions(raw_df, args.n_seeds, RESOLUTIONS_MS)

    subjects = sorted(raw_df["subject"].unique())
    if args.n_subjects:
        subjects = subjects[:args.n_subjects]

    all_rows = []
    for i, subj in enumerate(subjects):
        all_rows.extend(evaluate_subject(raw_df, test_df, subj, feature_cols))
        pd.DataFrame(all_rows).to_csv(f"{args.out_prefix}_by_subject.csv", index=False)
        print(f"  {subj} ({i+1}/{len(subjects)}) done", flush=True)

    summary = summarise(pd.DataFrame(all_rows))
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    print("\nMean EER (%) by fusion weight (rows) and resolution (columns):")
    print(summary.pivot(index="alpha", columns="resolution_ms", values="eer_mean").round(2).to_string())

    print("\nHybrid EER minus Isolation-Forest-alone EER (points; negative = Hybrid better):")
    for res_ms in RESOLUTIONS_MS:
        at_res = summary[summary["resolution_ms"] == res_ms]
        best = at_res.loc[at_res["eer_mean"].idxmin()]
        paper = at_res[at_res["alpha"] == PAPER_ALPHA].iloc[0]
        print(f"  {res_ms:>2}ms  best alpha = {best['alpha']:.1f}: {best['diff_vs_if_alone']:+.2f} "
              f"[{best['diff_ci_low']:+.2f}, {best['diff_ci_high']:+.2f}]   "
              f"paper alpha = {PAPER_ALPHA}: {paper['diff_vs_if_alone']:+.2f} "
              f"[{paper['diff_ci_low']:+.2f}, {paper['diff_ci_high']:+.2f}]")

    print(f"\nSaved {args.out_prefix}_by_subject.csv and {args.out_prefix}_summary.csv")
