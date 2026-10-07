"""
EXPLORATORY analysis (beyond H1/H2/H3): where does degradation actually
become unacceptable?

This is explicitly supplementary to your core hypotheses, which were
pre-specified at 20ms/30ms with a 5-point EER degradation threshold. Both
of those groups showed only mild degradation (well under 5 points), so
this script pushes further out to find where a real "breaking point"
appears - useful for a discussion section on practical deployment limits,
but should be clearly labeled as exploratory, not part of H1-H3.

Efficiency note: reuses the SAME Option A training (one Isolation Forest +
one LSTM Autoencoder per subject, trained once on control_1ms). Only the
downsampled TEST data changes per resolution, which is cheap to generate
and score - so adding more resolutions doesn't multiply training cost.

Resolutions tested: 1ms (baseline), 20ms, 30ms (your core hypotheses),
plus 50ms, 75ms, 100ms, 150ms, 200ms, 300ms, 500ms (exploratory).

Breaking point definition: the first resolution at which mean EER
degradation from the 1ms baseline exceeds a threshold (default 5
percentage points, matching your original H1 threshold for consistency).

Run: python3 src/breaking_point_analysis.py
"""
import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from eer import compute_eer
from downsample import downsample_dataframe
from subject_models import (get_feature_columns, training_reps, split_subject,
                            fit_subject_models, score_samples)

RAW_CSV = "data/raw/DSL-StrongPasswordData.csv"

# Core hypothesis groups (already validated) + exploratory extension
RESOLUTIONS_MS = [1, 20, 30, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 200, 300, 500]


def evaluate_subject(raw_df: pd.DataFrame, downsampled_dfs: dict, subject: str,
                      feature_cols: list, alpha: float = 0.10, seed: int = 42) -> list:
    # train ONCE on control_1ms
    models = fit_subject_models(training_reps(raw_df, subject), feature_cols, seed=seed)

    # ---------- Evaluate against EVERY resolution's test data ----------
    rows = []
    for res_ms, df in downsampled_dfs.items():
        genuine_test_df, impostor_df = split_subject(df, subject)
        genuine = score_samples(models, genuine_test_df, alpha=alpha)
        impostor = score_samples(models, impostor_df, alpha=alpha)

        for model_name in genuine:
            result = compute_eer(genuine[model_name], impostor[model_name])
            rows.append({"subject": subject, "resolution_ms": res_ms,
                         "model": model_name, "eer": result["eer"]})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--breaking-threshold-pts", type=float, default=5.0,
                         help="Degradation (percentage points) from 1ms baseline "
                              "that defines the 'breaking point' (default 5.0, "
                              "matching your original H1 threshold)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"BREAKING POINT ANALYSIS: testing resolutions {RESOLUTIONS_MS} ms")
    print("=" * 70)

    raw_df = pd.read_csv(RAW_CSV)
    feature_cols = get_feature_columns(raw_df)

    print("Generating downsampled test data for each resolution...")
    downsampled_dfs = {}
    for res_ms in RESOLUTIONS_MS:
        dt = res_ms / 1000.0
        downsampled_dfs[res_ms] = downsample_dataframe(raw_df, dt=dt, seed=res_ms)
        print(f"  {res_ms}ms: done")

    subjects = sorted(raw_df["subject"].unique())
    if args.n_subjects:
        subjects = subjects[:args.n_subjects]

    all_rows = []
    for i, subj in enumerate(subjects):
        rows = evaluate_subject(raw_df, downsampled_dfs, subj, feature_cols, alpha=args.alpha)
        all_rows.extend(rows)
        print(f"  {subj} ({i+1}/{len(subjects)}) done")

    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv("results/breaking_point_by_subject.csv", index=False)
    print("\nSaved results/breaking_point_by_subject.csv")

    summary = detail_df.groupby(["resolution_ms", "model"])["eer"].mean().reset_index()
    summary = summary.pivot(index="resolution_ms", columns="model", values="eer").reset_index()
    summary = summary.sort_values("resolution_ms")
    summary.to_csv("results/breaking_point_summary.csv", index=False)

    print("\nMean EER by resolution and model:")
    print(summary.to_string(index=False))

    print(f"\nBreaking point (degradation > {args.breaking_threshold_pts} pts from 1ms baseline):")
    baseline = summary[summary["resolution_ms"] == 1].iloc[0]
    for model in ["Isolation Forest", "LSTM Autoencoder", "Hybrid"]:
        base_eer = baseline[model]
        breaking_res = None
        for _, row in summary.iterrows():
            degradation_pts = (row[model] - base_eer) * 100
            if degradation_pts > args.breaking_threshold_pts:
                breaking_res = row["resolution_ms"]
                break
        if breaking_res:
            print(f"  {model}: breaks at {breaking_res}ms "
                  f"(EER={summary[summary['resolution_ms']==breaking_res][model].iloc[0]*100:.2f}%, "
                  f"baseline={base_eer*100:.2f}%)")
        else:
            print(f"  {model}: did NOT break within tested range "
                  f"(max tested {RESOLUTIONS_MS[-1]}ms, still under {args.breaking_threshold_pts}pt threshold)")

    print("\nSaved results/breaking_point_summary.csv")