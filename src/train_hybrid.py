"""
Step 6 (Option A protocol + corrected weight): Hybrid model - score-level
fusion of Isolation Forest and LSTM Autoencoder, using the SAME train-once-
on-control_1ms models from the two scripts above.

Weight corrected to match the cited methodology: Shaheen and Alomari's
formula H = (0.90 x LSTM) + (0.10 x IF). Default alpha here is IF's weight,
so alpha=0.10 reproduces the cited paper's fusion exactly. (The previous
version defaulted to alpha=0.5, which did not match the methodology text -
this has been corrected.)

Run: python3 src/train_hybrid.py            # alpha=0.10 (paper default)
     python3 src/train_hybrid.py --alpha 0.5  # to compare against equal-weight
"""
import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from eer import compute_eer
from subject_models import (get_feature_columns, load_groups, training_reps,
                            split_subject, fit_subject_models, score_samples)


def evaluate_subject_hybrid(dfs: dict, subject: str, feature_cols: list,
                             alpha: float = 0.10, seed: int = 42) -> list:
    train_df = training_reps(dfs["control_1ms"], subject)
    models = fit_subject_models(train_df, feature_cols, seed=seed)

    rows = []
    for group_name, df in dfs.items():
        genuine_test_df, impostor_df = split_subject(df, subject)
        genuine = score_samples(models, genuine_test_df, alpha=alpha)
        impostor = score_samples(models, impostor_df, alpha=alpha)

        rows.append({
            "subject": subject,
            "group": group_name,
            "eer_hybrid": compute_eer(genuine["Hybrid"], impostor["Hybrid"])["eer"],
            "eer_isolation_forest": compute_eer(genuine["Isolation Forest"],
                                                impostor["Isolation Forest"])["eer"],
            "eer_lstm_autoencoder": compute_eer(genuine["LSTM Autoencoder"],
                                                impostor["LSTM Autoencoder"])["eer"],
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.10,
                         help="Weight on Isolation Forest score (default 0.10, "
                              "matching Shaheen & Alomari's 90%% LSTM / 10%% IF formula)")
    args = parser.parse_args()

    dfs = load_groups()
    feature_cols = get_feature_columns(dfs["control_1ms"])
    subjects = sorted(dfs["control_1ms"]["subject"].unique())
    if args.n_subjects:
        subjects = subjects[:args.n_subjects]

    print("=" * 70)
    print(f"HYBRID MODEL (alpha={args.alpha}, Option A: train once on control_1ms)")
    print("=" * 70)

    all_rows = []
    for i, subj in enumerate(subjects):
        rows = evaluate_subject_hybrid(dfs, subj, feature_cols, alpha=args.alpha)
        all_rows.extend(rows)
        summary_str = "  ".join(
            f"{r['group']}: H={r['eer_hybrid']*100:.2f}% "
            f"IF={r['eer_isolation_forest']*100:.2f}% "
            f"LSTM={r['eer_lstm_autoencoder']*100:.2f}%" for r in rows
        )
        print(f"  {subj} ({i+1}/{len(subjects)})  {summary_str}")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("results/hybrid_eer_by_subject.csv", index=False)
    print("\nSaved results/hybrid_eer_by_subject.csv")

    summary = results_df.groupby("group")[
        ["eer_hybrid", "eer_isolation_forest", "eer_lstm_autoencoder"]
    ].agg(["mean", "std"])
    summary = summary.reindex(["control_1ms", "test_20ms", "test_30ms"])
    summary.to_csv("results/hybrid_eer_summary.csv")
    print("\nSummary:")
    print(summary)