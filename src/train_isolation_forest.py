"""
Step 4 (Option A protocol): Isolation Forest - train ONCE per subject on
control_1ms data only, then evaluate that same model against genuine/impostor
test samples from EACH sampling group (control_1ms, test_20ms, test_30ms).

Why this protocol (Option A), not per-group retraining:
  Real banks don't re-enroll every customer whenever they change their live
  polling rate. The profile is learned once (from however the data was
  originally captured, e.g. a batch upload at native ~1ms resolution), and
  the open research question is whether that SAME learned profile still
  recognizes the person when live monitoring later happens at a coarser
  20ms/30ms poll rate. This matches the methodology text: "Each user's
  typing profile from the control group will be used to train a
  user-specific model; anomaly scores will then be computed for held-out
  genuine and impostor samples."

Protocol per subject:
  - TRAIN on control_1ms, first 200 reps (sessions 1-4) only
  - For EACH group (control_1ms, test_20ms, test_30ms):
      * GENUINE TEST = that group's version of the subject's last 200 reps
        (sessions 5-8) - same held-out reps, evaluated at 3 resolutions
      * IMPOSTOR TEST = that group's version of first 5 reps from every
        other subject
  - EER computed per (subject, group), then mean/std reported per group
    across all 51 subjects.

Run: python3 src/train_isolation_forest.py
"""
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from eer import compute_eer
from subject_models import (get_feature_columns, load_groups, training_reps,
                            split_subject, fit_isolation_forest)


def get_subject_scores(dfs: dict, subject: str, feature_cols: list,
                        seed: int = 42) -> dict:
    """
    Train ONE Isolation Forest per subject on control_1ms data, then return
    raw anomaly scores (train + genuine-test + impostor-test) for EACH group.
    dfs: dict of group_name -> full dataframe (all subjects, one sampling rate)
    """
    train_df = training_reps(dfs["control_1ms"], subject)  # sessions 1-4, control_1ms ONLY
    model, train_scores = fit_isolation_forest(train_df, feature_cols, seed)

    per_group_scores = {}
    for group_name, df in dfs.items():
        genuine_test_df, impostor_df = split_subject(df, subject)  # THIS group's resolution
        per_group_scores[group_name] = {
            "genuine_scores": -model.score_samples(genuine_test_df[feature_cols].to_numpy()),
            "impostor_scores": -model.score_samples(impostor_df[feature_cols].to_numpy()),
        }

    return {"train_scores": train_scores, "by_group": per_group_scores}


def evaluate_subject(dfs: dict, subject: str, feature_cols: list, seed: int = 42) -> list:
    scores = get_subject_scores(dfs, subject, feature_cols, seed)
    rows = []
    for group_name, g in scores["by_group"].items():
        result = compute_eer(g["genuine_scores"], g["impostor_scores"])
        rows.append({
            "subject": subject,
            "group": group_name,
            "eer": result["eer"],
            "n_genuine_test": len(g["genuine_scores"]),
            "n_impostor_test": len(g["impostor_scores"]),
        })
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=None)
    args = parser.parse_args()

    dfs = load_groups()
    feature_cols = get_feature_columns(dfs["control_1ms"])
    subjects = sorted(dfs["control_1ms"]["subject"].unique())
    if args.n_subjects:
        subjects = subjects[:args.n_subjects]

    print("=" * 60)
    print("ISOLATION FOREST (Option A: train once on control_1ms)")
    print("=" * 60)

    all_rows = []
    for i, subj in enumerate(subjects):
        rows = evaluate_subject(dfs, subj, feature_cols)
        all_rows.extend(rows)
        summary_str = "  ".join(f"{r['group']}={r['eer']*100:.2f}%" for r in rows)
        print(f"  {subj} ({i+1}/{len(subjects)})  {summary_str}")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("results/isolation_forest_eer_by_subject.csv", index=False)
    print("\nSaved results/isolation_forest_eer_by_subject.csv")

    summary = results_df.groupby("group")["eer"].agg(["mean", "std", "min", "max"])
    summary = summary.reindex(["control_1ms", "test_20ms", "test_30ms"])
    summary.to_csv("results/isolation_forest_eer_summary.csv")
    print("\nSummary:")
    print(summary)