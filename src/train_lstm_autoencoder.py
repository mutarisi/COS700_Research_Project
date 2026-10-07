"""
Step 5 (Option A protocol): LSTM Autoencoder - train ONCE per subject on
control_1ms data only, then evaluate against genuine/impostor test samples
from EACH sampling group. Same rationale as train_isolation_forest.py.

Sequence framing (implemented in subject_models.py)
----------------
Each rep of ".tie5Roanl" involves 11 keys. The 31 flat H/DD/UD columns
become a sequence of 11 timesteps, 3 features each: [H_i, DD_in_i, UD_in_i].

Run: python3 src/train_lstm_autoencoder.py
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from eer import compute_eer
from subject_models import (load_groups, training_reps, split_subject,
                            fit_lstm_autoencoder, scale_sequences, reconstruction_error)


def get_subject_scores(dfs: dict, subject: str, seed: int = 42,
                        epochs: int = 60, verbose_fit: int = 0) -> dict:
    """Train ONE LSTM Autoencoder per subject on control_1ms data, then
    return reconstruction-error scores for EACH group's test samples."""
    train_df = training_reps(dfs["control_1ms"], subject)  # control_1ms ONLY
    model, scaler, train_scores = fit_lstm_autoencoder(train_df, seed, epochs, verbose_fit)

    per_group_scores = {}
    for group_name, df in dfs.items():
        genuine_test_df, impostor_df = split_subject(df, subject)  # THIS group's resolution
        per_group_scores[group_name] = {
            "genuine_scores": reconstruction_error(model, scale_sequences(scaler, genuine_test_df)),
            "impostor_scores": reconstruction_error(model, scale_sequences(scaler, impostor_df)),
        }

    return {"train_scores": train_scores, "by_group": per_group_scores}


def evaluate_subject(dfs: dict, subject: str, seed: int = 42) -> list:
    scores = get_subject_scores(dfs, subject, seed)
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
    subjects = sorted(dfs["control_1ms"]["subject"].unique())
    if args.n_subjects:
        subjects = subjects[:args.n_subjects]

    print("=" * 60)
    print("LSTM AUTOENCODER (Option A: train once on control_1ms)")
    print("=" * 60)

    all_rows = []
    for i, subj in enumerate(subjects):
        rows = evaluate_subject(dfs, subj)
        all_rows.extend(rows)
        summary_str = "  ".join(f"{r['group']}={r['eer']*100:.2f}%" for r in rows)
        print(f"  {subj} ({i+1}/{len(subjects)})  {summary_str}")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("results/lstm_autoencoder_eer_by_subject.csv", index=False)
    print("\nSaved results/lstm_autoencoder_eer_by_subject.csv")

    summary = results_df.groupby("group")["eer"].agg(["mean", "std", "min", "max"])
    summary = summary.reindex(["control_1ms", "test_20ms", "test_30ms"])
    summary.to_csv("results/lstm_autoencoder_eer_summary.csv")
    print("\nSummary:")
    print(summary)