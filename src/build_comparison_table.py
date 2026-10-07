"""
Step 7 (Option A protocol + corrected weight): Final comparison table.

Trains Isolation Forest and LSTM Autoencoder ONCE per subject on control_1ms
data only, then evaluates both against genuine/impostor test samples from
EACH group (control_1ms, test_20ms, test_30ms) - same protocol as the
individual model scripts, done together here so latency and hybrid fusion
can reuse the same trained models without retraining a third time.

Hybrid weight corrected to alpha=0.10 (10% Isolation Forest, 90% LSTM),
matching Shaheen and Alomari's cited formula.

Metric notes
------------
- FAR / FRR reported AT the EER operating threshold (standard convention;
  will be very close to EER by construction).
- Inference latency measured as wall-clock time to SCORE (not train) the
  test samples for that subject/group, per sample, in ms. Hybrid latency
  = IF latency + LSTM latency (both run at inference time in deployment).
- Data volume reduction is analytic: reduction = 1 - (dt_baseline / dt),
  representing the theoretical reduction in hardware polls needed.

Run: python3 src/build_comparison_table.py            # alpha=0.10 (paper default)
     python3 src/build_comparison_table.py --alpha 0.5  # to compare weightings
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from eer import compute_eer
from subject_models import (get_feature_columns, load_groups, training_reps, split_subject,
                            fit_subject_models, scale_sequences, reconstruction_error,
                            fuse_scores)

DT_BASELINE = 0.001
GROUP_DT = {"control_1ms": 0.001, "test_20ms": 0.020, "test_30ms": 0.030}


def far_frr_at_threshold(eer_result: dict) -> tuple:
    far = np.interp(eer_result["threshold"], eer_result["thresholds"], eer_result["far_curve"])
    frr = np.interp(eer_result["threshold"], eer_result["thresholds"], eer_result["frr_curve"])
    return float(far), float(frr)


def evaluate_subject_all_models(dfs: dict, subject: str, feature_cols: list,
                                 alpha: float = 0.10, seed: int = 42) -> list:
    # train ONCE on control_1ms, sessions 1-4
    train_df = training_reps(dfs["control_1ms"], subject)
    models = fit_subject_models(train_df, feature_cols, seed=seed)

    # ---------- Evaluate against EACH group's test data ----------
    rows = []
    for group_name, df in dfs.items():
        genuine_test_df, impostor_df = split_subject(df, subject)

        # IF scoring + latency (timing covers scoring only)
        X_gen = genuine_test_df[feature_cols].to_numpy()
        X_imp = impostor_df[feature_cols].to_numpy()
        t0 = time.perf_counter()
        if_gen_scores = -models.if_model.score_samples(X_gen)
        if_imp_scores = -models.if_model.score_samples(X_imp)
        if_latency_ms = (time.perf_counter() - t0) * 1000 / (len(X_gen) + len(X_imp))

        # LSTM scoring + latency
        X_gen_seq = scale_sequences(models.scaler, genuine_test_df)
        X_imp_seq = scale_sequences(models.scaler, impostor_df)
        t0 = time.perf_counter()
        lstm_gen_scores = reconstruction_error(models.lstm_model, X_gen_seq)
        lstm_imp_scores = reconstruction_error(models.lstm_model, X_imp_seq)
        lstm_latency_ms = (time.perf_counter() - t0) * 1000 / (len(X_gen_seq) + len(X_imp_seq))

        # Hybrid fusion (alpha = IF weight; default 0.10 = 10% IF / 90% LSTM)
        hybrid_gen = fuse_scores(models, if_gen_scores, lstm_gen_scores, alpha)
        hybrid_imp = fuse_scores(models, if_imp_scores, lstm_imp_scores, alpha)
        hybrid_latency_ms = if_latency_ms + lstm_latency_ms

        for model_name, gen_scores, imp_scores, latency in [
            ("Isolation Forest", if_gen_scores, if_imp_scores, if_latency_ms),
            ("LSTM Autoencoder", lstm_gen_scores, lstm_imp_scores, lstm_latency_ms),
            ("Hybrid", hybrid_gen, hybrid_imp, hybrid_latency_ms),
        ]:
            result = compute_eer(gen_scores, imp_scores)
            far, frr = far_frr_at_threshold(result)
            rows.append({
                "subject": subject, "group": group_name, "model": model_name,
                "eer": result["eer"], "far": far, "frr": frr,
                "latency_ms": latency,
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
    print(f"BUILDING FULL COMPARISON TABLE (alpha={args.alpha}, Option A protocol)")
    print("=" * 70)

    all_rows = []
    for i, subj in enumerate(subjects):
        rows = evaluate_subject_all_models(dfs, subj, feature_cols, alpha=args.alpha)
        all_rows.extend(rows)
        print(f"  {subj} ({i+1}/{len(subjects)}) done")

    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv("results/full_comparison_by_subject.csv", index=False)
    print("\nSaved results/full_comparison_by_subject.csv")

    summary = detail_df.groupby(["group", "model"]).agg(
        eer_mean=("eer", "mean"), eer_std=("eer", "std"),
        far_mean=("far", "mean"), frr_mean=("frr", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
    ).reset_index()

    summary["data_volume_reduction_pct"] = summary["group"].map(
        lambda g: (1 - DT_BASELINE / GROUP_DT[g]) * 100
    )

    group_order = ["control_1ms", "test_20ms", "test_30ms"]
    model_order = ["Isolation Forest", "LSTM Autoencoder", "Hybrid"]
    summary["group"] = pd.Categorical(summary["group"], categories=group_order, ordered=True)
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values(["group", "model"]).reset_index(drop=True)

    summary.to_csv("results/final_comparison_table.csv", index=False)
    print("\nFinal comparison table:")
    print(summary.to_string(index=False))
    print("\nSaved results/final_comparison_table.csv")