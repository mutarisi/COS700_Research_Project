"""
Problem 4: Explicit hypothesis verdicts, with paired significance tests.

Reads the per-subject results and tests each hypothesis AS PRE-SPECIFIED in the
proposal, reporting a CONFIRMED / PARTIALLY SUPPORTED / REJECTED /
NOT SUPPORTED verdict for each.

Input: by default the five-training-seed results (run_training_seeds.py, run
on Kaggle), falling back to the single-training-seed results
(run_multiseed_experiment.py) if that file is absent. Both hold one EER per
subject x model x resolution x phase-jitter seed; the first adds a train_seed
column.

H1: EER for both models stays within 5 percentage points of the 1ms baseline
    at 20ms (clause A), but EXCEEDS that threshold at 30ms (clause B).
H2: LSTM Autoencoder is MORE sensitive to downsampling than Isolation Forest
    (LSTM's degradation, in points, exceeds IF's, at both 20ms and 30ms).
H3: Data volume reduction at 20ms is >= 80%, without exceeding the 5-point
    EER degradation threshold.

Statistics
----------
Every subject is evaluated at every resolution with the same trained model,
so the design is PAIRED: the unit of analysis is the per-subject change in
EER, not the group means. Each subject's EER at a resolution is first
averaged over the phase-jitter seeds and, where present, the training seeds.
Then, across the 51 subjects:
  - 95% confidence intervals are percentile bootstrap intervals (10,000
    resamples of subjects) on the mean per-subject difference.
  - p-values are Wilcoxon signed-rank tests (no normality assumption),
    Holm-corrected within each hypothesis's family of tests.
A threshold claim ("within 5 points" / "exceeds 5 points") is decided by
whether the whole confidence interval lies on one side of the threshold.

Run: python3 src/hypothesis_verdicts.py
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

H1_THRESHOLD_PTS = 5.0
H3_THRESHOLD_PCT = 80.0
BASELINE_MS = 1
CORE_MS = [20, 30]
MODELS = ["Isolation Forest", "LSTM Autoencoder", "Hybrid"]
N_BOOTSTRAP = 10_000


def load_per_subject(path: str) -> pd.DataFrame:
    """Per-subject EER in percentage points, averaged over jitter seeds.
    Returns a dataframe indexed by subject with (model, resolution_ms) columns."""
    df = pd.read_csv(path)
    per_subject = df.groupby(["subject", "model", "resolution_ms"])["eer"].mean() * 100
    return per_subject.unstack(["model", "resolution_ms"])


def bootstrap_ci(x: np.ndarray, seed: int = 0) -> tuple:
    """95% percentile bootstrap CI of the mean of per-subject values x."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOTSTRAP, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signed_rank_p(x: np.ndarray, alternative: str = "two-sided") -> float:
    if np.allclose(x, 0):
        return 1.0
    return float(wilcoxon(x, alternative=alternative).pvalue)


def holm(p_values: list) -> list:
    """Holm step-down correction; returns adjusted p-values in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted.tolist()


def degradation(eer: pd.DataFrame, model: str, res_ms: int) -> np.ndarray:
    """Per-subject EER change (points) from the 1ms baseline."""
    return (eer[(model, res_ms)] - eer[(model, BASELINE_MS)]).to_numpy()


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def check_h1(eer: pd.DataFrame) -> tuple:
    print("=" * 78)
    print(f"H1: within {H1_THRESHOLD_PTS:.0f} pts of baseline at 20ms (clause A), "
          f"but EXCEEDS {H1_THRESHOLD_PTS:.0f} pts at 30ms (clause B)")
    print("=" * 78)

    rows = []
    for res_ms in CORE_MS:
        for model in MODELS:
            d = degradation(eer, model, res_ms)
            lo, hi = bootstrap_ci(d)
            rows.append({
                "hypothesis": "H1", "resolution_ms": res_ms, "model": model,
                "quantity": "EER degradation vs 1ms (pts)",
                "mean": d.mean(), "ci_low": lo, "ci_high": hi,
                "p_differs_from_zero": signed_rank_p(d),
                "p_below_threshold": signed_rank_p(d - H1_THRESHOLD_PTS, "less"),
            })
    table = pd.DataFrame(rows)
    table["p_differs_from_zero"] = holm(table["p_differs_from_zero"])
    table["p_below_threshold"] = holm(table["p_below_threshold"])

    for _, r in table.iterrows():
        print(f"  {r['resolution_ms']:>3}ms {r['model']:17s} degradation = {r['mean']:+.2f} pts  "
              f"95% CI [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  "
              f"p(!=0) = {fmt_p(r['p_differs_from_zero'])}  "
              f"p(<{H1_THRESHOLD_PTS:.0f}pt) = {fmt_p(r['p_below_threshold'])}")

    at_20 = table[table["resolution_ms"] == 20]
    at_30 = table[table["resolution_ms"] == 30]
    clause_a = bool((at_20["ci_high"] < H1_THRESHOLD_PTS).all())
    clause_b_supported = bool((at_30["ci_low"] > H1_THRESHOLD_PTS).any())
    clause_b_rejected = bool((at_30["ci_high"] < H1_THRESHOLD_PTS).all())

    print(f"\n  Clause A (within {H1_THRESHOLD_PTS:.0f} pts at 20ms): "
          f"{'SUPPORTED' if clause_a else 'NOT SUPPORTED'}")
    print(f"  Clause B (exceeds {H1_THRESHOLD_PTS:.0f} pts at 30ms): "
          f"{'SUPPORTED' if clause_b_supported else 'REJECTED' if clause_b_rejected else 'INCONCLUSIVE'}")

    if clause_a and clause_b_supported:
        verdict = "CONFIRMED"
    elif clause_a and clause_b_rejected:
        verdict = ("PARTIALLY SUPPORTED (degradation stays within the threshold at 20ms as "
                   "predicted, but ALSO at 30ms, where it was predicted to exceed it)")
    elif clause_a:
        verdict = "PARTIALLY SUPPORTED (clause A holds; clause B inconclusive)"
    else:
        verdict = "REJECTED"
    print(f"\nH1 VERDICT: {verdict}\n")
    return table, clause_a


def check_h2(eer: pd.DataFrame) -> pd.DataFrame:
    print("=" * 78)
    print("H2: LSTM Autoencoder degrades MORE than Isolation Forest under downsampling")
    print("=" * 78)

    rows = []
    for res_ms in [r for r in sorted(eer.columns.get_level_values(1).unique()) if r != BASELINE_MS]:
        d = degradation(eer, "LSTM Autoencoder", res_ms) - degradation(eer, "Isolation Forest", res_ms)
        lo, hi = bootstrap_ci(d)
        rows.append({
            "hypothesis": "H2" if res_ms in CORE_MS else "H2 (exploratory)",
            "resolution_ms": res_ms, "model": "LSTM Autoencoder - Isolation Forest",
            "quantity": "difference in EER degradation (pts)",
            "mean": d.mean(), "ci_low": lo, "ci_high": hi,
            "p_differs_from_zero": signed_rank_p(d),
        })
    table = pd.DataFrame(rows)
    core = table["hypothesis"] == "H2"
    table.loc[core, "p_differs_from_zero"] = holm(table.loc[core, "p_differs_from_zero"])
    # exploratory resolutions are reported uncorrected and are not part of the verdict

    for _, r in table.iterrows():
        tag = "" if r["hypothesis"] == "H2" else "  [exploratory, uncorrected]"
        print(f"  {r['resolution_ms']:>3}ms  LSTM degradation - IF degradation = {r['mean']:+.2f} pts  "
              f"95% CI [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  "
              f"p = {fmt_p(r['p_differs_from_zero'])}{tag}")

    core_rows = table[core]
    if (core_rows["ci_low"] > 0).all():
        verdict = "CONFIRMED (LSTM degrades significantly more at both 20ms and 30ms)"
    elif (core_rows["ci_high"] < 0).all():
        verdict = "REJECTED (Isolation Forest degrades significantly MORE than LSTM at both 20ms and 30ms)"
    elif (core_rows["ci_high"] < 0).any():
        verdict = ("REJECTED (LSTM is not more sensitive; Isolation Forest degrades "
                   "significantly more at one of the two resolutions)")
    else:
        verdict = ("NOT SUPPORTED (no statistically reliable difference in sensitivity "
                   "between the two models at 20ms/30ms)")
    print(f"\nH2 VERDICT: {verdict}\n")
    return table


def check_h3(h1_clause_a: bool) -> None:
    print("=" * 78)
    print(f"H3: data volume reduction at 20ms >= {H3_THRESHOLD_PCT:.0f}%, "
          f"without exceeding the {H1_THRESHOLD_PTS:.0f}-pt threshold")
    print("=" * 78)

    reduction_20ms = (1 - BASELINE_MS / 20) * 100
    volume_ok = reduction_20ms >= H3_THRESHOLD_PCT
    print(f"  Data volume reduction at 20ms = {reduction_20ms:.2f}%  "
          f"(analytic: 1 - 1ms/20ms; fixed by the choice of interval, not measured)")
    print(f"  EER degradation at 20ms within {H1_THRESHOLD_PTS:.0f} pts for every model: "
          f"{'YES' if h1_clause_a else 'NO'}  (empirical, from H1 clause A)")
    verdict = "CONFIRMED" if volume_ok and h1_clause_a else "REJECTED"
    print(f"\nH3 VERDICT: {verdict} (the volume figure holds by construction; the empirical "
          f"content of H3 is the accuracy condition)\n")


def degradation_curve(eer: pd.DataFrame) -> pd.DataFrame:
    """Mean EER and degradation with 95% CIs for every model x resolution."""
    rows = []
    for model in MODELS:
        for res_ms in sorted(eer[model].columns):
            e = eer[(model, res_ms)].to_numpy()
            d = degradation(eer, model, res_ms)
            e_lo, e_hi = bootstrap_ci(e)
            d_lo, d_hi = bootstrap_ci(d)
            rows.append({"model": model, "resolution_ms": res_ms,
                         "eer_mean": e.mean(), "eer_ci_low": e_lo, "eer_ci_high": e_hi,
                         "degradation_mean": d.mean(),
                         "degradation_ci_low": d_lo, "degradation_ci_high": d_hi})
    return pd.DataFrame(rows)


def report_breaking_points(curve: pd.DataFrame) -> None:
    print("=" * 78)
    print(f"EXPLORATORY: breaking point ({H1_THRESHOLD_PTS:.0f}-pt degradation from 1ms baseline)")
    print("=" * 78)
    for model in MODELS:
        c = curve[curve["model"] == model].sort_values("resolution_ms")
        within = c[c["degradation_ci_high"] < H1_THRESHOLD_PTS]["resolution_ms"]
        mean_cross = c[c["degradation_mean"] > H1_THRESHOLD_PTS]["resolution_ms"]
        exceeded = c[c["degradation_ci_low"] > H1_THRESHOLD_PTS]["resolution_ms"]
        # "confidently within" only counts resolutions below the first mean crossing
        first_cross = mean_cross.min() if len(mean_cross) else np.inf
        within = within[within < first_cross]
        not_reached = "not within tested range"
        print(f"  {model:17s} confidently within up to {within.max()}ms;  "
              f"mean crosses at {f'{mean_cross.min()}ms' if len(mean_cross) else not_reached};  "
              f"confidently exceeded from {f'{exceeded.min()}ms' if len(exceeded) else not_reached}")
    print()


def report_seed_noise(path: str) -> None:
    df = pd.read_csv(path)
    core = df[df["resolution_ms"].isin(CORE_MS)]
    if core["jitter_seed"].nunique() < 2:
        return
    per_seed = core.groupby(["model", "resolution_ms", "jitter_seed"])["eer"].mean() * 100
    spread = per_seed.groupby(["model", "resolution_ms"]).agg(["min", "max", "std"])
    print("=" * 78)
    print("Phase-jitter seed noise: mean EER (%) of a SINGLE-seed run, across seeds")
    print("=" * 78)
    print(spread.round(2).to_string())
    print()


def report_training_seeds(path: str) -> None:
    """Per-training-seed breakdown: how much the headline quantities move when
    the models are retrained with a different random seed."""
    df = pd.read_csv(path)
    if "train_seed" not in df.columns or df["train_seed"].nunique() < 2:
        return

    rows = []
    for train_seed, seed_df in df.groupby("train_seed"):
        eer = (seed_df.groupby(["subject", "model", "resolution_ms"])["eer"].mean() * 100
               ).unstack(["model", "resolution_ms"])
        row = {"train_seed": train_seed}
        for model in MODELS:
            for res_ms in [BASELINE_MS] + CORE_MS:
                row[f"eer_{model}_{res_ms}ms"] = eer[(model, res_ms)].mean()
            crossed = [r for r in sorted(eer[model].columns)
                       if degradation(eer, model, r).mean() > H1_THRESHOLD_PTS]
            row[f"breaking_point_ms_{model}"] = crossed[0] if crossed else np.nan
        for res_ms in CORE_MS:
            d = degradation(eer, "LSTM Autoencoder", res_ms) - degradation(eer, "Isolation Forest", res_ms)
            lo, hi = bootstrap_ci(d)
            row.update({f"h2_diff_{res_ms}ms": d.mean(), f"h2_ci_low_{res_ms}ms": lo,
                        f"h2_ci_high_{res_ms}ms": hi, f"h2_p_{res_ms}ms": signed_rank_p(d)})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv("results/training_seed_breakdown.csv", index=False)

    print("=" * 78)
    print(f"Training-seed robustness: each quantity recomputed per training seed ({len(table)} seeds)")
    print("=" * 78)
    print("  Mean EER (%) at 1ms, across training seeds:")
    for model in MODELS:
        v = table[f"eer_{model}_{BASELINE_MS}ms"]
        print(f"    {model:17s} min {v.min():.2f}  max {v.max():.2f}  std {v.std():.2f}")
    print("  H2 difference (LSTM degradation - IF degradation), per training seed:")
    for _, r in table.iterrows():
        print(f"    seed {int(r['train_seed']):>2}  " + "   ".join(
            f"{res_ms}ms {r[f'h2_diff_{res_ms}ms']:+.2f} [{r[f'h2_ci_low_{res_ms}ms']:+.2f}, "
            f"{r[f'h2_ci_high_{res_ms}ms']:+.2f}] p={fmt_p(r[f'h2_p_{res_ms}ms'])}" for res_ms in CORE_MS))
    print(f"  Breaking point (mean degradation first exceeds {H1_THRESHOLD_PTS:.0f} pts), per training seed:")
    for model in MODELS:
        print(f"    {model:17s} " + ", ".join(f"{int(v)}ms" for v in table[f"breaking_point_ms_{model}"]))
    print()


if __name__ == "__main__":
    import os

    default_input = "results/training_seeds_eer_by_subject.csv"
    if not os.path.exists(default_input):
        default_input = "results/multiseed_eer_by_subject.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=default_input)
    args = parser.parse_args()

    eer = load_per_subject(args.input)
    raw = pd.read_csv(args.input)
    n_train = raw["train_seed"].nunique() if "train_seed" in raw.columns else 1
    print(f"Input: {args.input}")
    print(f"{len(eer)} subjects, paired across resolutions ({n_train} training seed(s), "
          f"{raw['jitter_seed'].nunique()} jitter seeds per downsampled resolution)\n")

    h1_table, h1_clause_a = check_h1(eer)
    h2_table = check_h2(eer)
    check_h3(h1_clause_a)

    curve = degradation_curve(eer)
    report_breaking_points(curve)
    report_seed_noise(args.input)
    report_training_seeds(args.input)

    pd.concat([h1_table, h2_table], ignore_index=True).to_csv("results/hypothesis_tests.csv", index=False)
    curve.to_csv("results/degradation_curve.csv", index=False)
    print("Saved results/hypothesis_tests.csv and results/degradation_curve.csv")
