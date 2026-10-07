"""
Report figures, built from the hypothesis_verdicts.py outputs.

  Figure 1: mean EER vs sampling interval, 1-500ms, all three models
  Figure 2: EER degradation from the 1ms baseline vs the 5-point threshold
  Figure 3: sensitivity difference (LSTM degradation - IF degradation), the H2 test

All values are means over 51 subjects and 10 phase-jitter seeds; shaded bands
and error bars are 95% paired bootstrap confidence intervals over subjects.
Each series has its own marker shape as well as its own colour, so the
figures stay readable in greyscale print.

Run: python3 src/make_figures.py      (after hypothesis_verdicts.py)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CURVE_CSV = "results/degradation_curve.csv"
TESTS_CSV = "results/hypothesis_tests.csv"
OUT_DIR = "results/figures"

THRESHOLD_PTS = 5.0
ZOOM_MAX_MS = 100   # figures 2 and 3 show the range around the breaking point

# colour + marker per model (fixed, so a model looks the same in every figure)
SERIES = {
    "Isolation Forest": {"color": "#2a78d6", "marker": "o"},
    "LSTM Autoencoder": {"color": "#eb6834", "marker": "s"},
    "Hybrid": {"color": "#1baf7a", "marker": "^"},
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
})


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def plot_series(ax, x, y, lo, hi, model: str) -> None:
    s = SERIES[model]
    ax.fill_between(x, lo, hi, color=s["color"], alpha=0.12, linewidth=0)
    ax.plot(x, y, color=s["color"], linewidth=2, marker=s["marker"], markersize=6,
            markeredgecolor="white", markeredgewidth=1.2, label=model,
            solid_capstyle="round", solid_joinstyle="round")


def save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def figure_eer_curve(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model in SERIES:
        c = curve[curve["model"] == model].sort_values("resolution_ms")
        plot_series(ax, c["resolution_ms"], c["eer_mean"], c["eer_ci_low"], c["eer_ci_high"], model)

    # log axis: the tested intervals are dense at 50-100ms and sparse beyond
    ax.set_xscale("log")
    ticks = [1, 20, 30, 50, 100, 200, 500]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 55)
    ax.axhline(50, color=AXIS, linewidth=1)
    ax.text(1, 50.8, "50% = chance", color=INK_MUTED, fontsize=8.5, va="bottom")
    ax.set_xlabel("Sampling interval (ms, log scale)")
    ax.set_ylabel("Mean EER (%)")
    ax.set_title("Equal Error Rate by sampling interval")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    style_axes(ax)
    save(fig, "fig1_eer_by_sampling_interval.png")


def figure_degradation(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    zoom = curve[curve["resolution_ms"] <= ZOOM_MAX_MS]
    for model in SERIES:
        c = zoom[zoom["model"] == model].sort_values("resolution_ms")
        plot_series(ax, c["resolution_ms"], c["degradation_mean"],
                    c["degradation_ci_low"], c["degradation_ci_high"], model)

    ax.axhline(THRESHOLD_PTS, color=INK_SECONDARY, linewidth=1)
    ax.text(1, THRESHOLD_PTS + 0.25, f"{THRESHOLD_PTS:.0f}-point threshold (H1)",
            color=INK_SECONDARY, fontsize=9, va="bottom")
    ax.set_xlim(-2, ZOOM_MAX_MS + 3)
    ax.set_ylim(0, None)
    ax.set_xticks([1, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Sampling interval (ms)")
    ax.set_ylabel("EER increase over 1 ms baseline (points)")
    ax.set_title("EER degradation against the 5-point threshold")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    style_axes(ax)
    save(fig, "fig2_degradation_vs_threshold.png")


def figure_sensitivity_difference(tests: pd.DataFrame) -> None:
    h2 = tests[tests["hypothesis"].str.startswith("H2")
               & (tests["resolution_ms"] <= ZOOM_MAX_MS)].sort_values("resolution_ms")
    core = h2["hypothesis"] == "H2"

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(0, color=INK_SECONDARY, linewidth=1)
    for subset, filled, label in [(h2[core], True, "Pre-specified test (20, 30 ms)"),
                                  (h2[~core], False, "Exploratory")]:
        ax.errorbar(subset["resolution_ms"], subset["mean"],
                    yerr=[subset["mean"] - subset["ci_low"], subset["ci_high"] - subset["mean"]],
                    fmt="o", markersize=6.5, color=INK, ecolor=INK_SECONDARY,
                    elinewidth=1.4, capsize=0,
                    markerfacecolor=INK if filled else "white", markeredgewidth=1.4,
                    label=label)

    # headroom so the direction labels clear the error bars
    ymin = ax.get_ylim()[0]
    ymax = h2["ci_high"].max() + 1.5
    ax.set_ylim(ymin, ymax)
    ax.text(ZOOM_MAX_MS + 2, ymax - 0.1, "LSTM Autoencoder\ndegrades more", color=INK_SECONDARY,
            fontsize=9, va="top", ha="right")
    ax.text(ZOOM_MAX_MS + 2, ymin, "Isolation Forest\ndegrades more", color=INK_SECONDARY,
            fontsize=9, va="bottom", ha="right")
    ax.set_xlim(12, ZOOM_MAX_MS + 3)
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Sampling interval (ms)")
    ax.set_ylabel("LSTM degradation − IF degradation (points)")
    ax.set_title("Which model is more sensitive to downsampling (H2)")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    save(fig, "fig3_sensitivity_difference.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    curve = pd.read_csv(CURVE_CSV)
    tests = pd.read_csv(TESTS_CSV)
    figure_eer_curve(curve)
    figure_degradation(curve)
    figure_sensitivity_difference(tests)
