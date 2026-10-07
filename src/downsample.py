"""
Step 2: Down-sampling simulation protocol.

Method
------
The CMU dataset gives relative timing features (H = hold time, DD =
down-down latency, UD = up-down latency) per keystroke pair, not raw
timestamps. To simulate what a *coarser-resolution polling system*
would actually observe, we:

  1. Reconstruct absolute press/release timestamps for each of the 11
     keys in ".tie5Roanl" from the relative H/DD features
     (press_1 = 0; press_{i+1} = press_i + DD_i; release_i = press_i + H_i).
  2. Apply a random per-rep phase offset (the "sliding" part of the
     sliding-window protocol) representing the arbitrary alignment of a
     hardware poller's clock relative to the true keystroke event.
  3. Quantize every timestamp to the nearest multiple of the target
     interval (20 ms or 30 ms), which is what a poller sampling at that
     rate would actually report.
  4. Recompute H, DD, UD from the quantized timestamps.

This produces genuine quantization jitter and interpolation artefacts,
rather than naive point-dropping, which is what your methodology
section commits to.

Two modelling choices, and why
------------------------------
Nearest-tick rounding. A real poller reports an event at the NEXT tick
(a ceiling), not the nearest one. Because the phase offset is uniform over
one full poll interval, the two are equivalent for everything measured
here: ceiling is nearest-tick rounding with all timestamps shifted by
dt/2, and a constant shift is absorbed by the uniform phase. H, DD and UD
are differences of timestamps, so their distribution is identical under
either rule.

Floor of dt/2 on collapsed intervals. When a key's press and release (or
two consecutive presses) land on the same tick, the observed interval is
0. The true interval is then known only to be shorter than one poll
interval, so it is replaced by the midpoint dt/2 rather than 0; this
avoids physically impossible zero-length holds. It affects only intervals
shorter than dt, so it is rare at 20-30ms and increasingly common at the
coarse exploratory resolutions (holds are typically ~100ms).
"""
import numpy as np
import pandas as pd

KEYS = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]


def _h_cols():
    return [f"H.{k}" for k in KEYS]


def _dd_cols():
    return [f"DD.{KEYS[i]}.{KEYS[i+1]}" for i in range(len(KEYS) - 1)]


def _ud_cols():
    return [f"UD.{KEYS[i]}.{KEYS[i+1]}" for i in range(len(KEYS) - 1)]


def _reconstruct_timestamps(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (press_times, release_times) arrays, length 11, seconds, press_1=0."""
    h = row[_h_cols()].to_numpy(dtype=float)
    dd = row[_dd_cols()].to_numpy(dtype=float)

    press = np.zeros(len(KEYS))
    for i in range(1, len(KEYS)):
        press[i] = press[i - 1] + dd[i - 1]
    release = press + h
    return press, release


def _quantize(timestamps: np.ndarray, dt: float, phase_offset: float) -> np.ndarray:
    """Simulate a poller sampling every dt seconds, with a random phase offset,
    by snapping each timestamp to the nearest poll tick."""
    shifted = timestamps + phase_offset
    quantized = np.round(shifted / dt) * dt - phase_offset
    return quantized


def downsample_row(row: pd.Series, dt: float, rng: np.random.Generator) -> pd.Series:
    """Return a new row with H/DD/UD recomputed as if sampled at interval dt (seconds).
    dt=None (or <= the ~1ms baseline) returns the row unchanged (control group)."""
    if dt is None or dt <= 0.001:
        return row.copy()

    press, release = _reconstruct_timestamps(row)

    # random phase offset per rep: this is the "sliding" in sliding-window
    phase_offset = rng.uniform(0, dt)

    q_press = _quantize(press, dt, phase_offset)
    q_release = _quantize(release, dt, phase_offset)

    # Guard against degenerate quantization collapsing press==release (H=0)
    # or press_i+1 == press_i (DD=0): floor at half a poll tick.
    h_new = np.maximum(q_release - q_press, dt / 2)
    dd_new = np.maximum(np.diff(q_press), dt / 2)
    ud_new = dd_new - h_new[:-1]

    new_row = row.copy()
    for i, k in enumerate(KEYS):
        new_row[f"H.{k}"] = h_new[i]
    for i in range(len(KEYS) - 1):
        new_row[_dd_cols()[i]] = dd_new[i]
        new_row[_ud_cols()[i]] = ud_new[i]
    return new_row


def downsample_dataframe(df: pd.DataFrame, dt: float, seed: int = 0) -> pd.DataFrame:
    """Vectorised equivalent of applying downsample_row to every row in order
    (bit-identical output for the same seed, ~500x faster)."""
    if dt is None or dt <= 0.001:
        return df.copy()

    rng = np.random.default_rng(seed)
    h = df[_h_cols()].to_numpy(dtype=float)
    dd = df[_dd_cols()].to_numpy(dtype=float)

    press = np.concatenate([np.zeros((len(df), 1)), np.cumsum(dd, axis=1)], axis=1)
    release = press + h

    # one phase offset per rep, drawn in row order (same stream as downsample_row)
    phase_offset = rng.uniform(0, dt, size=len(df))[:, None]
    q_press = _quantize(press, dt, phase_offset)
    q_release = _quantize(release, dt, phase_offset)

    h_new = np.maximum(q_release - q_press, dt / 2)
    dd_new = np.maximum(np.diff(q_press, axis=1), dt / 2)

    out = df.copy()
    out[_h_cols()] = h_new
    out[_dd_cols()] = dd_new
    out[_ud_cols()] = dd_new - h_new[:, :-1]
    return out


def build_experimental_groups(df: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Table 1 from the proposal: control (baseline), 20ms, 30ms."""
    return {
        "control_1ms": df.copy(),
        "test_20ms": downsample_dataframe(df, dt=0.020, seed=seed),
        "test_30ms": downsample_dataframe(df, dt=0.030, seed=seed + 1),
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")
    groups = build_experimental_groups(df)

    for name, g in groups.items():
        g.to_csv(f"data/processed/{name}.csv", index=False)
        print(f"{name}: wrote {len(g)} rows")

    # Validation plot: one subject's first rep, control vs 20ms vs 30ms,
    # for a single feature (H.t) across many reps to see distribution shift
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    subj = df["subject"].iloc[0]
    for ax, (name, g) in zip(axes, groups.items()):
        vals = g[g["subject"] == subj]["H.t"]
        ax.hist(vals, bins=20)
        ax.set_title(name)
        ax.set_xlabel("H.t (seconds)")
    axes[0].set_ylabel("count")
    fig.suptitle(f"Subject {subj}: H.t distribution across sampling groups")
    fig.tight_layout()
    fig.savefig("results/downsample_validation_H_t.png", dpi=120)
    print("Saved results/downsample_validation_H_t.png")

    # Numeric check: mean absolute distortion vs baseline, should grow with dt
    for name, g in groups.items():
        if name == "control_1ms":
            continue
        diff = (g["H.t"] - df["H.t"]).abs().mean()
        print(f"{name}: mean |H.t - baseline| = {diff*1000:.2f} ms")