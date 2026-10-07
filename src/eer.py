"""
Step 3: EER (Equal Error Rate) calculation utility.

EER is the point on the DET curve where FAR (False Acceptance Rate) equals
FRR (False Rejection Rate). scikit-learn has no built-in EER function, so
this implements it from raw anomaly scores.

Convention used here: HIGHER score = more anomalous (more likely an impostor).
If your model outputs "normality" scores instead (higher = more genuine),
negate them before calling these functions, or pass invert_scores=True.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def far_frr_at_thresholds(genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                           thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    A sample is REJECTED (flagged as impostor) if score > threshold.
    FAR = fraction of impostor scores that fall BELOW threshold (wrongly accepted)
    FRR = fraction of genuine scores that fall ABOVE threshold (wrongly rejected)
    """
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)

    far = np.array([(impostor_scores <= t).mean() for t in thresholds])
    frr = np.array([(genuine_scores > t).mean() for t in thresholds])
    return far, frr


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                 invert_scores: bool = False, n_thresholds: int = 2000) -> dict:
    """
    Returns a dict with:
      eer: the equal error rate (0-1)
      threshold: the score threshold at which FAR == FRR
      far_curve, frr_curve, thresholds: full curves, useful for plotting DET curve
    """
    genuine_scores = np.asarray(genuine_scores, dtype=float)
    impostor_scores = np.asarray(impostor_scores, dtype=float)

    if invert_scores:
        genuine_scores = -genuine_scores
        impostor_scores = -impostor_scores

    lo = min(genuine_scores.min(), impostor_scores.min())
    hi = max(genuine_scores.max(), impostor_scores.max())
    thresholds = np.linspace(lo, hi, n_thresholds)

    far, frr = far_frr_at_thresholds(genuine_scores, impostor_scores, thresholds)

    # far is increasing in threshold, frr is decreasing (roughly) -> find crossing
    diff = far - frr
    # look for a sign change
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # no crossing found in range (degenerate case, e.g. perfect separation)
        # fall back to the threshold minimizing |far - frr|
        idx = np.argmin(np.abs(diff))
        eer = (far[idx] + frr[idx]) / 2
        eer_threshold = thresholds[idx]
    else:
        i = sign_changes[0]
        # linear interpolation between thresholds[i] and thresholds[i+1]
        f = interp1d([thresholds[i], thresholds[i + 1]], [diff[i], diff[i + 1]])
        eer_threshold = brentq(f, thresholds[i], thresholds[i + 1])
        far_i = np.interp(eer_threshold, thresholds, far)
        frr_i = np.interp(eer_threshold, thresholds, frr)
        eer = (far_i + frr_i) / 2

    return {
        "eer": float(eer),
        "threshold": float(eer_threshold),
        "far_curve": far,
        "frr_curve": frr,
        "thresholds": thresholds,
    }


if __name__ == "__main__":
    from scipy.stats import norm

    # Self-test with synthetic scores where we KNOW the answer.
    # Two unit-variance normals whose means are d apart cross at the midpoint,
    # so the analytical EER is Phi(-d/2). With 50,000 samples per class the
    # sampling error of the estimate is ~0.2 points, hence the 0.5pt tolerance.
    rng = np.random.default_rng(0)
    n = 50_000
    tolerance = 0.005

    for d in [1.0, 2.0, 3.0]:
        analytical = norm.cdf(-d / 2)
        result = compute_eer(rng.normal(0, 1, n), rng.normal(d, 1, n))
        print(f"Test (means {d:.0f} sigma apart): EER = {result['eer']*100:.2f}%  "
              f"analytical = {analytical*100:.2f}%  threshold = {result['threshold']:.3f}")
        assert abs(result["eer"] - analytical) < tolerance, "EER does not match analytical value"
        assert abs(result["threshold"] - d / 2) < 0.05, "EER threshold is not at the midpoint"

    # Identical distributions -> EER must be 50%
    result2 = compute_eer(rng.normal(0, 1, n), rng.normal(0, 1, n))
    print(f"Test (identical distributions): EER = {result2['eer']*100:.2f}% (analytical 50.00%)")
    assert abs(result2["eer"] - 0.5) < 2 * tolerance

    # Perfectly separated -> EER must be 0%
    result3 = compute_eer(rng.normal(0, 0.1, 1000), rng.normal(10, 0.1, 1000))
    print(f"Test (perfectly separated): EER = {result3['eer']*100:.2f}% (analytical 0.00%)")
    assert result3["eer"] < 0.001

    # invert_scores: "normality" scores (higher = more genuine) give the same EER
    g, i = rng.normal(0, 1, n), rng.normal(2, 1, n)
    assert abs(compute_eer(-g, -i, invert_scores=True)["eer"] - compute_eer(g, i)["eer"]) < 1e-12

    print("All EER self-tests passed.")