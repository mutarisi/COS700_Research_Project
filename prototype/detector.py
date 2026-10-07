"""
Detector for the real-time alerting prototype.

Turns ONE login attempt (the press/release timestamps of the 11 keys of
".tie5Roanl") into a decision: accept, or raise an account-compromise alert.

It reuses the study's code rather than re-implementing it:
  - the Isolation Forest is the one in src/subject_models.py, trained the same
    way (once per account, on full-resolution enrolment attempts only);
  - coarser polling is the quantisation in src/downsample.py.

What is new here is the ALERT THRESHOLD. The study reports EER, whose threshold
is chosen afterwards using impostor scores. A live system has no impostor
data, so the threshold is set from the account holder's own enrolment attempts:
the 90th percentile of their anomaly scores, i.e. the rule flags roughly the
most unusual 10% of the account holder's own typing. Because coarser polling
raises every score, the enrolment attempts are first quantised to the polling
interval the client is using, so each interval gets its own threshold.
check_alert_rates.py measures the error rates this rule gives.

Accounts are either the 51 CMU subjects (enrolled on their first 200
repetitions, exactly as in the study) or people enrolled live through the
login page (prototype/profiles/<name>.json).
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from downsample import KEYS, downsample_dataframe, _h_cols, _dd_cols, _ud_cols
from subject_models import (get_feature_columns, training_reps, split_subject,
                            fit_isolation_forest)

RAW_CSV = os.path.join(ROOT, "data", "raw", "DSL-StrongPasswordData.csv")
PROFILES_DIR = os.path.join(HERE, "profiles")

# Key names used on the wire, in the same order as downsample.KEYS
PASSWORD_KEYS = [".", "t", "i", "e", "5", "R", "o", "a", "n", "l", "Enter"]
assert len(PASSWORD_KEYS) == len(KEYS)

THRESHOLD_PERCENTILE = 90.0
N_CALIBRATION_DRAWS = 5        # phase-jitter draws of the enrolment data per threshold
CALIBRATION_SEED = 1000
MIN_ENROL_ATTEMPTS = 15        # a live account can be used once it has this many
TARGET_ENROL_ATTEMPTS = 30     # what the login page asks for
ACCOUNT_NAME = re.compile(r"^[A-Za-z0-9_-]{1,20}$")


class WrongPassword(Exception):
    """The key events are not exactly one clean entry of the password."""


def pair_key_events(events: list) -> tuple:
    """events: (key, "down" | "up", time in ms) in arrival order.
    Returns (press, release) arrays in seconds, one entry per password key."""
    downs = [(key, t) for key, kind, t in events if kind == "down"]
    ups = [(key, t) for key, kind, t in events if kind == "up"]
    if [key for key, _ in downs] != PASSWORD_KEYS or sorted(k for k, _ in ups) != sorted(PASSWORD_KEYS):
        raise WrongPassword
    up_time = dict(ups)
    press = np.array([t for _, t in downs], dtype=float) / 1000.0
    release = np.array([up_time[key] for key in PASSWORD_KEYS], dtype=float) / 1000.0
    if not (np.all(np.isfinite(press)) and np.all(np.isfinite(release))) or np.any(release < press):
        raise WrongPassword
    return press, release


def features_from_timestamps(press: np.ndarray, release: np.ndarray, poll_ms: float) -> pd.DataFrame:
    """One-row dataframe of H/DD/UD features (seconds), built the way
    downsample.py builds them: at a polling interval above 1ms, an interval
    that collapsed onto a single poll tick is floored at half the interval."""
    dt = poll_ms / 1000.0
    h = release - press
    dd = np.diff(press)
    if dt > 0.001:
        h = np.maximum(h, dt / 2)
        dd = np.maximum(dd, dt / 2)
    ud = dd - h[:-1]
    values = dict(zip(_h_cols(), h)) | dict(zip(_dd_cols(), dd)) | dict(zip(_ud_cols(), ud))
    return pd.DataFrame([values])


class Profile:
    """One account's enrolled typing profile and its alert thresholds."""

    def __init__(self, train_df: pd.DataFrame, feature_cols: list,
                 percentile: float = THRESHOLD_PERCENTILE, seed: int = 42):
        self.train_df = train_df
        self.feature_cols = feature_cols
        self.percentile = percentile
        self.model, self.train_scores = fit_isolation_forest(train_df, feature_cols, seed)
        self._thresholds = {}

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Anomaly score per row (higher = less like the account holder)."""
        return -self.model.score_samples(df[self.feature_cols].to_numpy())

    def threshold(self, poll_ms: float) -> float:
        if poll_ms not in self._thresholds:
            if poll_ms <= 1:
                scores = self.train_scores
            else:
                scores = np.concatenate([
                    self.score(downsample_dataframe(self.train_df, poll_ms / 1000.0,
                                                    seed=CALIBRATION_SEED + i))
                    for i in range(N_CALIBRATION_DRAWS)])
            self._thresholds[poll_ms] = float(np.percentile(scores, self.percentile))
        return self._thresholds[poll_ms]


class Detector:
    def __init__(self, raw_csv: str = RAW_CSV, profiles_dir: str = PROFILES_DIR,
                 percentile: float = THRESHOLD_PERCENTILE):
        self.data = pd.read_csv(raw_csv)
        self.feature_cols = get_feature_columns(self.data)
        self.subjects = sorted(self.data["subject"].unique())
        self.profiles_dir = profiles_dir
        self.percentile = percentile
        self._profiles = {}

    # ---------- accounts ----------

    def _live_path(self, account: str) -> str:
        return os.path.join(self.profiles_dir, f"{account}.json")

    def _live_attempts(self, account: str) -> list:
        path = self._live_path(account)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def live_accounts(self) -> list:
        if not os.path.isdir(self.profiles_dir):
            return []
        return sorted(name[:-5] for name in os.listdir(self.profiles_dir) if name.endswith(".json"))

    def accounts(self) -> list:
        """Every account that can be logged into, live-enrolled ones first."""
        live = [a for a in self.live_accounts()
                if len(self._live_attempts(a)) >= MIN_ENROL_ATTEMPTS]
        return live + self.subjects

    def profile(self, account: str) -> Profile:
        """The account's profile, trained on first use. Raises KeyError for an
        unknown account or one without enough enrolment attempts."""
        if account not in self._profiles:
            if account in self.subjects:
                train_df = training_reps(self.data, account)
            else:
                attempts = self._live_attempts(account) if ACCOUNT_NAME.match(account) else []
                if len(attempts) < MIN_ENROL_ATTEMPTS:
                    raise KeyError(account)
                train_df = pd.concat(
                    [features_from_timestamps(np.array(a["press"]), np.array(a["release"]), 1)
                     for a in attempts], ignore_index=True)
            self._profiles[account] = Profile(train_df, self.feature_cols, self.percentile)
        return self._profiles[account]

    def replay_pool(self, account: str) -> tuple:
        """(genuine, impostor) held-out attempts for a CMU account: the study's
        test split (the holder's last 200 repetitions; the first 5 of every other subject)."""
        return split_subject(self.data, account)

    # ---------- live enrolment ----------

    def add_enrolment(self, account: str, press: np.ndarray, release: np.ndarray) -> int:
        """Store one full-resolution enrolment attempt; returns how many the account has."""
        if not ACCOUNT_NAME.match(account) or account in self.subjects:
            raise ValueError("Use 1-20 letters, digits, - or _ (and not a CMU subject id).")
        attempts = self._live_attempts(account)
        attempts.append({"press": (press - press[0]).tolist(),
                         "release": (release - press[0]).tolist()})
        os.makedirs(self.profiles_dir, exist_ok=True)
        with open(self._live_path(account), "w", encoding="utf-8") as f:
            json.dump(attempts, f)
        self._profiles.pop(account, None)      # retrain on next login
        return len(attempts)

    def reset_enrolment(self, account: str) -> None:
        if ACCOUNT_NAME.match(account) and os.path.exists(self._live_path(account)):
            os.remove(self._live_path(account))
        self._profiles.pop(account, None)

    # ---------- the decision ----------

    def assess(self, account: str, press: np.ndarray, release: np.ndarray, poll_ms: float) -> dict:
        """Score one attempt against the account's profile."""
        profile = self.profile(account)
        start = time.perf_counter()
        score = float(profile.score(features_from_timestamps(press, release, poll_ms))[0])
        detection_ms = (time.perf_counter() - start) * 1000.0
        threshold = profile.threshold(poll_ms)
        return {"score": score, "threshold": threshold, "alert": score > threshold,
                "detection_ms": detection_ms}
