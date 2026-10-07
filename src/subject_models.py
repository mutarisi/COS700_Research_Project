"""
Shared protocol + model code for one subject (Option A protocol).

Everything the experiment scripts have in common lives here, so a change to
the protocol or to a model is made once:

  Protocol   training_reps()  - a subject's control_1ms training reps (first 200)
             split_subject()  - genuine test reps (last 200) + impostor reps
                                (first 5 of every other subject) from one group
  Models     fit_isolation_forest(), fit_lstm_autoencoder()
             fit_subject_models() - both, trained ONCE on control_1ms
  Scoring    score_samples()  - Isolation Forest, LSTM Autoencoder and Hybrid
                                anomaly scores (higher = more anomalous) for
                                any rows, at any sampling resolution

LSTM sequence framing: each rep of ".tie5Roanl" involves 11 keys. The 31 flat
H/DD/UD columns become a sequence of 11 timesteps, 3 features each:
[H_i, DD_in_i, UD_in_i].
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
from downsample import KEYS

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

FEATURE_PREFIXES = ("H.", "DD.", "UD.")
GROUP_FILES = {
    "control_1ms": "data/processed/control_1ms.csv",
    "test_20ms": "data/processed/test_20ms.csv",
    "test_30ms": "data/processed/test_30ms.csv",
}
MODEL_NAMES = ["Isolation Forest", "LSTM Autoencoder", "Hybrid"]

N_TRAIN_REPS = 200      # sessions 1-4: training; sessions 5-8: genuine test
N_IMPOSTOR_REPS = 5     # first 5 reps of every other subject
N_TIMESTEPS = len(KEYS)
N_FEATURES = 3


# ---------- protocol ----------

def load_groups() -> dict:
    """group name -> full dataframe (all subjects, one sampling rate)."""
    return {name: pd.read_csv(path) for name, path in GROUP_FILES.items()}


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def training_reps(control_df: pd.DataFrame, subject: str) -> pd.DataFrame:
    """The subject's training data: control_1ms ONLY, sessions 1-4."""
    subj_df = (control_df[control_df["subject"] == subject]
               .sort_values(["sessionIndex", "rep"]))
    assert len(subj_df) == 400, f"{subject} has {len(subj_df)} reps"
    return subj_df.iloc[:N_TRAIN_REPS]


def split_subject(df: pd.DataFrame, subject: str) -> tuple:
    """Return (genuine_test_df, impostor_df) for one subject from one group's
    dataframe, i.e. at THAT group's sampling resolution."""
    subj_df = df[df["subject"] == subject].sort_values(["sessionIndex", "rep"])
    genuine_test_df = subj_df.iloc[N_TRAIN_REPS:]

    others = df[df["subject"] != subject]
    impostor_df = (others.sort_values(["subject", "sessionIndex", "rep"])
                         .groupby("subject").head(N_IMPOSTOR_REPS))
    return genuine_test_df, impostor_df


# ---------- LSTM Autoencoder building blocks ----------

def rows_to_sequences(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    seq = np.zeros((n, N_TIMESTEPS, N_FEATURES), dtype=float)
    for i, key in enumerate(KEYS):
        seq[:, i, 0] = df[f"H.{key}"].to_numpy()
        if i > 0:
            prev = KEYS[i - 1]
            seq[:, i, 1] = df[f"DD.{prev}.{key}"].to_numpy()
            seq[:, i, 2] = df[f"UD.{prev}.{key}"].to_numpy()
    return seq


def build_autoencoder(latent_dim: int = 16):
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(N_TIMESTEPS, N_FEATURES))
    encoded = layers.LSTM(latent_dim, activation="tanh")(inputs)
    repeated = layers.RepeatVector(N_TIMESTEPS)(encoded)
    decoded = layers.LSTM(latent_dim, activation="tanh", return_sequences=True)(repeated)
    outputs = layers.TimeDistributed(layers.Dense(N_FEATURES))(decoded)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=(1, 2))


def scale_sequences(scaler: StandardScaler, df: pd.DataFrame) -> np.ndarray:
    """Rows -> scaled (n, 11, 3) sequences, using a scaler fitted on training data."""
    X = rows_to_sequences(df)
    n = X.shape[0]
    return scaler.transform(X.reshape(n, -1)).reshape(n, N_TIMESTEPS, N_FEATURES)


# ---------- training (once per subject) ----------

def fit_isolation_forest(train_df: pd.DataFrame, feature_cols: list, seed: int = 42) -> tuple:
    """Returns (model, train_scores)."""
    X_train = train_df[feature_cols].to_numpy()
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=seed)
    model.fit(X_train)
    return model, -model.score_samples(X_train)


def fit_lstm_autoencoder(train_df: pd.DataFrame, seed: int = 42, epochs: int = 60,
                          verbose_fit: int = 0) -> tuple:
    """Returns (model, scaler, train_scores)."""
    import tensorflow as tf

    X_train_raw = rows_to_sequences(train_df)
    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(len(X_train_raw), -1))
    X_train = scale_sequences(scaler, train_df)

    tf.keras.utils.set_random_seed(seed)
    model = build_autoencoder()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5,
                                                    restore_best_weights=True)
    model.fit(X_train, X_train, epochs=epochs, batch_size=16, verbose=verbose_fit,
              callbacks=[early_stop])
    return model, scaler, reconstruction_error(model, X_train)


@dataclass
class SubjectModels:
    feature_cols: list
    if_model: IsolationForest
    lstm_model: object
    scaler: StandardScaler
    if_train_scores: np.ndarray
    lstm_train_scores: np.ndarray


def fit_subject_models(train_df: pd.DataFrame, feature_cols: list,
                        seed: int = 42, epochs: int = 60) -> SubjectModels:
    """train_df: the subject's control_1ms training reps (see training_reps)."""
    if_model, if_train_scores = fit_isolation_forest(train_df, feature_cols, seed)
    lstm_model, scaler, lstm_train_scores = fit_lstm_autoencoder(train_df, seed, epochs)
    return SubjectModels(feature_cols, if_model, lstm_model, scaler,
                         if_train_scores, lstm_train_scores)


# ---------- scoring ----------

def zscore(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Normalise scores by the subject's own TRAINING-score distribution."""
    mean, std = train_scores.mean(), train_scores.std()
    std = std if std > 1e-8 else 1e-8
    return (scores - mean) / std


def fuse_scores(models: SubjectModels, if_scores: np.ndarray, lstm_scores: np.ndarray,
                 alpha: float = 0.10) -> np.ndarray:
    """Hybrid score. alpha = Isolation Forest weight (0.10 = 10% IF / 90% LSTM,
    Shaheen & Alomari's formula)."""
    return (alpha * zscore(models.if_train_scores, if_scores)
            + (1 - alpha) * zscore(models.lstm_train_scores, lstm_scores))


def score_samples(models: SubjectModels, df: pd.DataFrame, alpha: float = 0.10) -> dict:
    """Score every row of df. Returns model name -> array of anomaly scores."""
    if_scores = -models.if_model.score_samples(df[models.feature_cols].to_numpy())
    lstm_scores = reconstruction_error(models.lstm_model, scale_sequences(models.scaler, df))
    return {
        "Isolation Forest": if_scores,
        "LSTM Autoencoder": lstm_scores,
        "Hybrid": fuse_scores(models, if_scores, lstm_scores, alpha),
    }
