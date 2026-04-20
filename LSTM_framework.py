import os
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================
# Configuration
# =========================

@dataclass
class Config:
    csv_path: str = "data.csv"
    time_col: str = "time"
    feature_cols: tuple = ("wind", "temp", "solar")
    target_col: str = "price"

    # sequence settings
    sequence_length: int = 24     # use past 24 rows to predict next target
    forecast_horizon: int = 1     # 1 = predict next row; 24 = predict 24 steps ahead

    # train / val / test split (time-based)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # model / training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dropout_rate: float = 0.2

    # output
    artifacts_dir: str = "artifacts"
    random_seed: int = 42


# =========================
# Reproducibility
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# Data loading / validation
# =========================

def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    required_cols = [cfg.time_col, *cfg.feature_cols, cfg.target_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).reset_index(drop=True)

    # numeric conversion
    for col in [*cfg.feature_cols, cfg.target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # simple missing-value handling
    df[[*cfg.feature_cols, cfg.target_col]] = (
        df[[*cfg.feature_cols, cfg.target_col]]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    return df


# =========================
# Feature engineering
# =========================

def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()

    out["hour"] = out[time_col].dt.hour
    out["dayofweek"] = out[time_col].dt.dayofweek
    out["month"] = out[time_col].dt.month

    # cyclical encoding
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out


# =========================
# Train/val/test split
# =========================

def time_split(df: pd.DataFrame, cfg: Config):
    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the splits is empty. Check dataset size and split ratios.")

    return train_df, val_df, test_df


# =========================
# Scaling
# =========================

def fit_scalers(train_df: pd.DataFrame, feature_cols, target_col):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(train_df[list(feature_cols)])
    y_scaler.fit(train_df[[target_col]])

    return x_scaler, y_scaler


def transform_df(df: pd.DataFrame, feature_cols, target_col, x_scaler, y_scaler):
    x = x_scaler.transform(df[list(feature_cols)])
    y = y_scaler.transform(df[[target_col]])
    return x, y


# =========================
# Sequence creation
# =========================

def make_sequences(X, y, seq_len: int, horizon: int):
    X_seq, y_seq = [], []

    max_start = len(X) - seq_len - horizon + 1
    for i in range(max_start):
        x_window = X[i:i + seq_len]
        y_target = y[i + seq_len + horizon - 1]
        X_seq.append(x_window)
        y_seq.append(y_target)

    return np.array(X_seq), np.array(y_seq)


# =========================
# Model
# =========================

def build_model(input_shape, cfg: Config):
    model = Sequential([
        LSTM(cfg.lstm_units_1, return_sequences=True, input_shape=input_shape),
        Dropout(cfg.dropout_rate),
        LSTM(cfg.lstm_units_2),
        Dropout(cfg.dropout_rate),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


# =========================
# Metrics
# =========================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_predictions(y_true, y_pred, prefix=""):
    return {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(rmse(y_true, y_pred)),
    }


# =========================
# Main pipeline
# =========================

def run_pipeline(cfg: Config):
    set_seed(cfg.random_seed)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(cfg)
    df = add_time_features(df, cfg.time_col)

    # final feature set
    model_feature_cols = [
        *cfg.feature_cols,
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
    ]

    print("Splitting data...")
    train_df, val_df, test_df = time_split(df, cfg)

    print("Fitting scalers...")
    x_scaler, y_scaler = fit_scalers(train_df, model_feature_cols, cfg.target_col)

    X_train_raw, y_train_raw = transform_df(train_df, model_feature_cols, cfg.target_col, x_scaler, y_scaler)
    X_val_raw, y_val_raw = transform_df(val_df, model_feature_cols, cfg.target_col, x_scaler, y_scaler)
    X_test_raw, y_test_raw = transform_df(test_df, model_feature_cols, cfg.target_col, x_scaler, y_scaler)

    print("Creating sequences...")
    X_train, y_train = make_sequences(X_train_raw, y_train_raw, cfg.sequence_length, cfg.forecast_horizon)
    X_val, y_val = make_sequences(X_val_raw, y_val_raw, cfg.sequence_length, cfg.forecast_horizon)
    X_test, y_test = make_sequences(X_test_raw, y_test_raw, cfg.sequence_length, cfg.forecast_horizon)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "Not enough rows to build sequences. "
            "Try reducing sequence_length or forecast_horizon."
        )

    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape:   {X_val.shape}, {y_val.shape}")
    print(f"Test shape:  {X_test.shape}, {y_test.shape}")

    print("Building model...")
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), cfg=cfg)

    checkpoint_path = os.path.join(cfg.artifacts_dir, "best_lstm.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True)
    ]

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("Evaluating...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    metrics = evaluate_predictions(y_test_inv, y_pred_inv, prefix="test_")
    print("Test metrics:", metrics)

    # save artifacts
    print("Saving artifacts...")
    joblib.dump(x_scaler, os.path.join(cfg.artifacts_dir, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(cfg.artifacts_dir, "y_scaler.pkl"))

    with open(os.path.join(cfg.artifacts_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(cfg.artifacts_dir, "training_history.csv"), index=False)

    results_df = pd.DataFrame({
        "actual": y_test_inv,
        "predicted": y_pred_inv
    })
    results_df.to_csv(os.path.join(cfg.artifacts_dir, "test_predictions.csv"), index=False)

    print(f"Done. Artifacts saved in: {cfg.artifacts_dir}")
    return model, history_df, results_df, metrics


# =========================
# Inference helper
# =========================

def predict_from_recent_rows(recent_df: pd.DataFrame, cfg: Config):
    """
    Predict from the most recent sequence_length rows.
    recent_df must already contain:
      time, wind, temp, solar
    and must have at least cfg.sequence_length rows.
    """
    x_scaler = joblib.load(os.path.join(cfg.artifacts_dir, "x_scaler.pkl"))
    y_scaler = joblib.load(os.path.join(cfg.artifacts_dir, "y_scaler.pkl"))
    model = tf.keras.models.load_model(os.path.join(cfg.artifacts_dir, "best_lstm.keras"))

    recent_df = recent_df.copy()
    recent_df[cfg.time_col] = pd.to_datetime(recent_df[cfg.time_col], errors="coerce")
    recent_df = recent_df.sort_values(cfg.time_col).reset_index(drop=True)
    recent_df = add_time_features(recent_df, cfg.time_col)

    model_feature_cols = [
        *cfg.feature_cols,
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
    ]

    for col in model_feature_cols:
        if col not in recent_df.columns:
            raise ValueError(f"Missing column in recent_df: {col}")

    if len(recent_df) < cfg.sequence_length:
        raise ValueError(f"Need at least {cfg.sequence_length} rows for prediction.")

    latest_window = recent_df.iloc[-cfg.sequence_length:][model_feature_cols]
    X = x_scaler.transform(latest_window)
    X = np.expand_dims(X, axis=0)

    pred_scaled = model.predict(X, verbose=0)
    pred = y_scaler.inverse_transform(pred_scaled)[0, 0]
    return float(pred)


# =========================
# Run
# =========================

if __name__ == "__main__":
    cfg = Config(
        csv_path="data.csv",
        time_col="time",
        feature_cols=("wind", "temp", "solar"),
        target_col="price",
        sequence_length=24,
        forecast_horizon=1,
        epochs=30,
        batch_size=32
    )

    model, history_df, results_df, metrics = run_pipeline(cfg)
    print(metrics)