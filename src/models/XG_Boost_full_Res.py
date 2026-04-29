"""
XG_Boost_full_Res.py  -  XGBoost spot price forecasting for DK1
================================================================
5 XGBoost models, one per forecast day:
  Day 1: h=1-24    Day 2: h=25-48    Day 3: h=49-72
  Day 4: h=73-96   Day 5: h=97-120

Walk-forward cross-validation with expanding training window.

Outputs (outputs/model/):
  predictions.parquet       -- walk-forward CV predictions
  metrics.csv               -- per-fold MAE
  final_day_models.joblib   -- 5 final models trained on all data

Run:   python src/models/XG_Boost_full_Res.py
Then:  python src/analysis/explainable_ai.py  (to generate all plots)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# ── Configuration ──────────────────────────────────────────────────────────────
REGION    = "DK1_west"
TARGET_COL = "DayAheadPriceEUR"
ROOT       = Path(__file__).parent.parent.parent
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs/model"

INITIAL_TRAIN_MONTHS = 12
STEP_MONTHS          = 3

_XGB_BASE = dict(
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)
XGB_WF    = dict(**_XGB_BASE, n_estimators=300)
XGB_FINAL = dict(**_XGB_BASE, n_estimators=500)

DAY_GROUPS = {1: (1, 24), 2: (25, 48), 3: (49, 72), 4: (73, 96), 5: (97, 120)}

WEATHER_FEATURES = [
    "fcst_wind_speed_10m",     "fcst_wind_direction_10m",
    "fcst_wind_speed_100m",    "fcst_wind_direction_100m",
    "fcst_shortwave_radiation","fcst_cloud_cover",
    "fcst_temperature_2m",     "fcst_pressure_msl",
]
TIME_FEATURES = [
    "horizon_h",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
]
PRICE_LAG_FEATURES = [
    "price_lag_24h", "price_lag_48h", "price_lag_168h",
    "price_rolling_24h_mean",
]
FEATURES = TIME_FEATURES + WEATHER_FEATURES + PRICE_LAG_FEATURES

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


# ── Data preparation ───────────────────────────────────────────────────────────

def load_hourly_prices() -> pd.Series:
    raw = (
        pd.read_csv(DATA_DIR / "day_ahead_prices_dk1_raw.csv", parse_dates=["TimeDK"])
        .set_index("TimeDK")
        .sort_index()
    )
    return raw["DayAheadPriceEUR"].resample("h").mean()


def build_price_lags(df: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Price lags anchored to issue_time — always known at forecast time, no leakage."""
    price_map = prices.to_dict()
    unique_ts = pd.to_datetime(df["issue_time"].unique())

    rows = {}
    for t in unique_ts:
        past_24 = [price_map.get(t - pd.Timedelta(hours=h), np.nan) for h in range(1, 25)]
        rows[t] = {
            "price_lag_24h":          price_map.get(t - pd.Timedelta(hours=24),  np.nan),
            "price_lag_48h":          price_map.get(t - pd.Timedelta(hours=48),  np.nan),
            "price_lag_168h":         price_map.get(t - pd.Timedelta(hours=168), np.nan),
            "price_rolling_24h_mean": np.nanmean(past_24),
        }

    lag_df = pd.DataFrame.from_dict(rows, orient="index")
    lag_df.index.name = "issue_time"
    return df.merge(lag_df.reset_index(), on="issue_time", how="left")


def prepare_dataset() -> pd.DataFrame:
    print("Loading forecast dataset...")
    df = pd.read_parquet(DATA_DIR / "forecast_dataset.parquet")
    df = df[df["region"] == REGION].copy()
    print(f"  {len(df):,} rows | region={REGION}")

    print("Loading prices (resampling to hourly)...")
    prices = load_hourly_prices()
    print(f"  Price range: {prices.index.min().date()} to {prices.index.max().date()}")

    price_df = (
        prices
        .rename(TARGET_COL)
        .reset_index()
        .rename(columns={"TimeDK": "target_time"})
    )
    df = df.merge(price_df, on="target_time", how="inner")
    print(f"  After price join: {len(df):,} rows")

    print("Adding price lag features...")
    df = build_price_lags(df, prices)
    df = df.dropna(subset=[TARGET_COL] + PRICE_LAG_FEATURES).reset_index(drop=True)
    print(f"  After lag NaN drop: {len(df):,} rows")
    print(f"  Usable range: {df['issue_time'].min().date()} to {df['issue_time'].max().date()}")

    return df.sort_values("issue_time").reset_index(drop=True)


# ── Model training & prediction ────────────────────────────────────────────────

def fit_day_models(train_df: pd.DataFrame, params: dict) -> dict:
    models = {}
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        mask = train_df["horizon_h"].between(h_lo, h_hi)
        m = xgb.XGBRegressor(**params)
        m.fit(train_df.loc[mask, FEATURES], train_df.loc[mask, TARGET_COL])
        models[day] = m
    return models


def predict_day_models(models: dict, test_df: pd.DataFrame) -> np.ndarray:
    preds = np.full(len(test_df), np.nan)
    h = test_df["horizon_h"].values
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        mask = (h >= h_lo) & (h <= h_hi)
        if mask.any():
            preds[mask] = models[day].predict(test_df[mask][FEATURES])
    return preds


# ── Walk-forward validation ────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame):
    min_t = df["issue_time"].min()
    max_t = df["issue_time"].max()

    folds, t = [], min_t + pd.DateOffset(months=INITIAL_TRAIN_MONTHS)
    while t + pd.DateOffset(months=STEP_MONTHS) <= max_t:
        folds.append((pd.Timestamp(t), pd.Timestamp(t + pd.DateOffset(months=STEP_MONTHS))))
        t += pd.DateOffset(months=STEP_MONTHS)
    print(f"  {len(folds)} folds | initial={INITIAL_TRAIN_MONTHS}m, step={STEP_MONTHS}m")

    all_preds    = []
    fold_metrics = []

    for i, (train_end, test_end) in enumerate(folds):
        train = df[df["issue_time"] <  train_end]
        test  = df[(df["issue_time"] >= train_end) & (df["issue_time"] < test_end)]
        if train.empty or test.empty:
            continue

        models = fit_day_models(train, XGB_WF)
        test_r = test.reset_index(drop=True)
        preds  = predict_day_models(models, test_r)

        result = test_r[["issue_time", "target_time", "horizon_h", TARGET_COL]].copy()
        result["predicted"] = preds
        result["fold"]      = i
        result["fold_end"]  = train_end
        all_preds.append(result)

        row     = {"fold": i, "fold_end": train_end, "n_train": len(train)}
        summary = []
        for day, (h_lo, h_hi) in DAY_GROUPS.items():
            m = result["horizon_h"].between(h_lo, h_hi)
            if m.any():
                mae = mean_absolute_error(result.loc[m, TARGET_COL], result.loc[m, "predicted"])
                row[f"day{day}_mae"] = mae
                summary.append(f"d{day}={mae:.1f}")
        fold_metrics.append(row)
        print(f"  Fold {i+1} | train_end={train_end.date()} n={len(train):,} | {', '.join(summary)}")

    return pd.concat(all_preds, ignore_index=True), pd.DataFrame(fold_metrics)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("XGBoost  --  DK1 spot price forecasting")
    print("=" * 60)

    # 1. Prepare data
    df = prepare_dataset()
    print(f"\n  Features : {len(FEATURES)}")
    print(f"  Target   : {TARGET_COL}")

    # 2. Walk-forward CV
    print("\nWalk-forward validation...")
    preds_df, metrics_df = walk_forward(df)

    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    preds_df.to_parquet(OUTPUT_DIR / "predictions.parquet", index=False)

    print("\nOverall MAE (mean across folds):")
    for day in range(1, 6):
        col = f"day{day}_mae"
        if col in metrics_df:
            print(f"  Day {day} (h{(day-1)*24+1:3d}-{day*24}): "
                  f"{metrics_df[col].mean():.2f} EUR/MWh")

    # 3. Final models on all data
    print("\nTraining final models on full dataset...")
    final_models = {}
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        mask = df["horizon_h"].between(h_lo, h_hi)
        m = xgb.XGBRegressor(**XGB_FINAL)
        m.fit(df.loc[mask, FEATURES], df.loc[mask, TARGET_COL])
        final_models[day] = m
        print(f"  Day {day}: {mask.sum():,} rows")

    model_path = OUTPUT_DIR / "final_day_models.joblib"
    joblib.dump(final_models, model_path)
    print(f"  Models saved -> {model_path}")

    print(f"\nModel outputs saved to {OUTPUT_DIR}/")
    print("Run src/analysis/explainable_ai.py to generate all plots and SHAP exports.")


if __name__ == "__main__":
    main()
