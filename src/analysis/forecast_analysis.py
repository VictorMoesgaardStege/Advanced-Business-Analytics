from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_OUTPUT_DIR = ROOT / "outputs" / "model"

REGION = "DK1_west"
PRICE_AREA = "DK1"
TARGET_COL = "DayAheadPriceEUR"
EUR_DKK = 7.46
POST_CRISIS_CUTOFF = pd.Timestamp("2023-01-01")

PREDICTIONS_PATH = MODEL_OUTPUT_DIR / "predictions.parquet"
METRICS_PATH = MODEL_OUTPUT_DIR / "metrics.csv"
MODEL_DATASET_PATH = DATA_DIR / "model_dataset.parquet"

DAY_GROUPS = {
    1: (1, 24),
    2: (25, 48),
    3: (49, 72),
    4: (73, 96),
    5: (97, 120),
}

COLORS = {
    1: "#2563eb",
    2: "#16a34a",
    3: "#d97706",
    4: "#db2777",
    5: "#7c3aed",
}

TIME_FEATURES = [
    "horizon_h",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
]
WEATHER_FEATURES = [
    "fcst_wind_speed_10m",
    "fcst_wind_dir_10m_sin",
    "fcst_wind_dir_10m_cos",
    "fcst_wind_speed_100m",
    "fcst_wind_dir_100m_sin",
    "fcst_wind_dir_100m_cos",
    "fcst_shortwave_radiation",
    "fcst_cloud_cover",
    "fcst_temperature_2m",
    "fcst_pressure_msl",
]
PRICE_LAG_FEATURES = [
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_rolling_24h_mean",
]
FEATURES = TIME_FEATURES + WEATHER_FEATURES + PRICE_LAG_FEATURES

WEATHER_PLOT_VARS = {
    "Wind speed 100m": {
        "fcst": "fcst_wind_speed_100m",
        "actual": "actual_wind_speed_100m",
        "error_var": "wind_speed_120m",
        "unit": "m/s",
        "color": "#0891b2",
    },
    "Solar radiation": {
        "fcst": "fcst_shortwave_radiation",
        "actual": "actual_shortwave_radiation",
        "error_var": "shortwave_radiation",
        "unit": "W/m2",
        "color": "#d97706",
    },
    "Temperature": {
        "fcst": "fcst_temperature_2m",
        "actual": "actual_temperature_2m",
        "error_var": "temperature_2m",
        "unit": "C",
        "color": "#dc2626",
    },
    "Cloud cover": {
        "fcst": "fcst_cloud_cover",
        "actual": "actual_cloud_cover",
        "error_var": "cloud_cover",
        "unit": "%",
        "color": "#64748b",
    },
}


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _unit_scale(unit: str) -> tuple[float, str]:
    unit = unit.lower()
    if unit in {"eur_mwh", "eur/mwh"}:
        return 1.0, "EUR/MWh"
    if unit in {"dkk_mwh", "dkk/mwh"}:
        return EUR_DKK, "DKK/MWh"
    if unit in {"dkk_kwh", "dkk/kwh"}:
        return EUR_DKK / 1000, "DKK/kWh"
    raise ValueError("unit must be one of: 'eur_mwh', 'dkk_mwh', 'dkk_kwh'.")


def _forecast_day_from_horizon(horizon: pd.Series) -> pd.Series:
    day = ((horizon.astype(int) - 1) // 24 + 1).clip(1, 5)
    return day.astype(int)


def _ensure_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def _require_file(path: str | Path) -> Path:
    path = _as_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}.")
    return path


def _resolve_issue_time(
    preds_df: pd.DataFrame,
    issue_time: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    available = pd.to_datetime(preds_df["issue_time"].dropna().unique())
    if len(available) == 0:
        raise ValueError("No issue_time values found in predictions.")
    if issue_time is None:
        if "fold" in preds_df.columns:
            last_fold = preds_df["fold"].max()
            return pd.Timestamp(preds_df.loc[preds_df["fold"].eq(last_fold), "issue_time"].max())
        return pd.Timestamp(max(available))
    requested = pd.Timestamp(issue_time)
    return pd.Timestamp(min(available, key=lambda x: abs(x - requested)))


def _load_prices(
    price_csv: str | Path = DATA_DIR / "day_ahead_prices_dk1_raw.csv",
    price_area: str = PRICE_AREA,
) -> pd.DataFrame:
    price_csv = _require_file(price_csv)
    prices = pd.read_csv(price_csv, parse_dates=["TimeDK"])
    if "PriceArea" in prices.columns:
        prices = prices[prices["PriceArea"].astype(str).eq(price_area)].copy()
    prices = prices.sort_values("TimeDK")
    for col in ["DayAheadPriceEUR", "DayAheadPriceDKK"]:
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    return prices


def load_model_dataset(
    model_dataset: str | Path = MODEL_DATASET_PATH,
    region: str = REGION,
) -> pd.DataFrame:
    """Load the model-ready forecast table used by the XGBoost script."""
    model_dataset = _require_file(model_dataset)
    df = pd.read_parquet(model_dataset)
    df = _ensure_datetime(df, ["issue_time", "target_time"])
    if "region" in df.columns and region is not None:
        df = df[df["region"].astype(str).eq(region)].copy()
    return df.sort_values(["issue_time", "horizon_h"]).reset_index(drop=True)


def load_predictions(predictions_path: str | Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load walk-forward out-of-sample predictions."""
    predictions_path = _require_file(predictions_path)
    preds = pd.read_parquet(predictions_path)
    preds = _ensure_datetime(preds, ["issue_time", "target_time", "fold_end"])
    if "forecast_day" not in preds.columns:
        preds["forecast_day"] = _forecast_day_from_horizon(preds["horizon_h"])
    return preds.sort_values(["issue_time", "horizon_h"]).reset_index(drop=True)


def load_metrics(metrics_path: str | Path = METRICS_PATH) -> pd.DataFrame:
    """Load fold-level MAE values produced by the XGBoost training script."""
    metrics_path = _require_file(metrics_path)
    metrics = pd.read_csv(metrics_path, parse_dates=["fold_end"])
    return metrics.sort_values("fold_end").reset_index(drop=True)


def preview_forecast_source_data() -> pd.DataFrame:
    """Summarise the raw and processed files used by the forecast section."""
    files = [
        (
            "Day-ahead prices",
            DATA_DIR / "day_ahead_prices_dk1_raw.csv",
            "TimeDK",
            "Target price series from Energi Data Service.",
        ),
        (
            "Consumption",
            DATA_DIR / "consumption_dk1_raw.csv",
            "TimeDK",
            "System load series used later in the impact simulation.",
        ),
        (
            "Weather actuals",
            DATA_DIR / "weather_actuals_raw.csv",
            "TimeDK",
            "Historical observed weather for the DK1 weather locations.",
        ),
        (
            "Weather forecasts",
            DATA_DIR / "weather_forecasts_raw.csv",
            "TimeDK",
            "Open-Meteo previous-run forecasts used to estimate forecast errors.",
        ),
        (
            "Weather error distributions",
            DATA_DIR / "weather_error_distributions.csv",
            "horizon_hours",
            "Forecast-error summary by variable and horizon.",
        ),
        (
            "Model matrix",
            DATA_DIR / "model_dataset.parquet",
            "target_time",
            "Forecast-like weather, calendar features, price lags, and target prices.",
        ),
    ]

    rows = []
    for name, path, time_col, role in files:
        if not path.exists():
            rows.append(
                {
                    "data": name,
                    "file": str(path.relative_to(ROOT)),
                    "rows": "missing",
                    "time_range": "missing",
                    "columns": "missing",
                    "role": role,
                }
            )
            continue

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if time_col == "horizon_hours" and time_col in df.columns:
            horizons = pd.to_numeric(df[time_col], errors="coerce")
            if horizons.notna().any():
                time_range = f"h={int(horizons.min())} to h={int(horizons.max())}"
            else:
                time_range = "not time indexed"
        elif time_col in df.columns:
            values = pd.to_datetime(df[time_col], errors="coerce")
            if values.notna().any():
                time_range = f"{values.min():%Y-%m-%d} to {values.max():%Y-%m-%d}"
            else:
                time_range = "not time indexed"
        else:
            time_range = "not time indexed"

        cols = ", ".join(df.columns[:7])
        if len(df.columns) > 7:
            cols += ", ..."
        rows.append(
            {
                "data": name,
                "file": str(path.relative_to(ROOT)),
                "rows": f"{len(df):,}",
                "time_range": time_range,
                "columns": cols,
                "role": role,
            }
        )
    return pd.DataFrame(rows)


def plot_raw_price_and_weather(
    start: str = "2024-01-01",
    end: str = "2024-01-14",
    price_csv: str | Path = DATA_DIR / "day_ahead_prices_dk1_raw.csv",
    weather_csv: str | Path = DATA_DIR / "weather_actuals_raw.csv",
):
    """Plot scraped DK1 prices and actual weather for the same period."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    prices = _load_prices(price_csv)
    price_col = "DayAheadPriceDKK" if "DayAheadPriceDKK" in prices.columns else TARGET_COL
    prices = (
        prices.set_index("TimeDK")
        .loc[start_ts:end_ts, [price_col]]
        .resample("h")
        .mean()
    )

    weather_csv = _require_file(weather_csv)
    weather = pd.read_csv(weather_csv, parse_dates=["TimeDK"])
    if "region" in weather.columns:
        weather = weather[weather["region"].astype(str).eq(REGION)]
    weather = weather.sort_values("TimeDK").set_index("TimeDK")
    weather = weather.loc[
        start_ts:end_ts,
        ["temperature_2m", "wind_speed_100m", "shortwave_radiation"],
    ].resample("h").mean()

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(prices.index, prices[price_col], color="#2563eb", linewidth=1.4)
    axes[0].set_ylabel("Price\nDKK/MWh" if price_col.endswith("DKK") else "Price\nEUR/MWh")
    axes[0].set_title(f"Scraped DK1 price and weather data ({start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d})")

    axes[1].plot(weather.index, weather["temperature_2m"], color="#dc2626", linewidth=1.4)
    axes[1].set_ylabel("Temp\nC")

    axes[2].plot(weather.index, weather["wind_speed_100m"], color="#0891b2", linewidth=1.4)
    axes[2].set_ylabel("Wind 100m\nm/s")

    axes[3].plot(weather.index, weather["shortwave_radiation"], color="#d97706", linewidth=1.4)
    axes[3].set_ylabel("Solar\nW/m2")
    axes[3].set_xlabel("Time")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    plt.show()
    return fig


def plot_day_ahead_price_history(
    price_csv: str | Path = DATA_DIR / "day_ahead_prices_dk1_raw.csv",
    start: str | None = None,
    end: str | None = None,
):
    """Plot DK1 price history and the average intraday price profile."""
    prices = _load_prices(price_csv)
    if start is not None:
        prices = prices[prices["TimeDK"].ge(pd.Timestamp(start))]
    if end is not None:
        prices = prices[prices["TimeDK"].le(pd.Timestamp(end))]

    if "DayAheadPriceDKK" in prices.columns:
        prices["price"] = prices["DayAheadPriceDKK"] / 1000
        ylabel = "DKK/kWh"
    else:
        prices["price"] = prices[TARGET_COL]
        ylabel = "EUR/MWh"

    hourly = (
        prices.set_index("TimeDK")["price"]
        .resample("h")
        .mean()
        .dropna()
        .to_frame("price")
    )
    daily = hourly["price"].resample("D").mean()
    rolling = daily.rolling(30, min_periods=7).mean()

    hourly["hour"] = hourly.index.hour
    intraday = hourly.groupby("hour")["price"].agg(
        mean="mean",
        q25=lambda x: np.quantile(x, 0.25),
        q75=lambda x: np.quantile(x, 0.75),
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(daily.index, daily, color="#94a3b8", linewidth=0.8, alpha=0.8, label="Daily mean")
    axes[0].plot(rolling.index, rolling, color="#2563eb", linewidth=2.0, label="30-day rolling mean")
    axes[0].axhline(daily.median(), color="#111827", linestyle=":", linewidth=1.0, label="Median daily price")
    axes[0].set_title("DK1 day-ahead price history")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(intraday.index, intraday["mean"], color="#2563eb", linewidth=2, label="Mean")
    axes[1].fill_between(
        intraday.index,
        intraday["q25"],
        intraday["q75"],
        color="#2563eb",
        alpha=0.16,
        label="25th-75th percentile",
    )
    axes[1].set_title("Average hourly price pattern")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel(ylabel)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    plt.show()
    return fig


def _weather_error_params(
    variable: str,
    horizon_hours: int,
    error_dist_csv: str | Path = DATA_DIR / "weather_error_distributions.csv",
) -> tuple[float, float]:
    error_dist_csv = _require_file(error_dist_csv)
    err = pd.read_csv(error_dist_csv)
    sub = err[err["forecast_variable"].astype(str).eq(variable)].sort_values("horizon_hours")
    if sub.empty:
        return 0.0, np.nan
    h = sub["horizon_hours"].astype(float).to_numpy()
    mean = sub["mean_error"].astype(float).to_numpy()
    std = sub["std_error"].astype(float).to_numpy()
    return float(np.interp(horizon_hours, h, mean)), float(np.interp(horizon_hours, h, std))


def _prepare_weather_forecast_slice(
    df: pd.DataFrame,
    horizon_hours: int,
    start: str | None,
    days: int,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    sub = df[df["horizon_h"].astype(int).eq(int(horizon_hours))].copy()
    if sub.empty:
        raise ValueError(f"No rows found for horizon_h={horizon_hours}.")

    if start is None:
        start_ts = sub["target_time"].min()
    else:
        start_ts = pd.Timestamp(start)
        if not sub["target_time"].between(start_ts, start_ts + pd.Timedelta(days=days)).any():
            start_ts = sub["target_time"].min()
    end_ts = start_ts + pd.Timedelta(days=days)
    sub = sub[sub["target_time"].between(start_ts, end_ts)].sort_values("target_time")
    return sub, start_ts, end_ts


def _plot_weather_forecast_panel(
    axes: np.ndarray,
    sub: pd.DataFrame,
    horizon_hours: int,
) -> None:
    flat_axes = axes.flatten()
    for ax, (title, cfg) in zip(flat_axes, WEATHER_PLOT_VARS.items()):
        mean_err, std_err = _weather_error_params(cfg["error_var"], horizon_hours)
        actual = sub[cfg["actual"]].astype(float)
        fcst = sub[cfg["fcst"]].astype(float)
        color = cfg["color"]

        ax.plot(sub["target_time"], actual, color="#111827", linewidth=1.2, label="Actual")
        ax.plot(
            sub["target_time"],
            fcst,
            color=color,
            linewidth=1.2,
            linestyle="--",
            label="Forecast-like input",
        )
        if np.isfinite(std_err):
            lower = actual + mean_err - 1.96 * std_err
            upper = actual + mean_err + 1.96 * std_err
            ax.fill_between(
                sub["target_time"],
                lower,
                upper,
                color=color,
                alpha=0.12,
                label="Approx. 95% error band",
            )
        ax.set_title(f"{title} ({cfg['unit']})")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def plot_weather_forecast_grid(
    horizon_hours: int | Iterable[int] = 24,
    model_dataset: str | Path = MODEL_DATASET_PATH,
    start: str | None = "2025-01-01",
    days: int = 14,
):
    """Compare simulated weather forecasts with actual values for one or more horizons."""
    df = load_model_dataset(model_dataset)

    if isinstance(horizon_hours, int):
        sub, start_ts, end_ts = _prepare_weather_forecast_slice(df, horizon_hours, start, days)
        fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
        _plot_weather_forecast_panel(axes, sub, int(horizon_hours))
        fig.suptitle(
            f"Forecast-like weather inputs at h={int(horizon_hours)} "
            f"({start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d})",
            fontsize=13,
        )
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()
        return fig

    horizons = [int(h) for h in horizon_hours]
    if not horizons:
        raise ValueError("horizon_hours must contain at least one horizon.")

    if len(horizons) == 1:
        return plot_weather_forecast_grid(
            horizon_hours=horizons[0],
            model_dataset=model_dataset,
            start=start,
            days=days,
        )

    fig = plt.figure(figsize=(7.5 * len(horizons), 8.5))
    subfigs = fig.subfigures(1, len(horizons), wspace=0.04)
    if len(horizons) == 2:
        subfigs = list(subfigs)

    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for subfig, horizon in zip(subfigs, horizons):
        sub, start_ts, end_ts = _prepare_weather_forecast_slice(df, horizon, start, days)
        axes = subfig.subplots(2, 2, sharex=True)
        _plot_weather_forecast_panel(axes, sub, horizon)
        subfig.suptitle(
            f"h={horizon} ({start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d})",
            fontsize=12,
        )
        subfig.autofmt_xdate()
        ranges.append((start_ts, end_ts))

    overall_start = min(start_ts for start_ts, _ in ranges)
    overall_end = max(end_ts for _, end_ts in ranges)
    fig.suptitle(
        "Forecast-like weather inputs by horizon "
        f"({overall_start:%Y-%m-%d} to {overall_end:%Y-%m-%d})",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout()
    plt.show()
    return fig


def plot_weather_price_correlation(
    correlation_horizon: int | None = 120,
    model_dataset: str | Path = MODEL_DATASET_PATH,
):
    """Plot correlations between forecast predictors and the future price target."""
    df = load_model_dataset(model_dataset)
    if correlation_horizon is not None:
        df = df[df["horizon_h"].astype(int).eq(int(correlation_horizon))].copy()
        horizon_text = f"h={correlation_horizon}"
    else:
        horizon_text = "all horizons"

    cols = [col for col in WEATHER_FEATURES + PRICE_LAG_FEATURES + TIME_FEATURES if col in df.columns]
    corr = (
        df[cols + [TARGET_COL]]
        .corr(numeric_only=True)[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values()
    )

    labels = corr.index.str.replace("fcst_", "", regex=False).str.replace("_", " ")
    colors = np.where(corr >= 0, "#dc2626", "#2563eb")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, corr.values, color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title(f"Correlation with future DK1 price ({horizon_text})")
    ax.set_xlabel("Pearson correlation with DayAheadPriceEUR")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    plt.show()
    return fig


def preview_feature_matrix_24h(
    model_dataset: str | Path = MODEL_DATASET_PATH,
    n: int = 8,
) -> pd.DataFrame:
    """Return a compact preview of the 24-hour-ahead model matrix."""
    df = load_model_dataset(model_dataset)
    sub = df[df["horizon_h"].astype(int).eq(24)].copy()
    cols = [
        "issue_time",
        "target_time",
        "horizon_h",
        TARGET_COL,
        "fcst_wind_speed_100m",
        "fcst_shortwave_radiation",
        "fcst_temperature_2m",
        "fcst_cloud_cover",
        "price_lag_24h",
        "price_lag_168h",
        "hour_of_day",
        "day_of_week",
    ]
    cols = [col for col in cols if col in sub.columns]
    out = sub[cols].head(n).copy()
    return out.rename(columns={TARGET_COL: "y_future_price_eur_mwh"})


def preview_xgboost_design() -> pd.DataFrame:
    """Show the five-model design used in src/models/XG_Boost_full_Res.py."""
    rows = []
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        rows.append(
            {
                "model": f"Day {day}",
                "forecast_horizons": f"h={h_lo}-{h_hi}",
                "target": "DayAheadPriceEUR at target_time",
                "feature_groups": "calendar/time, forecast weather, issue-time price lags",
                "validation": "expanding walk-forward split; 3-month test blocks",
            }
        )
    return pd.DataFrame(rows)


def _prediction_frame(preds_df: pd.DataFrame, unit: str) -> pd.DataFrame:
    scale, _ = _unit_scale(unit)
    out = preds_df.dropna(subset=[TARGET_COL, "predicted", "horizon_h"]).copy()
    out["forecast_day"] = _forecast_day_from_horizon(out["horizon_h"])
    out["actual_unit"] = out[TARGET_COL] * scale
    out["predicted_unit"] = out["predicted"] * scale
    out["residual_unit"] = out["predicted_unit"] - out["actual_unit"]
    out["abs_error_unit"] = out["residual_unit"].abs()
    return out


def make_forecast_metrics_table(
    predictions_path: str | Path = PREDICTIONS_PATH,
    unit: str = "dkk_kwh",
) -> pd.DataFrame:
    """Compute out-of-sample forecast metrics by forecast day."""
    preds = _prediction_frame(load_predictions(predictions_path), unit)
    _, label = _unit_scale(unit)

    rows = []
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        sub = preds[preds["horizon_h"].between(h_lo, h_hi)]
        if sub.empty:
            continue
        residual = sub["residual_unit"]
        rows.append(
            {
                "forecast_day": f"Day {day}",
                "horizon_window": f"h={h_lo}-{h_hi}",
                "n_predictions": len(sub),
                f"MAE ({label})": sub["abs_error_unit"].mean(),
                f"RMSE ({label})": np.sqrt(np.mean(np.square(residual))),
                f"Bias ({label})": residual.mean(),
                f"Residual std ({label})": residual.std(),
                f"P90 abs error ({label})": sub["abs_error_unit"].quantile(0.90),
            }
        )
    return pd.DataFrame(rows)


def _horizon_metrics(preds_df: pd.DataFrame, unit: str) -> pd.DataFrame:
    preds = _prediction_frame(preds_df, unit)
    rows = []
    for h, sub in preds.groupby("horizon_h"):
        residual = sub["residual_unit"]
        rows.append(
            {
                "horizon_h": int(h),
                "mae": sub["abs_error_unit"].mean(),
                "rmse": np.sqrt(np.mean(np.square(residual))),
                "bias": residual.mean(),
                "n": len(sub),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_h")


def plot_forecast_model_outputs(
    predictions_path: str | Path = PREDICTIONS_PATH,
    metrics_path: str | Path = METRICS_PATH,
    unit: str = "dkk_kwh",
    sample_days: int = 21,
):
    """Show walk-forward performance, horizon error, and actual-vs-predicted prices."""
    preds = load_predictions(predictions_path)
    metrics = load_metrics(metrics_path)
    scale, label = _unit_scale(unit)
    horizon_metrics = _horizon_metrics(preds, unit)
    day_metrics = make_forecast_metrics_table(predictions_path, unit)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    for day in DAY_GROUPS:
        col = f"day{day}_mae"
        if col in metrics.columns:
            ax.plot(
                metrics["fold_end"],
                metrics[col] * scale,
                marker="o",
                linewidth=1.5,
                markersize=3,
                color=COLORS[day],
                label=f"Day {day}",
            )
    ax.set_title("Walk-forward MAE by fold")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[0, 1]
    ax.plot(
        horizon_metrics["horizon_h"],
        horizon_metrics["mae"],
        color="#2563eb",
        linewidth=1.6,
    )
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        ax.axvspan(h_lo - 0.5, h_hi + 0.5, color=COLORS[day], alpha=0.08)
    ax.set_title("MAE by forecast horizon")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    pred_frame = _prediction_frame(preds, unit)
    last_fold = pred_frame["fold"].max() if "fold" in pred_frame.columns else None
    if last_fold is not None:
        window_source = pred_frame[pred_frame["fold"].eq(last_fold)]
    else:
        window_source = pred_frame
    end_t = window_source["target_time"].max()
    start_t = end_t - pd.Timedelta(days=sample_days)
    window = pred_frame[pred_frame["target_time"].between(start_t, end_t)]
    window = (
        window.groupby("target_time", as_index=False)
        .agg(actual=("actual_unit", "mean"), predicted=("predicted_unit", "mean"))
        .sort_values("target_time")
    )
    ax.plot(window["target_time"], window["actual"], color="#111827", linewidth=1.4, label="Actual")
    ax.plot(window["target_time"], window["predicted"], color="#2563eb", linewidth=1.4, linestyle="--", label="Predicted")
    ax.set_title(f"Actual vs predicted price ({sample_days} day sample)")
    ax.set_ylabel(label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    table_cols = [
        "forecast_day",
        "horizon_window",
        f"MAE ({label})",
        f"Bias ({label})",
        f"P90 abs error ({label})",
    ]
    table_df = day_metrics[table_cols].copy()
    for col in table_df.columns:
        if col not in {"forecast_day", "horizon_window"}:
            table_df[col] = table_df[col].map(lambda x: f"{x:.3f}")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    ax.set_title("Out-of-sample metrics by day model", pad=16)

    fig.suptitle("XGBoost forecast results for DK1", fontsize=14)
    fig.tight_layout()
    plt.show()
    return fig


def plot_forecast_mae_summary(
    predictions_path: str | Path = PREDICTIONS_PATH,
    metrics_path: str | Path = METRICS_PATH,
    unit: str = "dkk_kwh",
):
    """Plot walk-forward MAE by fold and MAE by forecast horizon."""
    preds = load_predictions(predictions_path)
    metrics = load_metrics(metrics_path)
    scale, label = _unit_scale(unit)
    horizon_metrics = _horizon_metrics(preds, unit)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    for day in DAY_GROUPS:
        col = f"day{day}_mae"
        if col in metrics.columns:
            ax.plot(
                metrics["fold_end"],
                metrics[col] * scale,
                marker="o",
                linewidth=1.5,
                markersize=3,
                color=COLORS[day],
                label=f"Day {day}",
            )
    ax.set_title("Walk-forward MAE by fold")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1]
    ax.plot(
        horizon_metrics["horizon_h"],
        horizon_metrics["mae"],
        color="#2563eb",
        linewidth=1.6,
    )
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        ax.axvspan(h_lo - 0.5, h_hi + 0.5, color=COLORS[day], alpha=0.08)
    ax.set_title("MAE by forecast horizon")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    plt.show()
    return fig


def compute_residual_uncertainty(
    predictions_path: str | Path = PREDICTIONS_PATH,
    unit: str = "dkk_kwh",
    cutoff: str | pd.Timestamp = POST_CRISIS_CUTOFF,
) -> pd.DataFrame:
    """Estimate horizon-specific uncertainty from walk-forward residuals."""
    preds = _prediction_frame(load_predictions(predictions_path), unit)
    cutoff = pd.Timestamp(cutoff)
    preds = preds[preds["target_time"].ge(cutoff)].copy()
    rows = []
    for h, sub in preds.groupby("horizon_h"):
        resid = sub["residual_unit"].dropna()
        if resid.empty:
            continue
        rows.append(
            {
                "horizon_h": int(h),
                "forecast_day": int((int(h) - 1) // 24 + 1),
                "n": len(resid),
                "bias": resid.mean(),
                "sigma": resid.std(),
                "mae": resid.abs().mean(),
                "q05": resid.quantile(0.05),
                "q10": resid.quantile(0.10),
                "q90": resid.quantile(0.90),
                "q95": resid.quantile(0.95),
                "p90_abs_error": resid.abs().quantile(0.90),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_h").reset_index(drop=True)


def make_uncertainty_summary_table(
    predictions_path: str | Path = PREDICTIONS_PATH,
    unit: str = "dkk_kwh",
    cutoff: str | pd.Timestamp = POST_CRISIS_CUTOFF,
) -> pd.DataFrame:
    """Summarise residual uncertainty by forecast day."""
    unc = compute_residual_uncertainty(predictions_path, unit, cutoff)
    _, label = _unit_scale(unit)
    rows = []
    for day, sub in unc.groupby("forecast_day"):
        rows.append(
            {
                "forecast_day": f"Day {day}",
                "horizon_window": f"h={(day - 1) * 24 + 1}-{day * 24}",
                "mean_horizon_sigma": sub["sigma"].mean(),
                "max_horizon_sigma": sub["sigma"].max(),
                "mean_p90_abs_error": sub["p90_abs_error"].mean(),
                "mean_bias": sub["bias"].mean(),
                "residual_sample_size": int(sub["n"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    rename = {
        "mean_horizon_sigma": f"Mean residual sigma ({label})",
        "max_horizon_sigma": f"Max residual sigma ({label})",
        "mean_p90_abs_error": f"Mean P90 abs error ({label})",
        "mean_bias": f"Mean bias ({label})",
    }
    return out.rename(columns=rename)


def plot_price_history_with_5day_forecast(
    predictions_path: str | Path = PREDICTIONS_PATH,
    price_csv: str | Path = DATA_DIR / "day_ahead_prices_dk1_raw.csv",
    issue_time: str | pd.Timestamp | None = None,
    ctx_days: int = 7,
    unit: str = "dkk_kwh",
    show_actuals: bool = True,
    show_uncertainty: bool = True,
    cutoff: str | pd.Timestamp = POST_CRISIS_CUTOFF,
):
    """Plot recent price history followed by one 120-hour XGBoost forecast."""
    import plotly.graph_objects as go

    raw = load_predictions(predictions_path)
    issue = _resolve_issue_time(raw, issue_time)
    scale, label = _unit_scale(unit)

    single = raw[raw["issue_time"].eq(issue)].sort_values("target_time").copy()
    single = single[single["target_time"].gt(issue)].head(120)
    if single.empty:
        raise ValueError(f"No 120-hour forecast rows found for issue_time={issue}.")

    single["predicted_unit"] = single["predicted"] * scale
    single["actual_unit"] = single[TARGET_COL] * scale

    prices = _load_prices(price_csv)
    if unit.lower() in {"dkk_kwh", "dkk/kwh"} and "DayAheadPriceDKK" in prices.columns:
        prices["price_unit"] = prices["DayAheadPriceDKK"] / 1000
    elif unit.lower() in {"dkk_mwh", "dkk/mwh"} and "DayAheadPriceDKK" in prices.columns:
        prices["price_unit"] = prices["DayAheadPriceDKK"]
    else:
        prices["price_unit"] = prices[TARGET_COL] * scale

    hist_end = single["target_time"].min()
    hist_start = hist_end - pd.Timedelta(days=ctx_days)
    hist = (
        prices[prices["TimeDK"].between(hist_start, hist_end, inclusive="left")]
        .dropna(subset=["price_unit"])
        .sort_values("TimeDK")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["TimeDK"],
            y=hist["price_unit"],
            mode="lines",
            line=dict(color="#fb923c", width=2),
            name="Historical actual",
        )
    )

    if show_uncertainty:
        unc = compute_residual_uncertainty(predictions_path, unit=unit, cutoff=cutoff)
        sigma_map = unc.set_index("horizon_h")["sigma"].to_dict()
        single["sigma"] = single["horizon_h"].map(sigma_map).fillna(0.0)
        if single["sigma"].gt(0).any():
            fig.add_trace(
                go.Scatter(
                    x=single["target_time"],
                    y=single["predicted_unit"] + single["sigma"],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=single["target_time"],
                    y=single["predicted_unit"] - single["sigma"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(167,139,250,0.16)",
                    name="+/- 1 sigma",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=single["target_time"],
            y=single["predicted_unit"],
            mode="lines",
            line=dict(color="#7c3aed", width=2.5),
            name="XGBoost forecast",
        )
    )

    if show_actuals and single["actual_unit"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=single["target_time"],
                y=single["actual_unit"],
                mode="lines",
                line=dict(color="#16a34a", width=2, dash="dot"),
                name="Actual in forecast window",
            )
        )

    fig.add_vrect(
        x0=single["target_time"].iloc[0],
        x1=single["target_time"].iloc[-1],
        fillcolor="rgba(124,58,237,0.08)",
        line_width=0,
        layer="below",
        annotation_text="5-day forecast",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white",
        height=430,
        title=f"Historical price context + 5-day XGBoost forecast (issued {issue:%Y-%m-%d %H:%M})",
        xaxis_title="Time",
        yaxis_title=label,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(l=50, r=25, t=70, b=85),
    )
    fig.show()
    return fig


def plot_single_forecast_with_uncertainty(
    predictions_path: str | Path = PREDICTIONS_PATH,
    issue_time: str | pd.Timestamp | None = None,
    unit: str = "dkk_kwh",
    cutoff: str | pd.Timestamp = POST_CRISIS_CUTOFF,
):
    """Plot one 120-hour forecast with residual-based +/-1 sigma bands."""
    raw = load_predictions(predictions_path)
    issue = _resolve_issue_time(raw, issue_time)
    scale, label = _unit_scale(unit)
    single = raw[raw["issue_time"].eq(issue)].sort_values("horizon_h").copy()
    single["actual_unit"] = single[TARGET_COL] * scale
    single["predicted_unit"] = single["predicted"] * scale

    unc = compute_residual_uncertainty(predictions_path, unit=unit, cutoff=cutoff)
    sigma_map = unc.set_index("horizon_h")["sigma"].to_dict()
    single["sigma"] = single["horizon_h"].map(sigma_map)

    fig, ax = plt.subplots(figsize=(14, 5))
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        day_rows = single[single["horizon_h"].between(h_lo, h_hi)]
        if day_rows.empty:
            continue
        ax.axvspan(
            day_rows["target_time"].iloc[0],
            day_rows["target_time"].iloc[-1],
            color=COLORS[day],
            alpha=0.07,
            label=f"Day {day}" if day == 1 else None,
        )
    upper = single["predicted_unit"] + single["sigma"]
    lower = single["predicted_unit"] - single["sigma"]
    ax.fill_between(single["target_time"], lower, upper, color="#7c3aed", alpha=0.16, label="+/-1 residual sigma")
    ax.plot(single["target_time"], single["actual_unit"], color="#111827", linewidth=1.6, label="Actual")
    ax.plot(single["target_time"], single["predicted_unit"], color="#2563eb", linewidth=1.7, linestyle="--", label="Forecast")
    ax.set_title(f"120-hour forecast with residual uncertainty (issued {issue:%Y-%m-%d %H:%M})")
    ax.set_ylabel(label)
    ax.set_xlabel("Target time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    plt.show()
    return fig


def plot_residual_diagnostics(
    predictions_path: str | Path = PREDICTIONS_PATH,
    unit: str = "dkk_kwh",
    cutoff: str | pd.Timestamp = POST_CRISIS_CUTOFF,
):
    """Plot residual diagnostics used to interpret forecast uncertainty."""
    raw = load_predictions(predictions_path)
    preds = _prediction_frame(raw, unit)
    cutoff = pd.Timestamp(cutoff)
    post = preds[preds["target_time"].ge(cutoff)].copy()
    unc = compute_residual_uncertainty(predictions_path, unit=unit, cutoff=cutoff)
    _, label = _unit_scale(unit)

    sigma_map = unc.set_index("horizon_h")["sigma"].to_dict()
    post["sigma"] = post["horizon_h"].map(sigma_map)
    post["covered_1sigma"] = post["abs_error_unit"].le(post["sigma"])
    post["covered_196sigma"] = post["abs_error_unit"].le(1.96 * post["sigma"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    bins = np.linspace(
        post["residual_unit"].quantile(0.01),
        post["residual_unit"].quantile(0.99),
        60,
    )
    for day, day_rows in post.groupby("forecast_day"):
        ax.hist(
            day_rows["residual_unit"],
            bins=bins,
            alpha=0.28,
            color=COLORS[int(day)],
            label=f"Day {int(day)}",
            density=True,
        )
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Residual distribution by day model")
    ax.set_xlabel(f"Prediction minus actual ({label})")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(unc["horizon_h"], unc["sigma"], color="#7c3aed", linewidth=1.7, label="Residual sigma")
    ax.plot(unc["horizon_h"], unc["p90_abs_error"], color="#2563eb", linewidth=1.3, label="P90 absolute error")
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        ax.axvspan(h_lo - 0.5, h_hi + 0.5, color=COLORS[day], alpha=0.07)
    ax.set_title(f"Horizon-specific uncertainty after {cutoff:%Y-%m-%d}")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    coverage = (
        post.groupby("forecast_day")
        .agg(
            coverage_1sigma=("covered_1sigma", "mean"),
            coverage_196sigma=("covered_196sigma", "mean"),
        )
        .reset_index()
    )
    x = np.arange(len(coverage))
    width = 0.35
    ax.bar(x - width / 2, coverage["coverage_1sigma"], width, color="#7c3aed", label="+/-1 sigma")
    ax.bar(x + width / 2, coverage["coverage_196sigma"], width, color="#2563eb", label="+/-1.96 sigma")
    ax.axhline(0.68, color="#7c3aed", linestyle=":", linewidth=1)
    ax.axhline(0.95, color="#2563eb", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Day {int(day)}" for day in coverage["forecast_day"]])
    ax.set_ylim(0, 1)
    ax.set_title("Empirical interval coverage")
    ax.set_ylabel("Share of realised prices inside band")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    issue = _resolve_issue_time(raw)
    single = post[post["issue_time"].eq(issue)].sort_values("horizon_h").copy()
    if single.empty:
        single = _prediction_frame(raw[raw["issue_time"].eq(issue)], unit).sort_values("horizon_h")
        single["sigma"] = single["horizon_h"].map(sigma_map)
    upper = single["predicted_unit"] + single["sigma"]
    lower = single["predicted_unit"] - single["sigma"]
    ax.fill_between(single["target_time"], lower, upper, color="#7c3aed", alpha=0.16, label="+/-1 sigma")
    ax.plot(single["target_time"], single["actual_unit"], color="#111827", linewidth=1.3, label="Actual")
    ax.plot(single["target_time"], single["predicted_unit"], color="#2563eb", linestyle="--", linewidth=1.4, label="Forecast")
    ax.set_title(f"Example forecast band (issued {issue:%Y-%m-%d %H:%M})")
    ax.set_ylabel(label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Residual-based prediction uncertainty", fontsize=14)
    fig.tight_layout()
    plt.show()
    return fig


def preview_recommendation_workflow(
    predictions_path: str | Path = PREDICTIONS_PATH,
    issue_time: str | pd.Timestamp | None = None,
    unit: str = "dkk_kwh",
) -> pd.DataFrame:
    """Create a small forecast-to-recommendation preview for the dashboard section."""
    raw = load_predictions(predictions_path)
    issue = _resolve_issue_time(raw, issue_time)
    scale, label = _unit_scale(unit)
    fcst = raw[raw["issue_time"].eq(issue)].sort_values("horizon_h").copy()
    fcst["price"] = fcst["predicted"] * scale
    fcst = fcst[fcst["target_time"].gt(issue)].head(120)
    if fcst.empty:
        raise ValueError("No future forecast rows found for the selected issue time.")

    def window_summary(hours: int, cheapest: bool = True) -> tuple[pd.Timestamp, pd.Timestamp, float]:
        rolling = fcst["price"].rolling(hours, min_periods=hours).mean()
        idx = rolling.idxmin() if cheapest else rolling.idxmax()
        end_pos = fcst.index.get_loc(idx)
        window = fcst.iloc[end_pos - hours + 1 : end_pos + 1]
        return window["target_time"].iloc[0], window["target_time"].iloc[-1], float(window["price"].mean())

    current_price = float(fcst["price"].iloc[0])
    cheap_start, cheap_end, cheap_price = window_summary(3, cheapest=True)
    high_start, high_end, high_price = window_summary(6, cheapest=False)
    gain = current_price - cheap_price

    return pd.DataFrame(
        [
            {
                "dashboard_step": "Forecast state",
                "output": f"120 hourly prices issued {issue:%Y-%m-%d %H:%M}",
                "value": f"First forecast hour: {current_price:.3f} {label}",
            },
            {
                "dashboard_step": "Find cheap window",
                "output": "Cheapest 3-hour predicted block",
                "value": f"{cheap_start:%d %b %H:%M} to {cheap_end:%d %b %H:%M} at {cheap_price:.3f} {label}",
            },
            {
                "dashboard_step": "Find expensive window",
                "output": "Most expensive 6-hour predicted block",
                "value": f"{high_start:%d %b %H:%M} to {high_end:%d %b %H:%M} at {high_price:.3f} {label}",
            },
            {
                "dashboard_step": "Recommendation signal",
                "output": "Wait-or-consume-now comparison",
                "value": (
                    f"Waiting for the cheap block saves about {gain:.3f} {label}"
                    if gain > 0
                    else f"The current hour is already competitive; gain is {gain:.3f} {label}"
                ),
            },
        ]
    )


__all__ = [
    "compute_residual_uncertainty",
    "load_metrics",
    "load_model_dataset",
    "load_predictions",
    "make_forecast_metrics_table",
    "make_uncertainty_summary_table",
    "plot_day_ahead_price_history",
    "plot_forecast_mae_summary",
    "plot_forecast_model_outputs",
    "plot_price_history_with_5day_forecast",
    "plot_raw_price_and_weather",
    "plot_residual_diagnostics",
    "plot_single_forecast_with_uncertainty",
    "plot_weather_forecast_grid",
    "plot_weather_price_correlation",
    "preview_feature_matrix_24h",
    "preview_forecast_source_data",
    "preview_recommendation_workflow",
    "preview_xgboost_design",
]
