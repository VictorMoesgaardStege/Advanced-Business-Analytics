"""
data_processing.py  —  Build the XGBoost model input dataset
=============================================================
Reads weather actuals and error distributions, simulates NWP-style
forecasts by adding horizon-scaled Gaussian noise, and saves the
result as forecast_dataset.parquet.

Quick start
-----------
    from data_processing import build_forecast_dataset

    build_forecast_dataset()

Or run directly:
    python data_processing.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

FORECAST_HORIZON_HOURS = 120
ISSUE_INTERVAL_HOURS   = 12

FORECAST_VAR_MAP = {
    "wind_speed_10m":      "wind_speed_10m",
    "wind_direction_10m":  "wind_direction_10m",
    "wind_speed_100m":     "wind_speed_120m",
    "wind_direction_100m": "wind_direction_120m",
    "shortwave_radiation": "shortwave_radiation",
    "cloud_cover":         "cloud_cover",
    "temperature_2m":      "temperature_2m",
    "pressure_msl":        "pressure_msl",
}

VAR_BOUNDS = {
    "wind_speed_10m":      (0,    None),
    "wind_direction_10m":  (0,    360),
    "wind_speed_100m":     (0,    None),
    "wind_direction_100m": (0,    360),
    "shortwave_radiation": (0,    None),
    "cloud_cover":         (0,    100),
    "temperature_2m":      (None, None),
    "pressure_msl":        (None, None),
}

WEATHER_VARS = list(FORECAST_VAR_MAP.keys())
HORIZONS     = np.arange(1, FORECAST_HORIZON_HOURS + 1, dtype=int)


def _build_error_params(err_dist: pd.DataFrame) -> dict:
    known_horizons = sorted(err_dist["horizon_hours"].unique())
    params = {}
    for actual_var, fcst_var in FORECAST_VAR_MAP.items():
        sub      = err_dist[err_dist["forecast_variable"] == fcst_var].set_index("horizon_hours")
        kn_means = [sub.loc[h, "mean_error"] for h in known_horizons]
        kn_stds  = [sub.loc[h, "std_error"]  for h in known_horizons]
        mean_arr = np.interp(HORIZONS, known_horizons, kn_means)
        std_arr  = np.interp(HORIZONS, known_horizons, kn_stds)
        first_h  = known_horizons[0]
        below    = HORIZONS < first_h
        mean_arr[below] = kn_means[0] * (HORIZONS[below] / first_h)
        std_arr [below] = kn_stds [0] * (HORIZONS[below] / first_h)
        params[actual_var] = {"mean": mean_arr, "std": std_arr}
    return params


def _time_features(ts: pd.Timestamp) -> dict:
    return {
        "hour_of_day": ts.hour,
        "day_of_week": ts.dayofweek,
        "month":       ts.month,
        "is_weekend":  int(ts.dayofweek >= 5),
        "sin_hour":    np.sin(2 * np.pi * ts.hour / 24),
        "cos_hour":    np.cos(2 * np.pi * ts.hour / 24),
        "sin_doy":     np.sin(2 * np.pi * ts.dayofyear / 365),
        "cos_doy":     np.cos(2 * np.pi * ts.dayofyear / 365),
    }


def build_forecast_dataset(
    actuals_csv:    str | Path = DATA_DIR / "weather_actuals_raw.csv",
    error_dist_csv: str | Path = DATA_DIR / "weather_error_distributions.csv",
    output_path:    str | Path = DATA_DIR / "forecast_dataset.parquet",
    seed: int = 42,
) -> pd.DataFrame:
    """Build the synthetic forecast dataset used by XGBoost.

    Reads weather actuals and error distributions, simulates NWP-style
    forecasts by adding horizon-scaled Gaussian noise, and saves the
    result as a parquet file. Also writes a slim dashboard parquet
    (weather_dk1_dashboard.parquet) as a side effect.

    Parameters
    ----------
    actuals_csv    : path to weather_actuals_raw.csv
    error_dist_csv : path to weather_error_distributions.csv
    output_path    : where to write the parquet (default: data/forecast_dataset.parquet)
    seed           : random seed for reproducibility (default: 42)

    Returns
    -------
    pd.DataFrame with columns region, issue_time, target_time, horizon_h,
                 fcst_*, actual_*, and time features.
    """
    actuals_csv    = Path(actuals_csv)
    error_dist_csv = Path(error_dist_csv)
    output_path    = Path(output_path)

    print("[build_forecast_dataset] Loading inputs...")
    actuals = (
        pd.read_csv(actuals_csv, parse_dates=["TimeDK"])
        .rename(columns={"TimeDK": "time"})
        .sort_values(["region", "time"])
        .reset_index(drop=True)
    )
    err_dist = pd.read_csv(error_dist_csv)
    print(f"  Actuals: {len(actuals):,} rows | regions: {actuals['region'].unique().tolist()}")

    print("[build_forecast_dataset] Interpolating error parameters...")
    err_params = _build_error_params(err_dist)

    print(f"[build_forecast_dataset] Simulating forecasts "
          f"(every {ISSUE_INTERVAL_HOURS}h, horizon 1-{FORECAST_HORIZON_HOURS}h)...")
    rng     = np.random.default_rng(seed)
    records = []

    for region, rdf in actuals.groupby("region"):
        rdf         = rdf.set_index("time").sort_index()
        all_times   = rdf.index
        cutoff      = all_times[-1] - pd.Timedelta(hours=int(FORECAST_HORIZON_HOURS))
        valid_times = all_times[all_times <= cutoff]
        issue_times = valid_times[::ISSUE_INTERVAL_HOURS]
        print(f"  {region}: {len(issue_times)} issue times x {len(HORIZONS)} horizons "
              f"= {len(issue_times) * len(HORIZONS):,} rows")

        for issue_time in issue_times:
            for h_idx, h in enumerate(HORIZONS):
                target_time = issue_time + pd.Timedelta(hours=int(h))
                if target_time not in rdf.index:
                    continue
                actual_row = rdf.loc[target_time]
                rec = {
                    "region":      region,
                    "issue_time":  issue_time,
                    "target_time": target_time,
                    "horizon_h":   int(h),
                    **_time_features(target_time),
                }
                for var in WEATHER_VARS:
                    actual_val = float(actual_row[var])
                    if var == "shortwave_radiation" and actual_val < 0.2:
                        rec[f"fcst_{var}"]   = 0.0
                        rec[f"actual_{var}"] = actual_val
                        continue
                    noise = rng.normal(
                        loc=float(err_params[var]["mean"][h_idx]),
                        scale=max(float(err_params[var]["std"][h_idx]), 1e-9),
                    )
                    fcst_val = actual_val + noise
                    lo, hi = VAR_BOUNDS[var]
                    if lo is not None:
                        fcst_val = max(fcst_val, lo)
                    if hi is not None:
                        fcst_val = min(fcst_val, hi)
                    rec[f"fcst_{var}"]   = round(fcst_val, 4)
                    rec[f"actual_{var}"] = actual_val
                records.append(rec)

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  {len(df):,} rows x {df.shape[1]} cols saved -> {output_path}")

    # Slim dashboard file (DK1_west only)
    dashboard_cols = [
        "region", "issue_time", "target_time", "horizon_h",
        "fcst_wind_speed_100m",     "actual_wind_speed_100m",
        "fcst_shortwave_radiation", "actual_shortwave_radiation",
        "fcst_temperature_2m",      "actual_temperature_2m",
    ]
    dash_df   = df.loc[df["region"] == "DK1_west", dashboard_cols].copy()
    dash_path = DATA_DIR / "weather_dk1_dashboard.parquet"
    dash_df.to_parquet(dash_path, index=False, compression="snappy")
    print(f"  Dashboard file: {dash_path}  ({len(dash_df):,} rows)")

    return df


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_forecast_dataset()
