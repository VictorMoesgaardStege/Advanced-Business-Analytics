"""
data_processing.py  —  Build the input dataset for the XGBoost model
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
                    if lo is not None and hi is not None and "wind_direction" in var:
                        fcst_val = fcst_val % 360
                    else:
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


WEATHER_PLOT_VARS = {
    "Wind 100m":   ("fcst_wind_speed_100m",    "actual_wind_speed_100m",    "m/s"),
    "Solar":       ("fcst_shortwave_radiation", "actual_shortwave_radiation", "W/m²"),
    "Temperature": ("fcst_temperature_2m",      "actual_temperature_2m",      "°C"),
}


def plot_weather_error_distributions(
    error_dist_csv: str | Path = DATA_DIR / "weather_error_distributions.csv",
) -> None:
    import matplotlib.pyplot as plt

    err       = pd.read_csv(error_dist_csv)
    variables = err["forecast_variable"].unique()
    n         = len(variables)
    ncols     = 4
    nrows     = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    axes      = axes.flatten()

    for ax, var in zip(axes, variables):
        sub = err[err["forecast_variable"] == var].sort_values("horizon_hours")
        h   = sub["horizon_hours"].values
        ax.fill_between(h, sub["p05_error"], sub["p95_error"], alpha=0.15, color="steelblue", label="p05-p95")
        ax.fill_between(h, sub["p25_error"], sub["p75_error"], alpha=0.35, color="steelblue", label="p25-p75")
        ax.plot(h, sub["mean_error"], color="steelblue", linewidth=2, marker="o", markersize=4, label="Mean error")
        ax.plot(h, sub["p50_error"], color="steelblue", linewidth=1.5, linestyle="--", marker="o", markersize=3, alpha=0.7, label="Median error")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("Error")
        ax.set_xticks(h)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=2)
    fig.suptitle("Weather forecast error distributions by horizon", fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_weather_forecast(
    dashboard_parquet: str | Path = DATA_DIR / "weather_dk1_dashboard.parquet",
    ctx_days: int = 7,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df         = pd.read_parquet(dashboard_parquet)
    issue_ts   = pd.to_datetime(df["issue_time"].max())
    hist_start = issue_ts - pd.Timedelta(days=ctx_days)
    fcst_end   = issue_ts + pd.Timedelta(days=5)

    hist = (
        df[(df["target_time"] >= hist_start) & (df["target_time"] < issue_ts)]
        .groupby("target_time", as_index=False).first()
        .sort_values("target_time")
    )
    fcst = (
        df[(df["issue_time"] == issue_ts) & (df["target_time"] > issue_ts) & (df["target_time"] <= fcst_end)]
        .sort_values("target_time")
    )

    colors = ["#38bdf8", "#fbbf24", "#f87171"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, (var_name, (fcst_col, actual_col, unit)), color in zip(axes, WEATHER_PLOT_VARS.items(), colors):
        ax.plot(hist["target_time"], hist[actual_col], color=color, linewidth=1.5, label="Actual")
        if not fcst.empty:
            ax.plot(fcst["target_time"], fcst[fcst_col], color=color, linewidth=1.5, linestyle="--", label="Forecast")
            if fcst[actual_col].notna().any():
                ax.plot(fcst["target_time"], fcst[actual_col], color="grey", linewidth=1, linestyle=":", label="Actual (fcst window)")
            ax.axvspan(fcst["target_time"].iloc[0], fcst["target_time"].iloc[-1], alpha=0.07, color=color)
        ax.axvline(issue_ts, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_ylabel(f"{var_name}\n({unit})", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    fig.suptitle(
        f"Weather · DK1 West · {ctx_days}d history + 5-day forecast\n"
        f"issued at {issue_ts.strftime('%Y-%m-%d %H:%M')}",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_forecast_dataset()
