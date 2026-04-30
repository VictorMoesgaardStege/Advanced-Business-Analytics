"""
FORSOEG_DATA.py  -  Synthetic forecast dataset for hourly spot-price modelling
===============================================================================
Inputs
------
  data/weather_actuals_raw.csv           - ground-truth hourly weather observations
  data/weather_error_distributions.csv  - bias & std as a function of forecast horizon

Output
------
  data/forecast_dataset.parquet           - one row per (region, issue_time, horizon_h)

Each row contains
-----------------
  issue_time   : when the forecast was issued
  target_time  : the hour being predicted  (= issue_time + horizon_h)
  horizon_h    : hours ahead (1 … 120, i.e. 5 days)
  fcst_*       : simulated forecast value  (actual + horizon-scaled Gaussian noise)
  actual_*     : ground-truth value        (for joining spot prices / evaluation)
  time features derived from target_time  (hour, weekday, month, sin/cos encodings)

Usage
-----
  python FORSOEG_DATA.py

Then join spot prices on target_time to add the model target column.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
SEED                   = 42
FORECAST_HORIZON_HOURS = 120         # 5 days = 120 h
ISSUE_INTERVAL_HOURS   = 12          # issue a new forecast every 12 h
DATA_DIR               = Path("data")

# ── Variable mapping ───────────────────────────────────────────────────────────
# actual column name  =>  forecast_variable name used in the error-distribution file.
# NWP forecasts were generated at 120 m but compared against the 100 m measurement.
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

# Physical clipping bounds [lo, hi]  (None = unconstrained)
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
HORIZONS     = np.arange(1, FORECAST_HORIZON_HOURS + 1, dtype=int)   # [1, 2, …, 120]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    actuals = (
        pd.read_csv(DATA_DIR / "weather_actuals_raw.csv", parse_dates=["TimeDK"])
        .rename(columns={"TimeDK": "time"})
        .sort_values(["region", "time"])
        .reset_index(drop=True)
    )
    err_dist = pd.read_csv(DATA_DIR / "weather_error_distributions.csv")
    return actuals, err_dist


# ── Error-parameter interpolation ──────────────────────────────────────────────

def build_error_params(err_dist: pd.DataFrame) -> dict:
    """
    Interpolate mean_error and std_error at every horizon h = 1 … 120 from the
    five known distribution points (24, 48, 72, 96, 120 h).

    For h < 24 we linearly scale towards 0 (a forecast at h = 0 has zero error
    by definition).  For h > 120 the values are held flat (not used here).
    """
    known_horizons = sorted(err_dist["horizon_hours"].unique())   # [24, 48, 72, 96, 120]
    params = {}

    for actual_var, fcst_var in FORECAST_VAR_MAP.items():
        sub = err_dist[err_dist["forecast_variable"] == fcst_var].set_index("horizon_hours")

        kn_means = [sub.loc[h, "mean_error"] for h in known_horizons]
        kn_stds  = [sub.loc[h, "std_error"]  for h in known_horizons]

        # Linear interpolation across known horizons; flat extrapolation beyond 120 h
        mean_arr = np.interp(HORIZONS, known_horizons, kn_means)
        std_arr  = np.interp(HORIZONS, known_horizons, kn_stds)

        # Scale down linearly for horizons shorter than the first known point (24 h)
        first_h = known_horizons[0]
        below   = HORIZONS < first_h
        mean_arr[below] = kn_means[0] * (HORIZONS[below] / first_h)
        std_arr [below] = kn_stds [0] * (HORIZONS[below] / first_h)

        params[actual_var] = {"mean": mean_arr, "std": std_arr}

    return params


# ── Forecast simulation ────────────────────────────────────────────────────────

def _time_features(ts: pd.Timestamp) -> dict:
    """Cyclic and categorical time features derived from a target timestamp."""
    return {
        "hour_of_day": ts.hour,
        "day_of_week": ts.dayofweek,        # 0 = Monday
        "month":       ts.month,
        "is_weekend":  int(ts.dayofweek >= 5),
        "sin_hour":    np.sin(2 * np.pi * ts.hour / 24),
        "cos_hour":    np.cos(2 * np.pi * ts.hour / 24),
        "sin_doy":     np.sin(2 * np.pi * ts.dayofyear / 365),
        "cos_doy":     np.cos(2 * np.pi * ts.dayofyear / 365),
    }


def simulate_forecasts(actuals: pd.DataFrame, err_params: dict) -> pd.DataFrame:
    """
    For every (region, issue_time, horizon_h) triple, draw one synthetic forecast
    by adding horizon-appropriate Gaussian noise to the actual observed value.
    """
    rng = np.random.default_rng(SEED)
    records = []

    for region, rdf in actuals.groupby("region"):
        rdf = rdf.set_index("time").sort_index()
        all_times = rdf.index

        # Only issue forecasts where the full horizon stays within the data range
        cutoff      = all_times[-1] - pd.Timedelta(hours=int(FORECAST_HORIZON_HOURS))
        valid_times = all_times[all_times <= cutoff]

        # Sub-sample: every ISSUE_INTERVAL_HOURS (data is 1-h resolution, so step by that many positions)
        issue_times = valid_times[::ISSUE_INTERVAL_HOURS]

        n_issue = len(issue_times)
        print(f"  {region}: {n_issue} issue times × {len(HORIZONS)} horizons = {n_issue * len(HORIZONS):,} rows")

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

                    # Solar radiation is physically zero at night — no forecast error to draw
                    if var == "shortwave_radiation" and actual_val < 0.2:
                        rec[f"fcst_{var}"]   = 0.0
                        rec[f"actual_{var}"] = actual_val
                        continue

                    # Draw noise from N(mean_error(h), std_error(h))
                    noise = rng.normal(
                        loc=float(err_params[var]["mean"][h_idx]),
                        scale=max(float(err_params[var]["std"][h_idx]), 1e-9),
                    )
                    fcst_val = actual_val + noise

                    # Clip to physical range
                    lo, hi = VAR_BOUNDS[var]
                    if lo is not None:
                        fcst_val = max(fcst_val, lo)
                    if hi is not None:
                        fcst_val = min(fcst_val, hi)

                    rec[f"fcst_{var}"]   = round(fcst_val, 4)
                    rec[f"actual_{var}"] = actual_val

                records.append(rec)

    return pd.DataFrame(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FORSOEG_DATA  -  generating synthetic forecast dataset")
    print("=" * 60)

    print("\n[1/4] Loading data …")
    actuals, err_dist = load_data()
    print(f"  Actuals : {len(actuals):,} rows | "
          f"regions: {actuals['region'].unique().tolist()} | "
          f"{actuals['time'].min().date()} to {actuals['time'].max().date()}")
    print(f"  Error dist: {len(err_dist)} rows | "
          f"horizons: {sorted(err_dist['horizon_hours'].unique())} h | "
          f"variables: {err_dist['forecast_variable'].nunique()}")

    print("\n[2/4] Interpolating error parameters for horizons 1-120 h …")
    err_params = build_error_params(err_dist)
    for var, p in err_params.items():
        print(f"  {var:30s}  bias@24h={p['mean'][23]:+.3f}  std@24h={p['std'][23]:.3f}"
              f"  =>  bias@120h={p['mean'][119]:+.3f}  std@120h={p['std'][119]:.3f}")

    print(f"\n[3/4] Simulating forecasts "
          f"(issue every {ISSUE_INTERVAL_HOURS} h, horizon 1-{FORECAST_HORIZON_HOURS} h) …")
    df = simulate_forecasts(actuals, err_params)

    print(f"\n[4/4] Saving dataset …")
    out_path = DATA_DIR / "forecast_dataset.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\n  Output : {out_path}")
    print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\n  Columns:")
    for col in df.columns:
        print(f"    {col}")
    print(f"\n  Sample (first 3 rows):")

    print("\n[5/5] Exporting slim dashboard file (DK1_west only) …")
    dashboard_cols = [
        "region", "issue_time", "target_time", "horizon_h",
        "fcst_wind_speed_100m",    "actual_wind_speed_100m",
        "fcst_shortwave_radiation","actual_shortwave_radiation",
        "fcst_temperature_2m",     "actual_temperature_2m",
    ]
    dash_df   = df.loc[df["region"] == "DK1_west", dashboard_cols].copy()
    dash_path = DATA_DIR / "weather_dk1_dashboard.parquet"
    dash_df.to_parquet(dash_path, index=False, compression="snappy")
    print(f"  Output : {dash_path}  ({dash_path.stat().st_size / 1e6:.1f} MB  |  {len(dash_df):,} rows)")
    print(df.head(3).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
