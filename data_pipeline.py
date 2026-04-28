"""
data_pipeline.py  —  one-stop data fetching and dataset generation
===================================================================
All raw data fetching and the forsoeg_dataset build step exposed as
simple Python functions.  Each function saves to a default path and
returns a DataFrame so results can be used immediately in notebooks.

Quick start
-----------
    from data_pipeline import (
        fetch_weather_actuals,
        fetch_weather_forecasts,
        fetch_consumption,
        fetch_supply,
        fetch_day_ahead_prices,
        build_forecast_dataset,
        fetch_all,
    )

    # Fetch everything for a new date range and rebuild the model dataset
    fetch_all("2024-01-01", "2026-04-28")
    build_forecast_dataset()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make sure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ── Weather actuals ────────────────────────────────────────────────────────────

def fetch_weather_actuals(
    start: str,
    end: str,
    csv_path: str | Path = DATA_DIR / "weather_actuals_raw.csv",
) -> pd.DataFrame:
    """Fetch historical hourly weather actuals from Open-Meteo Archive API.

    Parameters
    ----------
    start : str  e.g. "2022-01-01"
    end   : str  e.g. "2026-04-28"
    csv_path : path where the CSV is written (default: data/weather_actuals_raw.csv)

    Returns
    -------
    pd.DataFrame with one row per (region, hour).
    """
    from src.data.fetch_weather_actuals_data import fetch_records, write_csv

    print(f"[fetch_weather_actuals] {start} → {end}")
    records = fetch_records(start=start, end=end)
    csv_path = Path(csv_path)
    write_csv(records, csv_path)
    print(f"  {len(records):,} rows saved → {csv_path}")

    df = pd.DataFrame(records)
    if "TimeDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
    return df


# ── Weather forecasts ──────────────────────────────────────────────────────────

def fetch_weather_forecasts(
    start: str,
    end: str,
    csv_path: str | Path = DATA_DIR / "weather_forecasts_raw.csv",
) -> pd.DataFrame:
    """Fetch NWP previous-run forecast data from Open-Meteo Previous Runs API.

    Parameters
    ----------
    start : str  e.g. "2025-11-01"
    end   : str  e.g. "2026-04-28"
    csv_path : path where the CSV is written (default: data/weather_forecasts_raw.csv)

    Returns
    -------
    pd.DataFrame with one row per (region, hour, previous_day_offset).
    """
    from src.data.fetch_weather_forecast_data import fetch_records, write_csv

    print(f"[fetch_weather_forecasts] {start} → {end}")
    records = fetch_records(start=start, end=end)
    csv_path = Path(csv_path)
    write_csv(records, csv_path)
    print(f"  {len(records):,} rows saved → {csv_path}")

    df = pd.DataFrame(records)
    if "TimeDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
    return df


# ── Consumption ────────────────────────────────────────────────────────────────

def fetch_consumption(
    start: str,
    end: str,
    price_area: str = "DK1",
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch hourly grid-area electricity consumption from Energi Data Service.

    Parameters
    ----------
    start      : str  e.g. "2021-01-01"
    end        : str  e.g. "2026-04-28"
    price_area : "DK1" or "DK2"
    csv_path   : output path (default: data/consumption_<price_area>_raw.csv)

    Returns
    -------
    pd.DataFrame  with columns Date, TimeUTC, TimeDK, PriceArea, GridArea,
                  GridCompanyName, ConsumptionkWh.
    """
    from src.data.fetch_consumption_data import fetch_records, write_csv

    if csv_path is None:
        csv_path = DATA_DIR / f"consumption_{price_area.lower()}_raw.csv"
    csv_path = Path(csv_path)

    print(f"[fetch_consumption] {start} → {end}  area={price_area}")
    records = fetch_records(start=start, end=end, price_area=[price_area])
    write_csv(records, csv_path)
    print(f"  {len(records):,} rows saved → {csv_path}")

    df = pd.DataFrame(records)
    if "TimeDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
    return df


# ── Supply forecasts ───────────────────────────────────────────────────────────

def fetch_supply(
    start: str,
    end: str,
    price_area: str = "DK1",
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch hourly wind/solar supply forecasts from Energi Data Service.

    Parameters
    ----------
    start      : str  e.g. "2021-01-01"
    end        : str  e.g. "2026-04-28"
    price_area : "DK1" or "DK2"
    csv_path   : output path (default: data/supply_forecasts_<price_area>_raw.csv)

    Returns
    -------
    pd.DataFrame  with ForecastDayAhead, ForecastIntraday, Forecast5Hour, etc.
    """
    from src.data.fetch_supply_forecast_data import fetch_records, write_csv

    if csv_path is None:
        csv_path = DATA_DIR / f"supply_forecasts_{price_area.lower()}_raw.csv"
    csv_path = Path(csv_path)

    print(f"[fetch_supply] {start} → {end}  area={price_area}")
    records = fetch_records(start=start, end=end, price_area=[price_area])
    write_csv(records, csv_path)
    print(f"  {len(records):,} rows saved → {csv_path}")

    df = pd.DataFrame(records)
    for col in ("HourUTC", "HourDK"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ── Day-ahead prices ───────────────────────────────────────────────────────────

def fetch_day_ahead_prices(
    start: str,
    end: str,
    price_area: str = "DK1",
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch and merge Elspotprices + DayAheadPrices from Energi Data Service.

    Parameters
    ----------
    start      : str  e.g. "2021-01-01"
    end        : str  e.g. "2026-04-28"
    price_area : "DK1" or "DK2"
    csv_path   : output path (default: data/day_ahead_prices_<price_area>_raw.csv)

    Returns
    -------
    pd.DataFrame  with columns TimeUTC, TimeDK, PriceArea,
                  DayAheadPriceEUR, DayAheadPriceDKK, _source_dataset.
    """
    from src.data.fetch_day_ahead_price_data import (
        fetch_records_from_url,
        normalize_dayahead_record,
        normalize_elspot_record,
        deduplicate_records,
        write_csv,
        DAYAHEAD_URL,
        ELSPOT_URL,
        DAYAHEAD_COLUMNS,
        ELSPOT_COLUMNS,
    )

    if csv_path is None:
        csv_path = DATA_DIR / f"day_ahead_prices_{price_area.lower()}_raw.csv"
    csv_path = Path(csv_path)

    print(f"[fetch_day_ahead_prices] {start} → {end}  area={price_area}")

    elspot_raw = fetch_records_from_url(
        ELSPOT_URL, start=start, end=end,
        price_area=[price_area],
        sort="HourUTC desc,PriceArea",
        columns=ELSPOT_COLUMNS,
    )
    dayahead_raw = fetch_records_from_url(
        DAYAHEAD_URL, start=start, end=end,
        price_area=[price_area],
        sort="TimeUTC desc,PriceArea",
        columns=DAYAHEAD_COLUMNS,
    )

    print(f"  Elspotprices: {len(elspot_raw):,} rows | DayAheadPrices: {len(dayahead_raw):,} rows")

    merged = (
        [normalize_elspot_record(r) for r in elspot_raw]
        + [normalize_dayahead_record(r) for r in dayahead_raw]
    )
    merged = deduplicate_records(merged)
    write_csv(merged, csv_path)
    print(f"  {len(merged):,} merged rows saved → {csv_path}")

    df = pd.DataFrame(merged)
    for col in ("TimeUTC", "TimeDK"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ── Forsoeg dataset (XGBoost model input) ─────────────────────────────────────

def build_forecast_dataset(
    actuals_csv: str | Path = DATA_DIR / "weather_actuals_raw.csv",
    error_dist_csv: str | Path = DATA_DIR / "weather_error_distributions.csv",
    output_path: str | Path = DATA_DIR / "forecast_dataset.parquet",
    seed: int = 42,
) -> pd.DataFrame:
    """Build the synthetic forecast dataset used by XGBoost (forecast_dataset.parquet).

    Reads weather actuals and error distributions, simulates NWP-style forecasts
    by adding horizon-scaled Gaussian noise, and saves the result as a parquet file.

    Parameters
    ----------
    actuals_csv    : path to weather_actuals_raw.csv
    error_dist_csv : path to weather_error_distributions.csv
    output_path    : where to write the parquet (default: data/forecast_dataset.parquet)
    seed           : random seed for reproducibility (default: 42)

    Returns
    -------
    pd.DataFrame  with columns region, issue_time, target_time, horizon_h,
                  fcst_*, actual_*, and time features.
    """
    import numpy as np

    actuals_csv  = Path(actuals_csv)
    error_dist_csv = Path(error_dist_csv)
    output_path  = Path(output_path)

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
            sub = err_dist[err_dist["forecast_variable"] == fcst_var].set_index("horizon_hours")
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
        rdf        = rdf.set_index("time").sort_index()
        all_times  = rdf.index
        cutoff     = all_times[-1] - pd.Timedelta(hours=int(FORECAST_HORIZON_HOURS))
        valid_times = all_times[all_times <= cutoff]
        issue_times = valid_times[::ISSUE_INTERVAL_HOURS]
        print(f"  {region}: {len(issue_times)} issue times × {len(HORIZONS)} horizons "
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
    print(f"  {len(df):,} rows × {df.shape[1]} cols saved → {output_path}")

    # Slim dashboard file (DK1_west only)
    dashboard_cols = [
        "region", "issue_time", "target_time", "horizon_h",
        "fcst_wind_speed_100m",    "actual_wind_speed_100m",
        "fcst_shortwave_radiation","actual_shortwave_radiation",
        "fcst_temperature_2m",     "actual_temperature_2m",
    ]
    dash_df   = df.loc[df["region"] == "DK1_west", dashboard_cols].copy()
    dash_path = DATA_DIR / "weather_dk1_dashboard.parquet"
    dash_df.to_parquet(dash_path, index=False, compression="snappy")
    print(f"  Dashboard file: {dash_path}  ({len(dash_df):,} rows)")

    return df


# ── Convenience: fetch everything ─────────────────────────────────────────────

def fetch_all(start: str, end: str, price_area: str = "DK1") -> dict[str, pd.DataFrame]:
    """Fetch all raw data sources for the given date range.

    Runs the five fetch functions in order:
      1. weather actuals
      2. weather forecasts
      3. consumption
      4. supply forecasts
      5. day-ahead prices

    Parameters
    ----------
    start      : str  e.g. "2021-01-01"
    end        : str  e.g. "2026-04-28"
    price_area : "DK1" or "DK2" (applied to EDS datasets)

    Returns
    -------
    dict with keys: "weather_actuals", "weather_forecasts", "consumption",
                    "supply", "prices"
    """
    print("=" * 60)
    print(f"fetch_all  {start} → {end}  (area={price_area})")
    print("=" * 60)

    results = {}
    results["weather_actuals"]   = fetch_weather_actuals(start, end)
    results["weather_forecasts"] = fetch_weather_forecasts(start, end)
    results["consumption"]       = fetch_consumption(start, end, price_area=price_area)
    results["supply"]            = fetch_supply(start, end, price_area=price_area)
    results["prices"]            = fetch_day_ahead_prices(start, end, price_area=price_area)

    print("\n" + "=" * 60)
    print("fetch_all complete.  Run build_forecast_dataset() next to")
    print("regenerate the model input dataset.")
    print("=" * 60)
    return results


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch all raw data and/or rebuild the forsoeg dataset."
    )
    parser.add_argument("--start", required=True, help="Start date, e.g. 2021-01-01")
    parser.add_argument("--end",   required=True, help="End date, e.g. 2026-04-28")
    parser.add_argument("--area",  default="DK1", help="Price area (default: DK1)")
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip fetching raw data (use existing CSVs)"
    )
    parser.add_argument(
        "--build-dataset", action="store_true",
        help="Build forecast_dataset.parquet after fetching"
    )
    args = parser.parse_args()

    if not args.skip_fetch:
        fetch_all(args.start, args.end, price_area=args.area)

    if args.build_dataset:
        build_forecast_dataset()
