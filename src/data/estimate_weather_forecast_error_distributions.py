#!/usr/bin/env python3
"""Estimate forecast error distributions from weather actuals and previous-run forecasts.

This script:
- loads actual weather and forecast weather CSVs
- merges on TimeDK and region
- computes errors:
    error_h = forecast_previous_dayh - actual
- estimates mean/std and some optional percentiles
- writes summary CSV and optional raw error CSV

Example:
    python3 src/data/estimate_weather_forecast_error_distributions.py \
        --actual-csv data/weather_actuals_raw.csv \
        --forecast-csv data/weather_forecasts_raw.csv \
        --summary-csv data/weather_error_distributions.csv \
        --errors-csv data/weather_errors_raw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# forecast column base -> actual column base
VARIABLE_MAP = {
    "temperature_2m": "temperature_2m",
    "pressure_msl": "pressure_msl",
    "cloud_cover": "cloud_cover",
    "shortwave_radiation": "shortwave_radiation",
    "wind_speed_10m": "wind_speed_10m",
    "wind_direction_10m": "wind_direction_10m",
    # approximate hub-height match
    "wind_speed_120m": "wind_speed_100m",
    "wind_direction_120m": "wind_direction_100m",
}

HORIZONS = {
    24: "previous_day1",
    48: "previous_day2",
    72: "previous_day3",
    96: "previous_day4",
    120: "previous_day5",
}


def circular_diff_deg(forecast_deg: pd.Series, actual_deg: pd.Series) -> pd.Series:
    """
    Circular difference for angles in degrees, returned in [-180, 180).
    """
    diff = (forecast_deg - actual_deg + 180) % 360 - 180
    return diff


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
    return df


def merge_actuals_and_forecasts(actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    merged = forecast_df.merge(
        actual_df,
        on=["TimeDK", "region"],
        how="inner",
        suffixes=("_forecastfile", "_actualfile"),
    )
    return merged


def build_error_rows(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for forecast_base, actual_base in VARIABLE_MAP.items():
        # Figure out correct merged column names
        if forecast_base == actual_base:
            forecast_col_base = f"{forecast_base}_forecastfile"
            actual_col_base = f"{actual_base}_actualfile"
        else:
            forecast_col_base = forecast_base
            actual_col_base = actual_base

        if actual_col_base not in merged.columns:
            continue

        for horizon_hours, suffix in HORIZONS.items():
            forecast_col = f"{forecast_base}_{suffix}"

            # For direct matches, previous_day columns belong to forecast file only
            if forecast_col not in merged.columns:
                continue

            subset = merged[["TimeDK", "region", forecast_col, actual_col_base]].copy()
            subset = subset.rename(columns={forecast_col: "forecast_value", actual_col_base: "actual_value"})
            subset = subset.dropna(subset=["forecast_value", "actual_value"])

            if subset.empty:
                continue

            is_direction = "direction" in forecast_base

            if is_direction:
                subset["error"] = circular_diff_deg(subset["forecast_value"], subset["actual_value"])
            else:
                subset["error"] = subset["forecast_value"] - subset["actual_value"]

            subset["forecast_variable"] = forecast_base
            subset["actual_variable"] = actual_base
            subset["horizon_hours"] = horizon_hours

            rows.append(
                subset[
                    [
                        "TimeDK",
                        "region",
                        "forecast_variable",
                        "actual_variable",
                        "horizon_hours",
                        "forecast_value",
                        "actual_value",
                        "error",
                    ]
                ]
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "TimeDK",
                "region",
                "forecast_variable",
                "actual_variable",
                "horizon_hours",
                "forecast_value",
                "actual_value",
                "error",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def summarize_errors(error_df: pd.DataFrame) -> pd.DataFrame:
    if error_df.empty:
        return pd.DataFrame(
            columns=[
                "forecast_variable",
                "actual_variable",
                "horizon_hours",
                "n",
                "mean_error",
                "std_error",
                "mae",
                "rmse",
                "p05_error",
                "p25_error",
                "p50_error",
                "p75_error",
                "p95_error",
            ]
        )

    grouped = error_df.groupby(["forecast_variable", "actual_variable", "horizon_hours"])

    summary = grouped["error"].agg(
        n="count",
        mean_error="mean",
        std_error="std",
        mae=lambda x: np.mean(np.abs(x)),
        rmse=lambda x: np.sqrt(np.mean(np.square(x))),
        p05_error=lambda x: np.quantile(x, 0.05),
        p25_error=lambda x: np.quantile(x, 0.25),
        p50_error=lambda x: np.quantile(x, 0.50),
        p75_error=lambda x: np.quantile(x, 0.75),
        p95_error=lambda x: np.quantile(x, 0.95),
    ).reset_index()

    return summary.sort_values(["forecast_variable", "horizon_hours"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate weather forecast error distributions."
    )
    parser.add_argument(
        "--actual-csv",
        type=Path,
        default=Path("data/weather_actuals_raw.csv"),
        help="Path to historical actual weather CSV",
    )
    parser.add_argument(
        "--forecast-csv",
        type=Path,
        default=Path("data/weather_forecasts_raw.csv"),
        help="Path to previous-runs forecast weather CSV",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("data/weather_error_distributions.csv"),
        help="Output summary CSV path",
    )
    parser.add_argument(
        "--errors-csv",
        type=Path,
        default=Path("data/weather_errors_raw.csv"),
        help="Optional raw error rows CSV path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    actual_df = load_csv(args.actual_csv)
    forecast_df = load_csv(args.forecast_csv)

    merged = merge_actuals_and_forecasts(actual_df, forecast_df)
    print(f"Merged rows: {len(merged):,}")

    error_df = build_error_rows(merged)
    print(f"Error rows: {len(error_df):,}")

    summary_df = summarize_errors(error_df)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False)
    print(f"Saved summary to {args.summary_csv}")

    if args.errors_csv:
        args.errors_csv.parent.mkdir(parents=True, exist_ok=True)
        error_df.to_csv(args.errors_csv, index=False)
        print(f"Saved raw errors to {args.errors_csv}")

    if not summary_df.empty:
        print("\nSummary preview:")
        print(summary_df.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())