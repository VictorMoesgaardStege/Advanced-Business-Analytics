#!/usr/bin/env python3
"""Simulate multi-horizon weather forecasts from actuals and forecast errors.

The script uses the estimated error distributions from
``weather_error_distributions.csv`` to create synthetic forecast values:

    simulated_forecast_h = actual_value + sampled_error_h

where ``sampled_error_h`` is drawn from the historical error distribution
for the relevant weather variable and forecast horizon. The error summary
CSV stores the forecast horizon in ``horizon_hours`` and the estimated error
distribution as ``mean_error`` and ``std_error``.

Outputs
-------
1. ``sim_weather_forecasts_and_actuals.csv``
   Original actual weather columns plus ``<variable>_<horizon>`` columns.
2. ``sim_weather_forecast.csv``
   Metadata columns plus simulated ``<variable>_<horizon>`` forecast columns.

Example
-------
python src/data/simulate_weather_forecasts.py \
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ACTUAL_CSV = Path("data/weather_actuals_raw.csv")
DEFAULT_ERROR_CSV = Path("data/weather_error_distributions.csv")
DEFAULT_ACTUALS_AND_FORECASTS_CSV = Path("processed_data/sim_weather_forecasts_and_actuals.csv")
DEFAULT_FORECAST_CSV = Path("processed_data/sim_weather_forecast.csv")


def load_actuals(path: Path) -> pd.DataFrame:
    """Load actual weather data and parse ``TimeDK`` when present."""
    df = pd.read_csv(path)
    if "TimeDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
    df = df[df["region"] == "DK1_west"]
    return df


def load_error_distributions(path: Path, horizons: list[int] | None = None) -> pd.DataFrame:
    """Load error rows for the selected forecast horizons."""
    df = pd.read_csv(path)
    required = {"actual_variable", "horizon_hours", "mean_error", "std_error"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Error distribution file is missing columns: {sorted(missing)}")

    if horizons is not None:
        df = df[df["horizon_hours"].isin(horizons)].copy()

    if df.empty:
        available = sorted(pd.read_csv(path)["horizon_hours"].dropna().astype(int).unique().tolist())
        raise ValueError(
            f"No error distributions found for horizons {horizons}. "
            f"Available horizons: {available}"
        )

    return df


def wrap_degrees(values: pd.Series | np.ndarray) -> np.ndarray:
    """Wrap wind directions to [0, 360)."""
    return np.asarray(values) % 360


def apply_physical_bounds(variable: str, values: np.ndarray) -> np.ndarray:
    """Keep simulated weather forecasts within simple physical bounds."""
    if variable.startswith("wind_speed"):
        return np.clip(values, 0, None)
    if variable.startswith("wind_direction"):
        return wrap_degrees(values)
    if variable == "shortwave_radiation":
        return np.clip(values, 0, None)
    if variable == "cloud_cover":
        return np.clip(values, 0, 100)
    return values


def sample_errors(
    rng: np.random.Generator,
    mean_error: float,
    std_error: float,
    size: int,
) -> np.ndarray:
    """Sample random forecast errors from the estimated error distribution."""
    return rng.normal(
        loc=mean_error,
        scale=max(std_error, 0.0),
        size=size,
    )


def metadata_columns(actual_df: pd.DataFrame, weather_variables: set[str]) -> list[str]:
    """Return non-weather columns that should be copied to forecast-only output."""
    return [col for col in actual_df.columns if col not in weather_variables]


def simulate_forecast_columns(
    actual_df: pd.DataFrame,
    error_df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return combined actual/forecast data and forecast-only data."""
    rng = np.random.default_rng(seed)
    combined_df = actual_df.copy()
    forecast_columns: dict[str, np.ndarray] = {}

    for row in error_df.itertuples(index=False):
        variable = str(row.actual_variable)
        if variable not in actual_df.columns:
            continue

        horizon_hours = int(row.horizon_hours)
        forecast_col = f"{variable}_{horizon_hours}"
        actual_values = pd.to_numeric(actual_df[variable], errors="coerce")
        std_error = 0.0 if pd.isna(row.std_error) else float(row.std_error)
        mean_error = 0.0 if pd.isna(row.mean_error) else float(row.mean_error)

        sampled_errors = sample_errors(
            rng=rng,
            mean_error=mean_error,
            std_error=std_error,
            size=len(actual_df),
        )
        simulated_values = actual_values.to_numpy(dtype=float) + sampled_errors
        simulated_values = apply_physical_bounds(variable, simulated_values)

        combined_df[forecast_col] = simulated_values
        forecast_columns[forecast_col] = simulated_values

    if not forecast_columns:
        raise ValueError("No actual weather columns matched the error distribution variables.")

    weather_variables = set(error_df["actual_variable"].astype(str))
    forecast_df = actual_df[metadata_columns(actual_df, weather_variables)].copy()
    for col, values in forecast_columns.items():
        forecast_df[col] = values

    return combined_df, forecast_df, list(forecast_columns.keys())


def write_outputs(
    actuals_and_forecasts_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    actuals_and_forecasts_csv: Path,
    forecast_csv: Path,
) -> None:
    """Write both requested CSV outputs."""
    actuals_and_forecasts_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast_csv.parent.mkdir(parents=True, exist_ok=True)

    actuals_and_forecasts_df.to_csv(actuals_and_forecasts_csv, index=False)
    forecast_df.to_csv(forecast_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate weather forecast data from actuals and estimated forecast errors."
    )
    parser.add_argument(
        "--actual-csv",
        type=Path,
        default=DEFAULT_ACTUAL_CSV,
        help="Input actual weather CSV.",
    )
    parser.add_argument(
        "--error-csv",
        type=Path,
        default=DEFAULT_ERROR_CSV,
        help="Input weather error-distribution summary CSV.",
    )
    parser.add_argument(
        "--actuals-and-forecasts-csv",
        type=Path,
        default=DEFAULT_ACTUALS_AND_FORECASTS_CSV,
        help="Output CSV with actual variables and matching *_forecast columns.",
    )
    parser.add_argument(
        "--forecast-csv",
        type=Path,
        default=DEFAULT_FORECAST_CSV,
        help="Output CSV with metadata columns and simulated <variable>_<horizon> forecasts.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        default=None,
        help="Forecast horizons to simulate. Defaults to all horizons in the error distribution file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible simulated forecasts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    actual_df = load_actuals(args.actual_csv)
    error_df = load_error_distributions(args.error_csv, args.horizons)
    actuals_and_forecasts_df, forecast_df, forecast_columns = simulate_forecast_columns(
        actual_df=actual_df,
        error_df=error_df,
        seed=args.seed,
    )

    write_outputs(
        actuals_and_forecasts_df=actuals_and_forecasts_df,
        forecast_df=forecast_df,
        actuals_and_forecasts_csv=args.actuals_and_forecasts_csv,
        forecast_csv=args.forecast_csv,
    )

    horizons = sorted(error_df["horizon_hours"].astype(int).unique().tolist())
    print(f"Simulated horizons: {', '.join(str(h) for h in horizons)} hours")
    print(f"Rows: {len(actual_df):,}")
    print(f"Forecast columns generated: {len(forecast_columns)}")
    print(f"Saved actuals + forecasts to {args.actuals_and_forecasts_csv}")
    print(f"Saved forecasts only to {args.forecast_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
