#!/usr/bin/env python3
"""Simulate weather forecasts from actual weather and historical forecast errors.

The script uses the estimated error distributions from
``weather_error_distributions.csv`` to create synthetic forecast values:

    simulated_forecast = actual_value + sampled_error

where ``sampled_error`` is drawn from a normal distribution with the
historical ``mean_error`` and ``std_error`` for each weather variable and
forecast horizon. The simulation draws the error magnitude and sign
separately, so forecast errors can be either positive or negative.

Outputs
-------
1. ``sim_weather_forecasts_and_actuals.csv``
   Original actual weather columns plus ``<variable>_forecast`` columns.
2. ``sim_weather_forecast.csv``
   Same schema as ``weather_actuals_raw.csv``, but weather predictor columns
   contain simulated forecast values.

Example
-------
python src/data/simulate_weather_forecasts.py \
    --horizon-hours 120 \
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ACTUAL_CSV = Path("data/weather_actuals_raw.csv")
DEFAULT_ERROR_CSV = Path("data/weather_error_distributions.csv")
DEFAULT_ACTUALS_AND_FORECASTS_CSV = Path("data/sim_weather_forecasts_and_actuals.csv")
DEFAULT_FORECAST_CSV = Path("data/sim_weather_forecast.csv")


def load_actuals(path: Path) -> pd.DataFrame:
    """Load actual weather data and parse ``TimeDK`` when present."""
    df = pd.read_csv(path)
    if "TimeDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
    return df


def load_error_distributions(path: Path, horizon_hours: int) -> pd.DataFrame:
    """Load error rows for one forecast horizon."""
    df = pd.read_csv(path)
    required = {"actual_variable", "horizon_hours", "mean_error", "std_error"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Error distribution file is missing columns: {sorted(missing)}")

    df = df[df["horizon_hours"] == horizon_hours].copy()
    if df.empty:
        available = sorted(pd.read_csv(path)["horizon_hours"].dropna().unique().tolist())
        raise ValueError(
            f"No error distributions found for horizon {horizon_hours}. "
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


def sample_signed_errors(
    rng: np.random.Generator,
    mean_error: float,
    std_error: float,
    size: int,
) -> np.ndarray:
    """Sample random forecast errors with random sign and random magnitude.

    The magnitude is based on the historical mean/std of the error
    distribution. The sign is drawn independently, so each simulated forecast
    can end up above or below the actual value.
    """
    magnitude = np.abs(
        rng.normal(
            loc=abs(mean_error),
            scale=max(std_error, 0.0),
            size=size,
        )
    )
    sign = rng.choice([-1.0, 1.0], size=size)
    return sign * magnitude


def simulate_forecast_values(
    actual_df: pd.DataFrame,
    error_df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a forecast DataFrame with the same schema as the actual weather data."""
    rng = np.random.default_rng(seed)
    forecast_df = actual_df.copy()
    simulated_variables: list[str] = []

    for row in error_df.itertuples(index=False):
        variable = str(row.actual_variable)
        if variable not in actual_df.columns:
            continue

        actual_values = pd.to_numeric(actual_df[variable], errors="coerce")
        std_error = 0.0 if pd.isna(row.std_error) else float(row.std_error)
        mean_error = 0.0 if pd.isna(row.mean_error) else float(row.mean_error)

        sampled_errors = sample_signed_errors(
            rng=rng,
            mean_error=mean_error,
            std_error=std_error,
            size=len(actual_df),
        )
        simulated_values = actual_values.to_numpy(dtype=float) + sampled_errors
        simulated_values = apply_physical_bounds(variable, simulated_values)

        forecast_df[variable] = simulated_values
        simulated_variables.append(variable)

    if not simulated_variables:
        raise ValueError("No actual weather columns matched the error distribution variables.")

    return forecast_df, simulated_variables


def build_actuals_and_forecasts_df(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    simulated_variables: list[str],
) -> pd.DataFrame:
    """Combine actual weather columns with matching ``_forecast`` columns."""
    combined = actual_df.copy()
    for variable in simulated_variables:
        combined[f"{variable}_forecast"] = forecast_df[variable]
    return combined


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
        help="Output CSV with the same columns as the actual weather CSV, containing simulated forecasts.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=120,
        help="Forecast horizon to simulate from the error distribution file. Default is 120 hours / 5 days.",
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
    error_df = load_error_distributions(args.error_csv, args.horizon_hours)
    forecast_df, simulated_variables = simulate_forecast_values(
        actual_df=actual_df,
        error_df=error_df,
        seed=args.seed,
    )
    actuals_and_forecasts_df = build_actuals_and_forecasts_df(
        actual_df=actual_df,
        forecast_df=forecast_df,
        simulated_variables=simulated_variables,
    )

    write_outputs(
        actuals_and_forecasts_df=actuals_and_forecasts_df,
        forecast_df=forecast_df,
        actuals_and_forecasts_csv=args.actuals_and_forecasts_csv,
        forecast_csv=args.forecast_csv,
    )

    print(f"Simulated horizon: {args.horizon_hours} hours")
    print(f"Rows: {len(actual_df):,}")
    print(f"Variables simulated: {', '.join(simulated_variables)}")
    print(f"Saved actuals + forecasts to {args.actuals_and_forecasts_csv}")
    print(f"Saved forecasts only to {args.forecast_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
