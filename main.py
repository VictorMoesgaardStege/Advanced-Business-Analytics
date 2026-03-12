"""
Main pipeline for the DK1 electricity price forecasting system.

Usage examples
--------------
Train from the Energi Data Service API::

    python main.py --mode train --start 2021-01-01 --end 2024-01-01

Train from a pre-downloaded CSV::

    python main.py --mode train --csv data/spot_prices_DK1.csv

Run forecaster on the latest available data and show recommendations::

    python main.py --mode predict --model models/forecaster.joblib

Full pipeline (fetch → train → evaluate → recommend)::

    python main.py --mode full --start 2021-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.fetch_data_NOTused import get_default_date_range, load_from_csv, load_or_fetch
from src.evaluation.metrics import evaluate_forecaster
from src.models.forecasting import (
    DEFAULT_HORIZONS,
    ElectricityPriceForecaster,
    time_series_split,
    train_forecaster,
)
from src.recommendations.decision_support import (
    generate_recommendations,
    recommendations_to_dataframe,
    summarise_forecast,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/forecaster.joblib")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Fetch data, train models, save forecaster."""
    raw_df = _load_data(args)

    train_df, test_df = time_series_split(raw_df, test_fraction=0.15)

    logger.info("Training forecaster …")
    forecaster = train_forecaster(train_df, horizons=DEFAULT_HORIZONS)

    model_path = Path(args.model)
    forecaster.save(model_path)
    logger.info("Model saved to %s", model_path)

    logger.info("Evaluating on test set …")
    metrics_df = evaluate_forecaster(forecaster, test_df)
    print("\n=== Evaluation metrics (test set) ===")
    print(metrics_df.to_string())

    _run_recommendations(forecaster, raw_df)


def cmd_predict(args: argparse.Namespace) -> None:
    """Load a saved forecaster and generate recommendations."""
    forecaster = ElectricityPriceForecaster.load(args.model)
    raw_df = _load_data(args)
    _run_recommendations(forecaster, raw_df)


def cmd_full(args: argparse.Namespace) -> None:
    """Full pipeline: fetch → train → evaluate → recommend."""
    cmd_train(args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load raw price data based on CLI arguments."""
    if hasattr(args, "csv") and args.csv:
        logger.info("Loading data from CSV: %s", args.csv)
        return load_from_csv(args.csv)

    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if not start:
        start, end = get_default_date_range(years=3)
        logger.info("No date range specified; defaulting to %s – %s", start, end)

    return load_or_fetch(start, end, price_area="DK1")


def _run_recommendations(
    forecaster: ElectricityPriceForecaster,
    raw_df: pd.DataFrame,
) -> None:
    """Generate and print recommendations for the latest data point."""
    forecast_df = forecaster.predict_from_raw(raw_df)
    summary = summarise_forecast(forecast_df)

    print("\n=== 7-day price forecast summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    current_price = float(raw_df["SpotPriceDKK"].iloc[-1])
    recs = generate_recommendations(
        forecast_df,
        current_price_dkk=current_price,
        historical_prices=raw_df["SpotPriceDKK"].tail(24 * 30),
    )

    print(f"\n=== Consumer recommendations (current price: {current_price:.0f} DKK/MWh) ===")
    for i, rec in enumerate(recs, start=1):
        print(f"\n[{i}] {rec.message}")
        print(
            f"     Horizon: {rec.horizon_h}h | Category: {rec.category} | "
            f"Price range: {rec.lower_price_dkk} – {rec.upper_price_dkk} DKK/MWh"
        )

    recs_df = recommendations_to_dataframe(recs)
    print("\n=== Recommendations table ===")
    print(recs_df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DK1 electricity price forecasting and recommendation system."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "full"],
        default="full",
        help="Pipeline mode (default: full).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a pre-downloaded CSV file with price data.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date for API data fetch (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for API data fetch (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save/load the trained forecaster model.",
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "train": cmd_train,
        "predict": cmd_predict,
        "full": cmd_full,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
