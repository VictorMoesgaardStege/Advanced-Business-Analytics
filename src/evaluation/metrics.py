"""
Model evaluation for electricity price forecasting.

Provides standard regression metrics (MAE, RMSE, MAPE) as well as
pinball / quantile loss for evaluating probabilistic forecasts and
prediction interval coverage.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point-forecast metrics
# ---------------------------------------------------------------------------

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8
) -> float:
    """Mean absolute percentage error (%).

    Rows where ``|y_true| < eps`` are excluded to avoid division by zero.
    """
    mask = np.abs(y_true) >= eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def symmetric_mape(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8
) -> float:
    """Symmetric mean absolute percentage error (%)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------

def pinball_loss(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float
) -> float:
    """Pinball (quantile) loss.

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predicted quantile values.
    quantile:
        Quantile level (between 0 and 1).
    """
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def interval_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Empirical coverage of a prediction interval.

    Returns the fraction of observations that fall within
    ``[y_lower, y_upper]``.
    """
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(inside))


def interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Mean width of the prediction interval."""
    return float(np.mean(y_upper - y_lower))


# ---------------------------------------------------------------------------
# Evaluation over all horizons
# ---------------------------------------------------------------------------

def evaluate_forecaster(
    forecaster,
    raw_test_df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Evaluate a trained :class:`~src.models.forecasting.ElectricityPriceForecaster`.

    For each horizon, the function builds the feature matrix, generates
    predictions, and computes MAE, RMSE, MAPE, pinball loss for each quantile,
    and 80 % interval coverage (q0.10 – q0.90).

    Parameters
    ----------
    forecaster:
        A trained ``ElectricityPriceForecaster`` instance.
    raw_test_df:
        Held-out raw price data.
    horizons:
        Subset of horizons to evaluate.  Defaults to all trained horizons.

    Returns
    -------
    pd.DataFrame
        One row per horizon with evaluation metrics as columns.
    """
    from src.features.feature_engineering import build_feature_matrix, get_feature_columns

    horizons = horizons or forecaster.horizons
    results = []

    for horizon in horizons:
        feat_df = build_feature_matrix(raw_test_df, forecast_horizon=horizon, drop_na=True)
        if feat_df.empty:
            logger.warning("No data for horizon %dh after feature building.", horizon)
            continue

        feature_cols = get_feature_columns(feat_df)
        X = feat_df[feature_cols]  # DataFrame preserves feature names
        y_true = feat_df["y"].values

        quantile_preds: Dict[float, np.ndarray] = {}
        for q in forecaster.quantiles:
            model = forecaster.models[horizon][q]
            preds = np.maximum(model.predict(X), 0.0)
            quantile_preds[q] = preds

        median_pred = quantile_preds.get(0.5, list(quantile_preds.values())[0])

        row: Dict = {
            "horizon_h": horizon,
            "mae": mean_absolute_error(y_true, median_pred),
            "rmse": root_mean_squared_error(y_true, median_pred),
            "mape": mean_absolute_percentage_error(y_true, median_pred),
            "smape": symmetric_mape(y_true, median_pred),
            "n_samples": len(y_true),
        }

        for q, preds in quantile_preds.items():
            row[f"pinball_q{q:.2f}"] = pinball_loss(y_true, preds, quantile=q)

        lower_key = min(forecaster.quantiles)
        upper_key = max(forecaster.quantiles)
        if lower_key in quantile_preds and upper_key in quantile_preds:
            row["interval_coverage"] = interval_coverage(
                y_true, quantile_preds[lower_key], quantile_preds[upper_key]
            )
            row["interval_width_mean"] = interval_width(
                quantile_preds[lower_key], quantile_preds[upper_key]
            )

        results.append(row)

    return pd.DataFrame(results).set_index("horizon_h")
