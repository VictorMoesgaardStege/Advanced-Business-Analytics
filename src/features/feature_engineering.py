"""
Feature engineering for electricity price forecasting.

Transforms raw hourly price data into a rich feature matrix suitable for
gradient-boosted tree models (LightGBM).  All features are derived from
the price time-series itself plus calendar/clock information — no external
data sources are required.
"""

from __future__ import annotations

import logging
from typing import List

import holidays
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lags (in hours) used to create look-back price features
LAG_HOURS: List[int] = [1, 2, 3, 6, 12, 24, 48, 72, 168]

# Rolling window sizes (in hours) for statistics
ROLLING_WINDOWS: List[int] = [6, 12, 24, 48, 168]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and clock features derived from the index.

    Expects *df* to have a ``DatetimeIndex`` (``HourDK``).

    Added columns
    -------------
    ``hour``, ``day_of_week``, ``month``, ``year``,
    ``is_weekend``, ``is_danish_holiday``,
    ``hour_sin``, ``hour_cos`` (cyclical encoding),
    ``dow_sin``, ``dow_cos`` (cyclical encoding),
    ``month_sin``, ``month_cos`` (cyclical encoding).
    """
    df = df.copy()
    idx = df.index

    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek        # 0 = Monday
    df["month"] = idx.month
    df["year"] = idx.year
    df["day_of_year"] = idx.dayofyear
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Danish public holidays
    dk_holidays = holidays.Denmark(years=sorted(idx.year.unique().tolist()))
    df["is_danish_holiday"] = idx.normalize().isin(dk_holidays).astype(int)

    # Cyclical encodings to preserve circular continuity
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def add_lag_features(df: pd.DataFrame, lags: List[int] = LAG_HOURS) -> pd.DataFrame:
    """Add lagged price features.

    Parameters
    ----------
    df:
        DataFrame containing the ``SpotPriceDKK`` column.
    lags:
        List of lag offsets in hours.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``lag_<n>h`` columns.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df["SpotPriceDKK"].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, windows: List[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """Add rolling-window statistics over the price series.

    Parameters
    ----------
    df:
        DataFrame containing the ``SpotPriceDKK`` column.
    windows:
        List of window sizes in hours.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``rolling_mean_<n>h``,
        ``rolling_std_<n>h``, and ``rolling_min/max_<n>h`` columns.
    """
    df = df.copy()
    price = df["SpotPriceDKK"]
    for w in windows:
        rolled = price.shift(1).rolling(window=w, min_periods=max(1, w // 2))
        df[f"rolling_mean_{w}h"] = rolled.mean()
        df[f"rolling_std_{w}h"] = rolled.std()
        df[f"rolling_min_{w}h"] = rolled.min()
        df[f"rolling_max_{w}h"] = rolled.max()
    return df


def add_price_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add short-term trend and difference features.

    Added columns
    -------------
    ``price_diff_1h``, ``price_diff_24h``,
    ``price_pct_change_24h``, ``price_pct_change_168h``.
    """
    df = df.copy()
    price = df["SpotPriceDKK"]
    df["price_diff_1h"] = price.diff(1)
    df["price_diff_24h"] = price.diff(24)
    df["price_pct_change_24h"] = price.pct_change(24)
    df["price_pct_change_168h"] = price.pct_change(168)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    forecast_horizon: int = 1,
    lags: List[int] = LAG_HOURS,
    rolling_windows: List[int] = ROLLING_WINDOWS,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix for a given forecast horizon.

    The target column ``y`` contains the electricity price
    ``forecast_horizon`` hours ahead from the current row.

    Parameters
    ----------
    df:
        Raw DataFrame with ``SpotPriceDKK`` column and ``DatetimeIndex``.
    forecast_horizon:
        Number of hours ahead to predict (default: ``1``).
    lags:
        Lag offsets to include.
    rolling_windows:
        Rolling window sizes to include.
    drop_na:
        Whether to drop rows with any NaN feature values (default: ``True``).

    Returns
    -------
    pd.DataFrame
        Feature matrix with the target column ``y`` appended.
    """
    df = df.copy()

    # Create target: price h hours ahead
    df["y"] = df["SpotPriceDKK"].shift(-forecast_horizon)

    df = add_calendar_features(df)
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=rolling_windows)
    df = add_price_trend_features(df)

    if drop_na:
        before = len(df)
        df = df.dropna()
        logger.info(
            "Dropped %d rows with NaN values (horizon=%dh).",
            before - len(df),
            forecast_horizon,
        )

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all feature column names (everything except ``SpotPriceDKK`` and ``y``).

    Parameters
    ----------
    df:
        Feature matrix produced by :func:`build_feature_matrix`.

    Returns
    -------
    List[str]
        Ordered list of feature column names.
    """
    excluded = {"SpotPriceDKK", "y"}
    return [c for c in df.columns if c not in excluded]
