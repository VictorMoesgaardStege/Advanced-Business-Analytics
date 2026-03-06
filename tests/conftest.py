"""Shared test fixtures and helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_price_series(n_hours: int = 500, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic hourly price time-series for testing.

    Prices follow a simple seasonal pattern plus noise, mimicking realistic
    electricity spot price dynamics.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_hours, freq="h")

    # Seasonal + daily pattern
    hour_effect = 50 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    day_effect = 20 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 7))
    noise = rng.normal(0, 30, n_hours)
    prices = 400 + hour_effect + day_effect + noise

    df = pd.DataFrame({"SpotPriceDKK": np.maximum(prices, 0.0)}, index=idx)
    df.index.name = "HourDK"
    return df
