"""
Decision support layer: translating price forecasts into consumer recommendations.

This module converts a probabilistic electricity price forecast into
plain-language guidance that helps consumers decide *when* to schedule
flexible energy consumption (e.g. EV charging, dishwasher, heat pump).

Public API
----------
- :func:`generate_recommendations` — main entry point.
- :class:`Recommendation` — a single recommendation.
- :func:`summarise_forecast` — compute descriptive statistics over a forecast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Percentile thresholds (relative to the training price distribution) used to
# classify a forecasted hour as "cheap", "normal", or "expensive".
CHEAP_PERCENTILE = 30
EXPENSIVE_PERCENTILE = 70

# Minimum price drop (DKK/MWh) required to generate a "wait and save" message
MIN_MEANINGFUL_DROP_DKK = 50


@dataclass
class Recommendation:
    """A single consumer recommendation derived from a price forecast.

    Attributes
    ----------
    horizon_h:
        Number of hours ahead the recommendation refers to.
    category:
        One of ``"cheap"``, ``"expensive"``, or ``"normal"``.
    message:
        Human-readable guidance string.
    estimated_saving_dkk_per_mwh:
        Estimated saving (in DKK/MWh) relative to the current price, or
        ``None`` when not applicable.
    lower_price_dkk:
        Lower-bound price prediction (q0.10) for this horizon.
    median_price_dkk:
        Median price prediction (q0.50) for this horizon.
    upper_price_dkk:
        Upper-bound price prediction (q0.90) for this horizon.
    """

    horizon_h: int
    category: str
    message: str
    estimated_saving_dkk_per_mwh: Optional[float] = None
    lower_price_dkk: Optional[float] = None
    median_price_dkk: Optional[float] = None
    upper_price_dkk: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "horizon_h": self.horizon_h,
            "category": self.category,
            "message": self.message,
            "estimated_saving_dkk_per_mwh": self.estimated_saving_dkk_per_mwh,
            "lower_price_dkk": self.lower_price_dkk,
            "median_price_dkk": self.median_price_dkk,
            "upper_price_dkk": self.upper_price_dkk,
        }


def summarise_forecast(forecast_df: pd.DataFrame) -> dict:
    """Compute summary statistics over a full 7-day forecast.

    Parameters
    ----------
    forecast_df:
        DataFrame produced by
        :meth:`~src.models.forecasting.ElectricityPriceForecaster.predict`,
        with horizons as the index and ``q0.10``, ``q0.50``, ``q0.90``
        columns.

    Returns
    -------
    dict
        Keys: ``min_median``, ``max_median``, ``mean_median``,
        ``cheapest_horizon_h``, ``most_expensive_horizon_h``,
        ``price_range_dkk``.
    """
    medians = forecast_df["q0.50"]
    return {
        "min_median": float(medians.min()),
        "max_median": float(medians.max()),
        "mean_median": float(medians.mean()),
        "cheapest_horizon_h": int(medians.idxmin()),
        "most_expensive_horizon_h": int(medians.idxmax()),
        "price_range_dkk": float(medians.max() - medians.min()),
    }


def _classify_price(
    price: float,
    cheap_threshold: float,
    expensive_threshold: float,
) -> str:
    """Classify a price as 'cheap', 'normal', or 'expensive'."""
    if price <= cheap_threshold:
        return "cheap"
    if price >= expensive_threshold:
        return "expensive"
    return "normal"


def _hours_to_friendly(hours: int) -> str:
    """Convert a horizon in hours to a friendly string."""
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    remainder = hours % 24
    day_str = f"{days} day{'s' if days != 1 else ''}"
    if remainder == 0:
        return day_str
    return f"{day_str} and {remainder} hour{'s' if remainder != 1 else ''}"


def generate_recommendations(
    forecast_df: pd.DataFrame,
    current_price_dkk: float,
    historical_prices: Optional[pd.Series] = None,
    cheap_percentile: float = CHEAP_PERCENTILE,
    expensive_percentile: float = EXPENSIVE_PERCENTILE,
) -> List[Recommendation]:
    """Generate consumer-friendly recommendations from a probabilistic forecast.

    The function identifies the cheapest future window and highlights
    unusually expensive upcoming periods.  Recommendations are ordered from
    most to least actionable.

    Parameters
    ----------
    forecast_df:
        Forecast DataFrame (horizons × quantile columns ``q0.10``, ``q0.50``,
        ``q0.90``).
    current_price_dkk:
        The current spot price in DKK/MWh (used to compute potential savings).
    historical_prices:
        Optional series of recent historical prices used to calibrate the
        cheap/expensive thresholds.  If omitted, thresholds are derived from
        *forecast_df* itself.
    cheap_percentile:
        Prices below this percentile are classified as "cheap".
    expensive_percentile:
        Prices above this percentile are classified as "expensive".

    Returns
    -------
    List[Recommendation]
        Ordered list of recommendations; the first entry describes the
        overall best saving opportunity.
    """
    medians = forecast_df["q0.50"]
    lowers = forecast_df.get("q0.10", medians)
    uppers = forecast_df.get("q0.90", medians)

    # Determine classification thresholds
    if historical_prices is not None and len(historical_prices) >= 10:
        ref_prices = np.concatenate([historical_prices.values, medians.values])
    else:
        ref_prices = medians.values

    cheap_threshold = float(np.percentile(ref_prices, cheap_percentile))
    expensive_threshold = float(np.percentile(ref_prices, expensive_percentile))

    recommendations: List[Recommendation] = []

    # --- Best saving opportunity ---
    cheapest_h = int(medians.idxmin())
    cheapest_price = float(medians[cheapest_h])
    saving = current_price_dkk - cheapest_price

    if saving >= MIN_MEANINGFUL_DROP_DKK:
        category = _classify_price(cheapest_price, cheap_threshold, expensive_threshold)
        friendly_time = _hours_to_friendly(cheapest_h)
        message = (
            f"If you can wait {friendly_time}, electricity prices are forecast "
            f"to drop to around {cheapest_price:.0f} DKK/MWh — a potential "
            f"saving of ~{saving:.0f} DKK/MWh compared to now."
        )
        recommendations.append(
            Recommendation(
                horizon_h=cheapest_h,
                category=category,
                message=message,
                estimated_saving_dkk_per_mwh=round(saving, 1),
                lower_price_dkk=round(float(lowers[cheapest_h]), 1),
                median_price_dkk=round(cheapest_price, 1),
                upper_price_dkk=round(float(uppers[cheapest_h]), 1),
            )
        )

    # --- Upcoming expensive periods ---
    expensive_horizons = [
        h for h in medians.index if medians[h] >= expensive_threshold
    ]
    for h in expensive_horizons[:3]:   # cap at 3 warnings
        price = float(medians[h])
        friendly_time = _hours_to_friendly(h)
        message = (
            f"Prices are forecast to be unusually high in {friendly_time} "
            f"(~{price:.0f} DKK/MWh). Consider running energy-intensive "
            f"appliances before or after this window."
        )
        recommendations.append(
            Recommendation(
                horizon_h=h,
                category="expensive",
                message=message,
                lower_price_dkk=round(float(lowers[h]), 1),
                median_price_dkk=round(price, 1),
                upper_price_dkk=round(float(uppers[h]), 1),
            )
        )

    # --- Near-term cheap periods ---
    cheap_horizons = [
        h for h in medians.index
        if medians[h] <= cheap_threshold and h <= 24
    ]
    for h in cheap_horizons[:3]:
        price = float(medians[h])
        friendly_time = _hours_to_friendly(h)
        message = (
            f"Electricity is expected to be cheap in {friendly_time} "
            f"(~{price:.0f} DKK/MWh). A good window for high-consumption tasks."
        )
        recommendations.append(
            Recommendation(
                horizon_h=h,
                category="cheap",
                message=message,
                lower_price_dkk=round(float(lowers[h]), 1),
                median_price_dkk=round(price, 1),
                upper_price_dkk=round(float(uppers[h]), 1),
            )
        )

    # --- Fallback: no strong signal ---
    if not recommendations:
        avg_price = float(medians.mean())
        message = (
            f"Prices over the next 7 days are relatively stable, averaging "
            f"~{avg_price:.0f} DKK/MWh. No strong incentive to shift consumption."
        )
        recommendations.append(
            Recommendation(
                horizon_h=int(medians.idxmin()),
                category="normal",
                message=message,
                median_price_dkk=round(avg_price, 1),
            )
        )

    return recommendations


def recommendations_to_dataframe(recs: List[Recommendation]) -> pd.DataFrame:
    """Convert a list of :class:`Recommendation` objects to a DataFrame.

    Parameters
    ----------
    recs:
        List of recommendations.

    Returns
    -------
    pd.DataFrame
        One row per recommendation.
    """
    return pd.DataFrame([r.to_dict() for r in recs])
