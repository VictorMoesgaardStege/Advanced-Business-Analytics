"""
Data acquisition module for DK1 electricity spot prices.

Fetches hourly electricity price data from the Energi Data Service API
(https://api.energidataservice.dk) and optionally loads from/saves to
local CSV files.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ENERGI_DATA_SERVICE_URL = "https://api.energidataservice.dk/dataset/Elspotprices"
DEFAULT_PRICE_AREA = "DK1"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def fetch_spot_prices(
    start_date: str,
    end_date: str,
    price_area: str = DEFAULT_PRICE_AREA,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch hourly spot prices from the Energi Data Service API.

    Parameters
    ----------
    start_date:
        Inclusive start date in ``YYYY-MM-DD`` format.
    end_date:
        Exclusive end date in ``YYYY-MM-DD`` format.
    price_area:
        Price area to query (default ``"DK1"``).
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["HourDK", "SpotPriceDKK"]`` indexed by
        ``HourDK`` (UTC+1, hour-start timestamps).
    """
    params = {
        "start": start_date,
        "end": end_date,
        "filter": f'{{"PriceArea":"{price_area}"}}',
        "columns": "HourDK,SpotPriceDKK",
        "sort": "HourDK asc",
        "limit": 100_000,
    }

    logger.info("Fetching spot prices from %s to %s for %s", start_date, end_date, price_area)
    response = requests.get(ENERGI_DATA_SERVICE_URL, params=params, timeout=timeout)
    response.raise_for_status()

    records = response.json().get("records", [])
    if not records:
        raise ValueError(
            f"No data returned for price area {price_area!r} "
            f"between {start_date} and {end_date}."
        )

    df = pd.DataFrame(records)
    df["HourDK"] = pd.to_datetime(df["HourDK"])
    df = df.sort_values("HourDK").reset_index(drop=True)
    df = df.set_index("HourDK")
    df["SpotPriceDKK"] = pd.to_numeric(df["SpotPriceDKK"], errors="coerce")

    logger.info("Fetched %d hourly records.", len(df))
    return df


def load_or_fetch(
    start_date: str,
    end_date: str,
    price_area: str = DEFAULT_PRICE_AREA,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load price data from a local CSV cache or fetch from the API.

    The function writes the fetched data to *cache_path* so that subsequent
    calls can reuse it without hitting the network.

    Parameters
    ----------
    start_date:
        Inclusive start date in ``YYYY-MM-DD`` format.
    end_date:
        Exclusive end date in ``YYYY-MM-DD`` format.
    price_area:
        Price area (default ``"DK1"``).
    cache_path:
        Path to the local CSV file.  Defaults to
        ``data/spot_prices_<price_area>.csv``.
    force_refresh:
        When ``True`` the cache is ignored and fresh data is fetched.

    Returns
    -------
    pd.DataFrame
        Hourly price DataFrame as returned by :func:`fetch_spot_prices`.
    """
    if cache_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = DATA_DIR / f"spot_prices_{price_area}.csv"

    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached data from %s", cache_path)
        df = pd.read_csv(cache_path, index_col="HourDK", parse_dates=True)
        return df

    df = fetch_spot_prices(start_date, end_date, price_area)
    df.to_csv(cache_path)
    logger.info("Saved data to %s", cache_path)
    return df


def load_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load price data from a pre-downloaded CSV file.

    The CSV must contain at minimum the columns ``HourDK`` and
    ``SpotPriceDKK``.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Hourly price DataFrame indexed by ``HourDK``.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, parse_dates=["HourDK"])
    df = df.sort_values("HourDK").reset_index(drop=True)
    df = df.set_index("HourDK")
    df["SpotPriceDKK"] = pd.to_numeric(df["SpotPriceDKK"], errors="coerce")
    logger.info("Loaded %d records from %s", len(df), path)
    return df


def get_default_date_range(years: int = 3) -> tuple[str, str]:
    """Return a sensible (start, end) date range for training data.

    Parameters
    ----------
    years:
        Number of years of historical data to request.

    Returns
    -------
    tuple[str, str]
        ``(start_date, end_date)`` strings in ``YYYY-MM-DD`` format.
    """
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=365 * years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
