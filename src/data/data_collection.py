"""
data_collection.py  —  API fetching for all raw data sources
=============================================================
Callable functions for fetching raw data from Energi Data Service
and Open-Meteo. Each function saves to a default CSV path and returns
a DataFrame.

Quick start
-----------
    from data_collection import (
        fetch_weather_actuals,
        fetch_weather_forecasts,
        fetch_consumption,
        fetch_day_ahead_prices,
        fetch_all,
    )

    fetch_all("2021-01-01", "2026-04-28")
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    start    : str  e.g. "2022-01-01"
    end      : str  e.g. "2026-04-28"
    csv_path : output path (default: data/weather_actuals_raw.csv)

    Returns
    -------
    pd.DataFrame with one row per (region, hour).
    """
    from src.data.fetch_weather_actuals_data import fetch_records, write_csv

    print(f"[fetch_weather_actuals] {start} -> {end}")
    records = fetch_records(start=start, end=end)
    csv_path = Path(csv_path)
    write_csv(records, csv_path)
    print(f"  {len(records):,} rows saved -> {csv_path}")

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
    start    : str  e.g. "2025-11-01"
    end      : str  e.g. "2026-04-28"
    csv_path : output path (default: data/weather_forecasts_raw.csv)

    Returns
    -------
    pd.DataFrame with one row per (region, hour, previous_day_offset).
    """
    from src.data.fetch_weather_forecast_data import fetch_records, write_csv

    print(f"[fetch_weather_forecasts] {start} -> {end}")
    records = fetch_records(start=start, end=end)
    csv_path = Path(csv_path)
    write_csv(records, csv_path)
    print(f"  {len(records):,} rows saved -> {csv_path}")

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
    pd.DataFrame with columns Date, TimeUTC, TimeDK, PriceArea, GridArea,
                 GridCompanyName, ConsumptionkWh.
    """
    from src.data.fetch_consumption_data import fetch_records, write_csv

    if csv_path is None:
        csv_path = DATA_DIR / f"consumption_{price_area.lower()}_raw.csv"
    csv_path = Path(csv_path)

    print(f"[fetch_consumption] {start} -> {end}  area={price_area}")
    records = fetch_records(start=start, end=end, price_area=[price_area])
    write_csv(records, csv_path)
    print(f"  {len(records):,} rows saved -> {csv_path}")

    df = pd.DataFrame(records)
    if "TimeDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
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
    pd.DataFrame with columns TimeUTC, TimeDK, PriceArea,
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

    print(f"[fetch_day_ahead_prices] {start} -> {end}  area={price_area}")

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
    print(f"  {len(merged):,} merged rows saved -> {csv_path}")

    df = pd.DataFrame(merged)
    for col in ("TimeUTC", "TimeDK"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ── Private heating consumption ────────────────────────────────────────────────

def fetch_private_heating_national(
    start: str,
    end: str,
    csv_path: str | Path = DATA_DIR / "private_consumption_heating_national_raw.csv",
    heating_category: list[str] | None = None,
    housing_category: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch national-level private heating consumption (PrivateConsumptionHeatingNationalHour).

    Parameters
    ----------
    start            : str  e.g. "2021-01-01"
    end              : str  e.g. "2026-04-28"
    csv_path         : output path
    heating_category : optional filter, e.g. ["Elvarme eller varmepumpe"]
    housing_category : optional filter, e.g. ["Etageejendom"]

    Returns
    -------
    pd.DataFrame with columns TimeUTC, TimeDK, HousingCategory,
                 HeatingCategory, ConsumptionkWh.
    """
    from src.data.fetch_private_consumption_heating_data import fetch_records, write_csv

    csv_path = Path(csv_path)
    extra_filters: dict = {}
    if heating_category:
        extra_filters["HeatingCategory"] = heating_category
    if housing_category:
        extra_filters["HousingCategory"] = housing_category

    print(f"[fetch_private_heating_national] {start} -> {end}")
    records = fetch_records(
        dataset="national", start=start, end=end,
        extra_filters=extra_filters or None,
    )
    write_csv(records, csv_path)
    print(f"  {len(records):,} rows saved -> {csv_path}")

    df = pd.DataFrame(records)
    for col in ("TimeUTC", "TimeDK"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fetch_private_heating_municipal(
    start: str,
    end: str,
    csv_path: str | Path = DATA_DIR / "private_consumption_heating_municipal_raw.csv",
    heating_category: list[str] | None = None,
    housing_category: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch municipality-level private heating consumption (PrivateConsumptionHeatingHour).

    This dataset is large (~29M rows total). Use date filters to keep
    downloads manageable — a single year is ~3-4M rows.

    Parameters
    ----------
    start            : str  e.g. "2024-01-01"
    end              : str  e.g. "2026-04-28"
    csv_path         : output path
    heating_category : optional filter, e.g. ["Elvarme eller varmepumpe"]
    housing_category : optional filter, e.g. ["Etageejendom"]

    Returns
    -------
    pd.DataFrame with columns TimeUTC, TimeDK, MunicipalityCode, Municipality,
                 RegionName, HousingCategory, HeatingCategory, ConsumptionkWh.
    """
    from src.data.fetch_private_consumption_heating_data import fetch_records, write_csv

    csv_path = Path(csv_path)
    extra_filters: dict = {}
    if heating_category:
        extra_filters["HeatingCategory"] = heating_category
    if housing_category:
        extra_filters["HousingCategory"] = housing_category

    print(f"[fetch_private_heating_municipal] {start} -> {end}")
    records = fetch_records(
        dataset="municipal", start=start, end=end,
        extra_filters=extra_filters or None,
    )
    write_csv(records, csv_path)
    print(f"  {len(records):,} rows saved -> {csv_path}")

    df = pd.DataFrame(records)
    for col in ("TimeUTC", "TimeDK"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ── Convenience: fetch everything ─────────────────────────────────────────────

def fetch_all(start: str, end: str, price_area: str = "DK1") -> dict[str, pd.DataFrame]:
    """Fetch all raw data sources for the given date range.

    Runs the four fetch functions in order:
      1. weather actuals       -> data/weather_actuals_raw.csv
      2. weather forecasts     -> data/weather_forecasts_raw.csv
      3. consumption           -> data/consumption_<area>_raw.csv
      4. day-ahead prices      -> data/day_ahead_prices_<area>_raw.csv

    Parameters
    ----------
    start      : str  e.g. "2021-01-01"
    end        : str  e.g. "2026-04-28"
    price_area : "DK1" or "DK2"

    Returns
    -------
    dict with keys: "weather_actuals", "weather_forecasts", "consumption", "prices"
    """
    print("=" * 60)
    print(f"fetch_all  {start} -> {end}  (area={price_area})")
    print("=" * 60)

    results = {}
    results["weather_actuals"]   = fetch_weather_actuals(start, end)
    results["weather_forecasts"] = fetch_weather_forecasts(start, end)
    results["consumption"]       = fetch_consumption(start, end, price_area=price_area)
    results["prices"]            = fetch_day_ahead_prices(start, end, price_area=price_area)

    print("\n" + "=" * 60)
    print("fetch_all complete. Run data_processing.py next to build forecast_dataset.parquet.")
    print("=" * 60)
    return results


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch all raw data from APIs."
    )
    parser.add_argument("--start", required=True, help="Start date, e.g. 2021-01-01")
    parser.add_argument("--end",   required=True, help="End date, e.g. 2026-04-28")
    parser.add_argument("--area",  default="DK1", help="Price area (default: DK1)")
    args = parser.parse_args()

    fetch_all(args.start, args.end, price_area=args.area)
