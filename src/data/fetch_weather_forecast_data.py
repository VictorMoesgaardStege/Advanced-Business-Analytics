#!/usr/bin/env python3
"""Fetch forecast weather data from Open-Meteo Previous Runs API.

Downloads:
- Day 0
- previous_day1
- previous_day2
- previous_day3
- previous_day4
- previous_day5

Example:
    python src/data/fetch_weather_forecast_data.py \
        --start 2025-11-01 \
        --end 2026-03-16 \
        --csv data/weather_forecasts_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DEFAULT_CSV = Path("data/weather_forecasts_raw.csv")
DEFAULT_MODEL = "ecmwf_ifs"

DEFAULT_LOCATIONS = [
    {"region": "DK1_west", "latitude": 56.15, "longitude": 8.45},
    {"region": "DK2_east", "latitude": 55.68, "longitude": 12.57},
    {"region": "SE_south", "latitude": 55.60, "longitude": 13.00},
    {"region": "NO_south", "latitude": 58.15, "longitude": 8.00},
    {"region": "DE_north", "latitude": 54.30, "longitude": 9.70},
]

BASE_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_120m",
    "wind_direction_120m",
    "shortwave_radiation",
    "cloud_cover",
    "temperature_2m",
    "pressure_msl",
]


def build_hourly_vars(base_vars: list[str], max_previous_day: int = 5) -> list[str]:
    hourly_vars: list[str] = []
    for var in base_vars:
        hourly_vars.append(var)
        for d in range(1, max_previous_day + 1):
            hourly_vars.append(f"{var}_previous_day{d}")
    return hourly_vars


DEFAULT_HOURLY_VARS = build_hourly_vars(BASE_VARS, max_previous_day=5)


def month_ranges(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        if cur.month == 12:
            next_month = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            next_month = cur.replace(month=cur.month + 1, day=1)

        chunk_start = max(cur, start)
        chunk_end = min(next_month - timedelta(days=1), end)
        yield chunk_start, chunk_end
        cur = next_month


def fetch_json(session: requests.Session, params: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(8):
        try:
            response = session.get(BASE_URL, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            sleep_seconds = min(60, 3 * (attempt + 1))
            print(
                f"Request failed (attempt {attempt + 1}/8): {exc}. "
                f"Retrying in {sleep_seconds:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed after retries: {last_exc}")


def response_to_records(payload: dict[str, Any], region_name: str) -> list[dict[str, Any]]:
    hourly = payload.get("hourly", {})
    times = hourly.get("time")
    if not times:
        return []

    df = pd.DataFrame(hourly)
    df.insert(0, "region", region_name)
    df = df.rename(columns={"time": "TimeDK"})

    df["Latitude"] = payload.get("latitude")
    df["Longitude"] = payload.get("longitude")
    df["Elevation"] = payload.get("elevation")
    df["Timezone"] = payload.get("timezone")
    df["TimezoneAbbreviation"] = payload.get("timezone_abbreviation")

    first_cols = [
        "TimeDK",
        "region",
        "Latitude",
        "Longitude",
        "Elevation",
        "Timezone",
        "TimezoneAbbreviation",
    ]
    remaining_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + remaining_cols]

    return df.to_dict(orient="records")


def fetch_records(
    start: str,
    end: str,
    *,
    model: str = DEFAULT_MODEL,
    hourly_vars: list[str] | None = None,
    locations: list[dict[str, Any]] | None = None,
    timezone: str = "Europe/Copenhagen",
    wind_speed_unit: str = "ms",
    temperature_unit: str = "celsius",
    precipitation_unit: str = "mm",
) -> list[dict[str, Any]]:
    if hourly_vars is None:
        hourly_vars = DEFAULT_HOURLY_VARS
    if locations is None:
        locations = DEFAULT_LOCATIONS

    start_date = datetime.fromisoformat(start).date()
    end_date = datetime.fromisoformat(end).date()

    all_records: list[dict[str, Any]] = []

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "fetch_weather_forecast_data.py/1.0",
        }
    )

    for location in locations:
        region_name = str(location["region"])

        for chunk_start, chunk_end in month_ranges(start_date, end_date):
            params = {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(hourly_vars),
                "models": model,
                "timezone": timezone,
                "wind_speed_unit": wind_speed_unit,
                "temperature_unit": temperature_unit,
                "precipitation_unit": precipitation_unit,
            }

            payload = fetch_json(session, params)
            records = response_to_records(payload, region_name)
            all_records.extend(records)

            print(
                f"Fetched {len(records)} row(s) for {region_name} "
                f"from {chunk_start} to {chunk_end}"
            )
            time.sleep(1.0)

    return all_records


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["TimeDK", "region"])
        return

    fieldnames = list(records[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records: list[dict[str, Any]]) -> None:
    print(f"Fetched {len(records)} record(s).")
    if not records:
        return

    df = pd.DataFrame(records)
    df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")

    print(f"Time span: {df['TimeDK'].min()} -> {df['TimeDK'].max()}")
    print("Regions:", ", ".join(sorted(df["region"].dropna().unique())))

    cols = [
        "wind_speed_10m",
        "wind_speed_10m_previous_day1",
        "wind_speed_10m_previous_day2",
        "wind_speed_10m_previous_day3",
        "wind_speed_10m_previous_day4",
        "wind_speed_10m_previous_day5",
        "wind_speed_120m",
        "wind_speed_120m_previous_day1",
        "wind_speed_120m_previous_day2",
        "wind_speed_120m_previous_day3",
        "wind_speed_120m_previous_day4",
        "wind_speed_120m_previous_day5",
        "shortwave_radiation",
        "shortwave_radiation_previous_day1",
        "shortwave_radiation_previous_day2",
        "shortwave_radiation_previous_day3",
        "shortwave_radiation_previous_day4",
        "shortwave_radiation_previous_day5",
        "temperature_2m",
        "temperature_2m_previous_day1",
        "temperature_2m_previous_day2",
        "temperature_2m_previous_day3",
        "temperature_2m_previous_day4",
        "temperature_2m_previous_day5",
    ]

    print("\nNon-missing values by selected columns:")
    for col in cols:
        if col in df.columns:
            print(f"  {col}: {df[col].notna().sum():,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch forecast weather data from Previous Runs API.")
    parser.add_argument("--start", required=True, help="Start date, e.g. 2025-11-01")
    parser.add_argument("--end", required=True, help="End date, e.g. 2026-03-16")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model slug (default: {DEFAULT_MODEL})")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"CSV output path (default: {DEFAULT_CSV})")
    parser.add_argument("--json", type=Path, help="Optional JSON output path")
    parser.add_argument("--print-records", action="store_true", help="Print first records as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        records = fetch_records(
            start=args.start,
            end=args.end,
            model=args.model,
        )
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text, file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    print_summary(records)

    if args.csv:
        write_csv(records, args.csv)
        print(f"Saved CSV to {args.csv}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON to {args.json}")

    if args.print_records:
        print(json.dumps(records[:10], indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())