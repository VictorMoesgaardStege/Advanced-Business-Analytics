#!/usr/bin/env python3
"""Fetch private heating consumption from Energi Data Service.

Two datasets are supported:

  PrivateConsumptionHeatingNationalHour
    National-level hourly consumption broken down by HousingCategory and
    HeatingCategory.

  PrivateConsumptionHeatingHour
    Same breakdown but at municipality level (adds MunicipalityCode,
    Municipality, RegionName). Large dataset — use date filters.

Examples:

    python src/data/fetch_private_consumption_heating_data.py national \
  --start 2025-01-01 \
  --end 2026-01-01 \
  --csv data/private_consumption_heating_national_raw.csv

    python src/data/fetch_private_consumption_heating_data.py municipal \
  --start 2025-01-01 \
  --end 2026-01-01 \
  --csv data/private_consumption_heating_municipal_raw.csv

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

NATIONAL_URL  = "https://api.energidataservice.dk/dataset/PrivateConsumptionHeatingNationalHour"
MUNICIPAL_URL = "https://api.energidataservice.dk/dataset/PrivateConsumptionHeatingHour"

NATIONAL_COLUMNS = [
    "TimeUTC",
    "TimeDK",
    "HousingCategory",
    "HeatingCategory",
    "ConsumptionkWh",
]

MUNICIPAL_COLUMNS = [
    "TimeUTC",
    "TimeDK",
    "MunicipalityCode",
    "Municipality",
    "RegionName",
    "HousingCategory",
    "HeatingCategory",
    "ConsumptionkWh",
]


def build_params(
    start: str,
    end: str,
    *,
    limit: int,
    offset: int,
    sort: str,
    columns: list[str],
    extra_filters: dict[str, list[str]] | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "start":   start,
        "end":     end,
        "limit":   limit,
        "offset":  offset,
        "sort":    sort,
        "columns": ",".join(columns),
    }
    if extra_filters:
        params["filter"] = json.dumps(extra_filters, ensure_ascii=False)
    return params


def fetch_records(
    dataset: str,
    start: str,
    end: str,
    *,
    page_size: int = 5000,
    sort: str = "TimeUTC desc",
    columns: list[str] | None = None,
    extra_filters: dict[str, list[str]] | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Fetch all matching rows, paging with offset/limit."""
    if dataset == "national":
        base_url = NATIONAL_URL
        if columns is None:
            columns = NATIONAL_COLUMNS
    elif dataset == "municipal":
        base_url = MUNICIPAL_URL
        if columns is None:
            columns = MUNICIPAL_COLUMNS
    else:
        raise ValueError(f"dataset must be 'national' or 'municipal', got {dataset!r}")

    all_records: list[dict[str, Any]] = []
    offset = 0

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "fetch_private_consumption_heating_data.py/1.0",
    })

    while True:
        params = build_params(
            start, end,
            limit=page_size,
            offset=offset,
            sort=sort,
            columns=columns,
            extra_filters=extra_filters,
        )

        response = session.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        records = payload.get("records", [])
        if not isinstance(records, list):
            raise RuntimeError("Unexpected API response: 'records' is not a list")

        all_records.extend(records)

        total = payload.get("total")
        if not records:
            break

        offset += len(records)

        if isinstance(total, int) and offset >= total:
            break

        if len(records) < page_size:
            break

    return all_records


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else NATIONAL_COLUMNS
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _safe_float(value: Any) -> float:
    if value in (None, "", "null"):
        return 0.0
    return float(value)


def print_summary(records: list[dict[str, Any]], dataset: str) -> None:
    print(f"Fetched {len(records)} record(s).")
    if not records:
        return

    first = records[0]
    last  = records[-1]
    print(
        "Time span in returned rows: "
        f"{last.get('TimeUTC', '?')} -> {first.get('TimeUTC', '?')} "
        "(UTC, depending on sort)"
    )

    housing_cats  = sorted({r.get("HousingCategory", "")  for r in records if r.get("HousingCategory")})
    heating_cats  = sorted({r.get("HeatingCategory", "")  for r in records if r.get("HeatingCategory")})

    print(f"Housing categories in result: {', '.join(housing_cats) if housing_cats else 'n/a'}")
    print(f"Heating categories in result: {', '.join(heating_cats) if heating_cats else 'n/a'}")

    if dataset == "municipal":
        municipalities = sorted({r.get("Municipality", "") for r in records if r.get("Municipality")})
        print(f"Municipalities in result: {len(municipalities)}")

    total_kwh = sum(_safe_float(r.get("ConsumptionkWh")) for r in records)
    print(f"Total consumption in result set: {total_kwh:,.1f} kWh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch private heating consumption from Energi Data Service."
    )
    parser.add_argument(
        "dataset",
        choices=["national", "municipal"],
        help="'national' = PrivateConsumptionHeatingNationalHour, "
             "'municipal' = PrivateConsumptionHeatingHour",
    )
    parser.add_argument("--start", required=True, help="Start date, e.g. 2025-01-01")
    parser.add_argument("--end",   required=True, help="End date (exclusive), e.g. 2026-01-01")
    parser.add_argument("--page-size", type=int, default=5000, help="Rows per request (default: 5000)")
    parser.add_argument("--sort", default="TimeUTC desc", help="API sort expression")
    parser.add_argument("--csv",  type=Path, help="Write results to CSV")
    parser.add_argument("--json", type=Path, help="Write results to JSON")
    parser.add_argument("--print-records", action="store_true", help="Print all records as JSON to stdout")
    parser.add_argument(
        "--heating-category",
        action="append",
        dest="heating_category",
        help="Filter by HeatingCategory. Repeat to include multiple.",
    )
    parser.add_argument(
        "--housing-category",
        action="append",
        dest="housing_category",
        help="Filter by HousingCategory. Repeat to include multiple.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    extra_filters: dict[str, list[str]] = {}
    if args.heating_category:
        extra_filters["HeatingCategory"] = args.heating_category
    if args.housing_category:
        extra_filters["HousingCategory"] = args.housing_category

    try:
        records = fetch_records(
            dataset=args.dataset,
            start=args.start,
            end=args.end,
            page_size=args.page_size,
            sort=args.sort,
            extra_filters=extra_filters or None,
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

    print_summary(records, args.dataset)

    if args.csv:
        write_csv(records, args.csv)
        print(f"Saved CSV to {args.csv}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON to {args.json}")

    if args.print_records:
        print(json.dumps(records, indent=2, ensure_ascii=False))

    if not args.csv and not args.json and not args.print_records:
        preview = records[:5]
        print("\nFirst rows:")
        print(json.dumps(preview, indent=2, ensure_ascii=False))

    example_params = build_params(
        args.start, args.end,
        limit=min(args.page_size, 100),
        offset=0,
        sort=args.sort,
        columns=NATIONAL_COLUMNS if args.dataset == "national" else MUNICIPAL_COLUMNS,
        extra_filters=extra_filters or None,
    )
    base_url = NATIONAL_URL if args.dataset == "national" else MUNICIPAL_URL
    print("\nExample request URL:")
    print(f"{base_url}?{urlencode(example_params)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
