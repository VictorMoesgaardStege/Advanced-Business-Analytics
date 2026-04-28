#!/usr/bin/env python3
"""Fetch hourly electricity consumption by consumer category from Energi Data Service.

Dataset: ConsumptionConsumerCategoryHour

Columns: TimeUTC, TimeDK, RegionName, ConsumerCategory2, ConsumerCategory3,
         ConsumptionkWh

Examples:

    python src/data/fetch_consumption_consumer_category_data.py \
  --start 2025-01-01 \
  --end 2026-01-01 \
  --csv data/consumption_consumer_category_raw.csv

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

BASE_URL = "https://api.energidataservice.dk/dataset/ConsumptionConsumerCategoryHour"

DEFAULT_COLUMNS = [
    "TimeUTC",
    "TimeDK",
    "RegionName",
    "ConsumerCategory2",
    "ConsumerCategory3",
    "ConsumptionkWh",
]


def build_filter(
    region: list[str] | None,
    consumer_category: list[str] | None,
) -> dict[str, list[str]]:
    flt: dict[str, list[str]] = {}
    if region:
        flt["RegionName"] = region
    if consumer_category:
        flt["ConsumerCategory3"] = consumer_category
    return flt


def build_params(
    start: str,
    end: str,
    *,
    region: list[str] | None,
    consumer_category: list[str] | None,
    limit: int,
    offset: int,
    sort: str,
    columns: list[str],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "start":   start,
        "end":     end,
        "limit":   limit,
        "offset":  offset,
        "sort":    sort,
        "columns": ",".join(columns),
    }
    flt = build_filter(region, consumer_category)
    if flt:
        params["filter"] = json.dumps(flt, ensure_ascii=False)
    return params


def fetch_records(
    start: str,
    end: str,
    *,
    region: list[str] | None = None,
    consumer_category: list[str] | None = None,
    page_size: int = 5000,
    sort: str = "TimeUTC desc,RegionName,ConsumerCategory3",
    columns: list[str] | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Fetch all matching rows, paging with offset/limit."""
    if columns is None:
        columns = DEFAULT_COLUMNS

    all_records: list[dict[str, Any]] = []
    offset = 0

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "fetch_consumption_consumer_category_data.py/1.0",
    })

    while True:
        params = build_params(
            start, end,
            region=region,
            consumer_category=consumer_category,
            limit=page_size,
            offset=offset,
            sort=sort,
            columns=columns,
        )

        response = session.get(BASE_URL, params=params, timeout=timeout)
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
    fieldnames = list(records[0].keys()) if records else DEFAULT_COLUMNS
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _safe_float(value: Any) -> float:
    if value in (None, "", "null"):
        return 0.0
    return float(value)


def print_summary(records: list[dict[str, Any]]) -> None:
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

    regions = sorted({r.get("RegionName", "") for r in records if r.get("RegionName")})
    cat2    = sorted({r.get("ConsumerCategory2", "") for r in records if r.get("ConsumerCategory2")})
    cat3    = sorted({r.get("ConsumerCategory3", "") for r in records if r.get("ConsumerCategory3")})

    print(f"Regions in result:             {', '.join(regions) if regions else 'n/a'}")
    print(f"ConsumerCategory2 in result:   {', '.join(cat2) if cat2 else 'n/a'}")
    print(f"ConsumerCategory3 in result:   {', '.join(cat3) if cat3 else 'n/a'}")

    total_kwh = sum(_safe_float(r.get("ConsumptionkWh")) for r in records)
    print(f"Total consumption in result set: {total_kwh:,.1f} kWh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch hourly consumption by consumer category from Energi Data Service."
    )
    parser.add_argument("--start", required=True, help="Start date, e.g. 2025-01-01")
    parser.add_argument("--end",   required=True, help="End date (exclusive), e.g. 2026-01-01")
    parser.add_argument("--page-size", type=int, default=5000, help="Rows per request (default: 5000)")
    parser.add_argument("--sort", default="TimeUTC desc,RegionName,ConsumerCategory3", help="API sort expression")
    parser.add_argument("--csv",  type=Path, help="Write results to CSV")
    parser.add_argument("--json", type=Path, help="Write results to JSON")
    parser.add_argument("--print-records", action="store_true", help="Print all records as JSON to stdout")
    parser.add_argument(
        "--region", action="append",
        help="Filter by RegionName. Repeat to include multiple.",
    )
    parser.add_argument(
        "--consumer-category", action="append", dest="consumer_category",
        help="Filter by ConsumerCategory3. Repeat to include multiple.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        records = fetch_records(
            start=args.start,
            end=args.end,
            page_size=args.page_size,
            sort=args.sort,
            region=args.region,
            consumer_category=args.consumer_category,
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
        print(json.dumps(records, indent=2, ensure_ascii=False))

    if not args.csv and not args.json and not args.print_records:
        preview = records[:5]
        print("\nFirst rows:")
        print(json.dumps(preview, indent=2, ensure_ascii=False))

    example_params = build_params(
        args.start, args.end,
        region=args.region,
        consumer_category=args.consumer_category,
        limit=min(args.page_size, 100),
        offset=0,
        sort=args.sort,
        columns=DEFAULT_COLUMNS,
    )
    print("\nExample request URL:")
    print(f"{BASE_URL}?{urlencode(example_params)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
