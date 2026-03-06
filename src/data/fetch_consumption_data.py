#!/usr/bin/env python3
"""Fetch hourly electricity consumption from Energi Data Service.

Dataset: ConsumptionGridAreaHour
Docs:
- Data API: https://api.energidataservice.dk/dataset/ConsumptionGridAreaHour
- Metadata: https://api.energidataservice.dk/meta/dataset/ConsumptionGridAreaHour

Examples:
  python fetch_data.py --start 2026-02-01 --end 2026-02-02
  python fetch_data.py --start 2026-02-01 --end 2026-02-02 --price-area DK1
  python fetch_data.py --start 2026-02-01 --end 2026-02-02 --grid-area 791 --csv out.csv
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

BASE_URL = "https://api.energidataservice.dk/dataset/ConsumptionGridAreaHour"
DEFAULT_COLUMNS = [
    "Date",
    "TimeUTC",
    "TimeDK",
    "PriceArea",
    "GridArea",
    "GridCompanyName",
    "ConsumptionkWh",
]


def build_filter(price_area: list[str] | None, grid_area: list[str] | None) -> dict[str, list[str]]:
    flt: dict[str, list[str]] = {}
    if price_area:
        flt["PriceArea"] = price_area
    if grid_area:
        flt["GridArea"] = grid_area
    return flt


def build_params(
    start: str,
    end: str,
    *,
    price_area: list[str] | None,
    grid_area: list[str] | None,
    limit: int,
    offset: int,
    sort: str,
    columns: list[str],
    timezone: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "start": start,
        "end": end,
        "limit": limit,
        "offset": offset,
        "sort": sort,
        "columns": ",".join(columns),
    }
    flt = build_filter(price_area, grid_area)
    if flt:
        params["filter"] = json.dumps(flt, ensure_ascii=False)
    if timezone:
        params["timezone"] = timezone
    return params


def fetch_records(
    start: str,
    end: str,
    *,
    price_area: list[str] | None = None,
    grid_area: list[str] | None = None,
    page_size: int = 5000,
    sort: str = "TimeUTC desc,PriceArea,GridArea",
    columns: list[str] | None = None,
    timezone: str | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Fetch all matching rows, paging with offset/limit."""
    if columns is None:
        columns = DEFAULT_COLUMNS

    all_records: list[dict[str, Any]] = []
    offset = 0
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "fetch_data.py/1.0"})

    while True:
        params = build_params(
            start,
            end,
            price_area=price_area,
            grid_area=grid_area,
            limit=page_size,
            offset=offset,
            sort=sort,
            columns=columns,
            timezone=timezone,
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


def print_summary(records: list[dict[str, Any]]) -> None:
    print(f"Fetched {len(records)} record(s).")
    if not records:
        return

    total_kwh = sum(float(r.get("ConsumptionkWh", 0) or 0) for r in records)
    print(f"Total consumption in result set: {total_kwh:,.1f} kWh")

    first = records[0]
    last = records[-1]
    print(
        "Time span in returned rows: "
        f"{last.get('TimeUTC', '?')} -> {first.get('TimeUTC', '?')} (UTC, depending on sort)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch hourly electricity consumption from Energi Data Service (ConsumptionGridAreaHour)."
    )
    parser.add_argument("--start", required=True, help="Start in Danish local time, e.g. 2026-02-01 or 2026-02-01T00:00")
    parser.add_argument("--end", required=True, help="End in Danish local time, exclusive")
    parser.add_argument("--price-area", action="append", choices=["DK1", "DK2"], help="Filter by price area. Repeat for both.")
    parser.add_argument("--grid-area", action="append", help="Filter by grid area, e.g. 791. Repeat to include multiple.")
    parser.add_argument("--page-size", type=int, default=5000, help="Rows per request (default: 5000)")
    parser.add_argument("--sort", default="TimeUTC desc,PriceArea,GridArea", help="API sort expression")
    parser.add_argument("--timezone", help="Optional API timezone parameter, e.g. UTC")
    parser.add_argument("--csv", type=Path, help="Write results to CSV")
    parser.add_argument("--json", type=Path, help="Write results to JSON")
    parser.add_argument("--print-records", action="store_true", help="Print all records as JSON to stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        records = fetch_records(
            start=args.start,
            end=args.end,
            price_area=args.price_area,
            grid_area=args.grid_area,
            page_size=args.page_size,
            sort=args.sort,
            timezone=args.timezone,
        )
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text, file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
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

    # Print a reproducible URL for debugging.
    example_params = build_params(
        args.start,
        args.end,
        price_area=args.price_area,
        grid_area=args.grid_area,
        limit=min(args.page_size, 100),
        offset=0,
        sort=args.sort,
        columns=DEFAULT_COLUMNS,
        timezone=args.timezone,
    )
    print("\nExample request URL:")
    print(f"{BASE_URL}?{urlencode(example_params)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
