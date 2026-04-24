#!/usr/bin/env python3
"""Fetch and merge historical electricity prices from Energi Data Service.

Datasets:
- Elspotprices
- DayAheadPrices

Example:
python src/data/fetch_day_ahead_price_data.py --start 2021-01-01 --end 2026-02-02 --price-area DK1 --csv data/day_ahead_prices_dk1_raw.csv"""



from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import requests

DAYAHEAD_URL = "https://api.energidataservice.dk/dataset/DayAheadPrices"
ELSPOT_URL = "https://api.energidataservice.dk/dataset/Elspotprices"

DAYAHEAD_COLUMNS = [
    "TimeUTC",
    "TimeDK",
    "PriceArea",
    "DayAheadPriceEUR",
    "DayAheadPriceDKK",
]

ELSPOT_COLUMNS = [
    "HourUTC",
    "HourDK",
    "PriceArea",
    "SpotPriceEUR",
    "SpotPriceDKK",
]

OUTPUT_COLUMNS = [
    "TimeUTC",
    "TimeDK",
    "PriceArea",
    "DayAheadPriceEUR",
    "DayAheadPriceDKK",
    "_source_dataset",
]


def build_filter(price_area: list[str] | None) -> dict[str, list[str]]:
    flt: dict[str, list[str]] = {}
    if price_area:
        flt["PriceArea"] = price_area
    return flt


def build_params(
    start: str,
    end: str,
    *,
    price_area: list[str] | None,
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

    flt = build_filter(price_area)
    if flt:
        params["filter"] = json.dumps(flt, ensure_ascii=False)

    if timezone:
        params["timezone"] = timezone

    return params


def fetch_records_from_url(
    base_url: str,
    start: str,
    end: str,
    *,
    price_area: list[str] | None = None,
    page_size: int = 5000,
    sort: str = "",
    columns: list[str] | None = None,
    timezone: str | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    if columns is None:
        raise ValueError("columns must be provided")

    all_records: list[dict[str, Any]] = []
    offset = 0

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "fetch_day_ahead_price_data.py/1.0",
        }
    )

    while True:
        params = build_params(
            start,
            end,
            price_area=price_area,
            limit=page_size,
            offset=offset,
            sort=sort,
            columns=columns,
            timezone=timezone,
        )

        response = session.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        records = payload.get("records", [])
        if not isinstance(records, list):
            raise RuntimeError(f"Unexpected API response from {base_url}: 'records' is not a list")

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


def normalize_dayahead_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "TimeUTC": record.get("TimeUTC"),
        "TimeDK": record.get("TimeDK"),
        "PriceArea": record.get("PriceArea"),
        "DayAheadPriceEUR": record.get("DayAheadPriceEUR"),
        "DayAheadPriceDKK": record.get("DayAheadPriceDKK"),
        "_source_dataset": "DayAheadPrices",
    }


def normalize_elspot_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "TimeUTC": record.get("HourUTC"),
        "TimeDK": record.get("HourDK"),
        "PriceArea": record.get("PriceArea"),
        "DayAheadPriceEUR": record.get("SpotPriceEUR"),
        "DayAheadPriceDKK": record.get("SpotPriceDKK"),
        "_source_dataset": "Elspotprices",
    }


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_priority = {
        "Elspotprices": 0,
        "DayAheadPrices": 1,
    }

    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}

    for r in records:
        key = (
            str(r.get("TimeUTC") or ""),
            str(r.get("TimeDK") or ""),
            str(r.get("PriceArea") or ""),
        )

        existing = deduped.get(key)
        if existing is None:
            deduped[key] = r
            continue

        old_prio = source_priority.get(str(existing.get("_source_dataset")), -1)
        new_prio = source_priority.get(str(r.get("_source_dataset")), -1)

        if new_prio >= old_prio:
            deduped[key] = r

    merged = list(deduped.values())
    merged.sort(key=lambda r: (str(r.get("TimeUTC") or ""), str(r.get("PriceArea") or "")))
    return merged


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def _safe_float(value: Any) -> float:
    if value in (None, "", "null"):
        return 0.0
    if isinstance(value, str):
        value = value.replace(",", ".")
    return float(value)


def print_summary(records: list[dict[str, Any]]) -> None:
    print(f"Merged {len(records)} record(s).")
    if not records:
        return

    first = records[0]
    last = records[-1]

    print(f"Time span: {first.get('TimeUTC', '?')} -> {last.get('TimeUTC', '?')}")
    print(f"Price areas: {', '.join(sorted({r['PriceArea'] for r in records if r.get('PriceArea')}))}")

    counts: dict[str, int] = {}
    for r in records:
        src = str(r.get("_source_dataset", "unknown"))
        counts[src] = counts.get(src, 0) + 1

    print("Rows kept by source:")
    for src, count in sorted(counts.items()):
        print(f"  {src}: {count}")

    eur_values = [_safe_float(r["DayAheadPriceEUR"]) for r in records if r.get("DayAheadPriceEUR") is not None]
    dkk_values = [_safe_float(r["DayAheadPriceDKK"]) for r in records if r.get("DayAheadPriceDKK") is not None]

    if eur_values:
        print(f"EUR min={min(eur_values):.2f}, max={max(eur_values):.2f}, mean={sum(eur_values)/len(eur_values):.2f}")
    if dkk_values:
        print(f"DKK min={min(dkk_values):.2f}, max={max(dkk_values):.2f}, mean={sum(dkk_values)/len(dkk_values):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and merge Elspotprices + DayAheadPrices.")
    parser.add_argument("--start", required=True, help="Start datetime, e.g. 2021-01-01")
    parser.add_argument("--end", required=True, help="End datetime, exclusive")
    parser.add_argument(
        "--price-area",
        action="append",
        choices=["DK1", "DK2", "DE", "NO2", "SE3", "SE4"],
        help="Filter by price area. Repeat to include multiple.",
    )
    parser.add_argument("--page-size", type=int, default=5000, help="Rows per request")
    parser.add_argument("--timezone", help="Optional API timezone parameter")
    parser.add_argument("--csv", type=Path, help="Write merged CSV")
    parser.add_argument("--json", type=Path, help="Write merged JSON")
    parser.add_argument("--print-records", action="store_true", help="Print all merged records")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        elspot_records_raw = fetch_records_from_url(
            ELSPOT_URL,
            start=args.start,
            end=args.end,
            price_area=args.price_area,
            page_size=args.page_size,
            sort="HourUTC desc,PriceArea",
            timezone=args.timezone,
            columns=ELSPOT_COLUMNS,
        )

        dayahead_records_raw = fetch_records_from_url(
            DAYAHEAD_URL,
            start=args.start,
            end=args.end,
            price_area=args.price_area,
            page_size=args.page_size,
            sort="TimeUTC desc,PriceArea",
            timezone=args.timezone,
            columns=DAYAHEAD_COLUMNS,
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

    print(f"Fetched {len(elspot_records_raw)} row(s) from Elspotprices")
    print(f"Fetched {len(dayahead_records_raw)} row(s) from DayAheadPrices")

    merged = (
        [normalize_elspot_record(r) for r in elspot_records_raw]
        + [normalize_dayahead_record(r) for r in dayahead_records_raw]
    )

    merged = deduplicate_records(merged)
    print_summary(merged)

    if args.csv:
        write_csv(merged, args.csv)
        print(f"Saved CSV to {args.csv}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON to {args.json}")

    if args.print_records:
        print(json.dumps(merged, indent=2, ensure_ascii=False))

    if not args.csv and not args.json and not args.print_records:
        print(json.dumps(merged[:5], indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())