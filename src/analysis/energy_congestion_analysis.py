#!/usr/bin/env python3
"""Baseline energy congestion analysis for DK1 consumption data.

This module turns the raw hourly DK1 consumption file into a resilience-oriented
baseline analysis focused on:

- identifying peak and low-load hours
- defining a data-driven overload threshold
- estimating a maintenance-pressure proxy over time
- highlighting which grid areas contribute the most stress

The analysis is intentionally transparent about what is observed directly versus
what is inferred. Because the project only has consumption data, the script does
not claim to measure physical congestion or asset damage directly. Instead, it
creates a reproducible stress proxy from historical loading patterns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CONSUMPTION_PATH = Path("data/consumption_dk1_raw.csv")
DEFAULT_OUTPUT_DIR = Path("simulation_outputs")


def load_consumption_data(csv_path: Path | str = DEFAULT_CONSUMPTION_PATH) -> pd.DataFrame:
    """Load raw DK1 hourly consumption data."""

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, parse_dates=["TimeDK", "TimeUTC"])

    required_columns = {
        "TimeDK",
        "GridArea",
        "GridCompanyName",
        "ConsumptionkWh",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["ConsumptionkWh"] = pd.to_numeric(df["ConsumptionkWh"], errors="coerce")
    df = df.dropna(subset=["TimeDK", "GridArea", "GridCompanyName", "ConsumptionkWh"])
    df["GridArea"] = df["GridArea"].astype(str).str.zfill(3)

    return df.sort_values(["TimeDK", "GridArea"]).reset_index(drop=True)


def build_hourly_profiles(consumption_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create hourly system-level and grid-area-level load tables."""

    area_hourly = (
        consumption_df.groupby(["TimeDK", "GridArea", "GridCompanyName"], as_index=False)["ConsumptionkWh"]
        .sum()
        .sort_values(["GridArea", "TimeDK"])
        .reset_index(drop=True)
    )

    system_hourly = (
        area_hourly.groupby("TimeDK", as_index=False)["ConsumptionkWh"]
        .sum()
        .sort_values("TimeDK")
        .reset_index(drop=True)
    )

    return system_hourly, area_hourly


def _quantile_by_group(
    df: pd.DataFrame,
    value_col: str,
    quantile: float,
    group_cols: list[str] | None = None,
) -> pd.Series:
    if group_cols:
        return df.groupby(group_cols)[value_col].transform(lambda s: s.quantile(quantile))
    return pd.Series(df[value_col].quantile(quantile), index=df.index)


def add_stress_metrics(
    hourly_df: pd.DataFrame,
    *,
    value_col: str = "ConsumptionkWh",
    group_cols: list[str] | None = None,
    low_quantile: float = 0.05,
    peak_quantile: float = 0.95,
    overload_quantile: float = 0.99,
    critical_multiplier: float = 1.10,
) -> pd.DataFrame:
    """Add low-load, peak, overload, and maintenance-pressure metrics."""

    group_cols = group_cols or []
    sort_cols = [*group_cols, "TimeDK"]

    df = hourly_df.copy().sort_values(sort_cols).reset_index(drop=True)
    df["capacity_proxy_kWh"] = _quantile_by_group(df, value_col, overload_quantile, group_cols)
    df["peak_threshold_kWh"] = _quantile_by_group(df, value_col, peak_quantile, group_cols)
    df["low_threshold_kWh"] = _quantile_by_group(df, value_col, low_quantile, group_cols)

    if group_cols:
        previous_load = df.groupby(group_cols)[value_col].shift(1)
    else:
        previous_load = df[value_col].shift(1)

    df["delta_kWh"] = df[value_col].sub(previous_load).fillna(0.0)
    df["load_ratio"] = df[value_col] / df["capacity_proxy_kWh"].replace(0, np.nan)
    df["load_ratio"] = df["load_ratio"].fillna(0.0)
    df["ramp_ratio"] = df["delta_kWh"].abs() / df["capacity_proxy_kWh"].replace(0, np.nan)
    df["ramp_ratio"] = df["ramp_ratio"].fillna(0.0)

    df["is_low_load"] = df[value_col] <= df["low_threshold_kWh"]
    df["is_peak"] = df[value_col] >= df["peak_threshold_kWh"]
    df["is_overloaded"] = df["load_ratio"] > 1.0
    df["is_critical"] = df["load_ratio"] > critical_multiplier
    df["unused_capacity_ratio"] = np.clip(1.0 - df["load_ratio"], 0.0, None)

    overload_gap = np.clip(df["load_ratio"] - 1.0, 0.0, None)

    # Baseline wear rises with loading; overload and volatility add nonlinear stress.
    df["hourly_maintenance_pressure"] = (
        df["load_ratio"].pow(2)
        + 25.0 * overload_gap.pow(2)
        + 0.50 * df["ramp_ratio"]
    )

    return df


def summarise_daily_system(system_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate system stress to a daily horizon."""

    daily = (
        system_hourly.assign(Date=system_hourly["TimeDK"].dt.floor("D"))
        .groupby("Date", as_index=False)
        .agg(
            total_consumption_kWh=("ConsumptionkWh", "sum"),
            peak_load_kWh=("ConsumptionkWh", "max"),
            avg_load_kWh=("ConsumptionkWh", "mean"),
            overloaded_hours=("is_overloaded", "sum"),
            critical_hours=("is_critical", "sum"),
            low_load_hours=("is_low_load", "sum"),
            peak_hours=("is_peak", "sum"),
            daily_maintenance_pressure=("hourly_maintenance_pressure", "sum"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    daily["rolling_30d_pressure"] = (
        daily["daily_maintenance_pressure"].rolling(window=30, min_periods=7).sum()
    )
    daily["rolling_30d_overloaded_hours"] = (
        daily["overloaded_hours"].rolling(window=30, min_periods=7).sum()
    )

    return daily


def summarise_yearly_system(system_hourly: pd.DataFrame) -> pd.DataFrame:
    """Create long-horizon yearly trend metrics."""

    rows: list[dict[str, Any]] = []
    for year, group in system_hourly.groupby(system_hourly["TimeDK"].dt.year):
        rows.append(
            {
                "year": int(year),
                "hours": int(len(group)),
                "avg_load_MWh": group["ConsumptionkWh"].mean() / 1000.0,
                "peak_load_MWh": group["ConsumptionkWh"].max() / 1000.0,
                "p95_load_MWh": group["ConsumptionkWh"].quantile(0.95) / 1000.0,
                "p99_load_MWh": group["ConsumptionkWh"].quantile(0.99) / 1000.0,
                "overloaded_hours": int(group["is_overloaded"].sum()),
                "critical_hours": int(group["is_critical"].sum()),
                "maintenance_pressure": float(group["hourly_maintenance_pressure"].sum()),
            }
        )

    yearly = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    if not yearly.empty:
        yearly["p95_growth_pct"] = yearly["p95_load_MWh"].pct_change()
        yearly["overloaded_hours_growth_pct"] = yearly["overloaded_hours"].replace(0, np.nan).pct_change()
        yearly["maintenance_growth_pct"] = yearly["maintenance_pressure"].pct_change()

    return yearly


def summarise_area_stress(area_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stress at the grid-area level."""

    summary = (
        area_hourly.groupby(["GridArea", "GridCompanyName"], as_index=False)
        .agg(
            avg_load_MWh=("ConsumptionkWh", lambda s: s.mean() / 1000.0),
            peak_load_MWh=("ConsumptionkWh", lambda s: s.max() / 1000.0),
            capacity_proxy_MWh=("capacity_proxy_kWh", lambda s: s.iloc[0] / 1000.0),
            overloaded_hours=("is_overloaded", "sum"),
            critical_hours=("is_critical", "sum"),
            low_load_hours=("is_low_load", "sum"),
            peak_hours=("is_peak", "sum"),
            cumulative_maintenance_pressure=("hourly_maintenance_pressure", "sum"),
            max_load_ratio=("load_ratio", "max"),
        )
        .sort_values("cumulative_maintenance_pressure", ascending=False)
        .reset_index(drop=True)
    )

    return summary


def build_definition_table(report: dict[str, Any]) -> pd.DataFrame:
    """Create human-readable definitions for notebook and report use."""

    metrics = report["system_metrics"]
    return pd.DataFrame(
        [
            {
                "Concept": "Low-load hour",
                "Operational definition": (
                    f"Hourly DK1 system load at or below the historical 5th percentile "
                    f"({metrics['low_load_threshold_MWh']:.1f} MWh)."
                ),
                "Interpretation": (
                    "Low thermal stress and high unused capacity. With consumption data alone this "
                    "is an under-utilisation signal, not physical damage."
                ),
            },
            {
                "Concept": "Peak hour",
                "Operational definition": (
                    f"Hourly DK1 system load at or above the historical 95th percentile "
                    f"({metrics['peak_threshold_MWh']:.1f} MWh)."
                ),
                "Interpretation": (
                    "The grid is operating close to the top of its observed range and should be "
                    "monitored for recurring stress."
                ),
            },
            {
                "Concept": "Overloaded hour",
                "Operational definition": (
                    f"Hourly DK1 system load above the historical capacity proxy, defined as the "
                    f"99th percentile of observed load ({metrics['overload_threshold_MWh']:.1f} MWh)."
                ),
                "Interpretation": (
                    "This is treated as the point where observed loading moves beyond normal "
                    "historical operation and maintenance pressure starts accelerating."
                ),
            },
            {
                "Concept": "Maintenance pressure",
                "Operational definition": (
                    "Hourly proxy = load_ratio^2 + 25 * max(load_ratio - 1, 0)^2 + 0.5 * ramp_ratio, "
                    "aggregated over time."
                ),
                "Interpretation": (
                    "Normal loading creates baseline wear, but overload and strong hour-to-hour ramps "
                    "increase pressure nonlinearly."
                ),
            },
            {
                "Concept": "Maintenance attention trigger",
                "Operational definition": (
                    f"A 30-day rolling maintenance pressure above {metrics['maintenance_attention_threshold']:.0f} "
                    "or a visible cluster of overloaded hours."
                ),
                "Interpretation": (
                    "A practical flag for inspection planning or local reinforcement studies."
                ),
            },
        ]
    )


def build_resilience_baseline_report(
    csv_path: Path | str = DEFAULT_CONSUMPTION_PATH,
    *,
    low_quantile: float = 0.05,
    peak_quantile: float = 0.95,
    overload_quantile: float = 0.99,
    critical_multiplier: float = 1.10,
) -> dict[str, Any]:
    """Create a complete baseline resilience report from raw consumption data."""

    consumption_df = load_consumption_data(csv_path)
    system_hourly_raw, area_hourly_raw = build_hourly_profiles(consumption_df)

    system_hourly = add_stress_metrics(
        system_hourly_raw,
        low_quantile=low_quantile,
        peak_quantile=peak_quantile,
        overload_quantile=overload_quantile,
        critical_multiplier=critical_multiplier,
    )
    area_hourly = add_stress_metrics(
        area_hourly_raw,
        group_cols=["GridArea"],
        low_quantile=low_quantile,
        peak_quantile=peak_quantile,
        overload_quantile=overload_quantile,
        critical_multiplier=critical_multiplier,
    )

    daily_system = summarise_daily_system(system_hourly)
    yearly_system = summarise_yearly_system(system_hourly)
    area_summary = summarise_area_stress(area_hourly)

    system_capacity_proxy_kWh = float(system_hourly["capacity_proxy_kWh"].iloc[0])
    low_threshold_kWh = float(system_hourly["low_threshold_kWh"].iloc[0])
    peak_threshold_kWh = float(system_hourly["peak_threshold_kWh"].iloc[0])
    maintenance_attention_threshold = float(daily_system["rolling_30d_pressure"].quantile(0.90))

    top_area = area_summary.iloc[0]
    key_metrics = pd.DataFrame(
        [
            {"Metric": "Observed period start", "Value": system_hourly["TimeDK"].min()},
            {"Metric": "Observed period end", "Value": system_hourly["TimeDK"].max()},
            {"Metric": "System capacity proxy (historical p99, MWh)", "Value": round(system_capacity_proxy_kWh / 1000.0, 2)},
            {"Metric": "Peak-hour threshold (historical p95, MWh)", "Value": round(peak_threshold_kWh / 1000.0, 2)},
            {"Metric": "Low-load threshold (historical p05, MWh)", "Value": round(low_threshold_kWh / 1000.0, 2)},
            {"Metric": "Observed max load (MWh)", "Value": round(system_hourly["ConsumptionkWh"].max() / 1000.0, 2)},
            {"Metric": "Overloaded system hours", "Value": int(system_hourly["is_overloaded"].sum())},
            {"Metric": "Critical system hours", "Value": int(system_hourly["is_critical"].sum())},
            {"Metric": "Share of hours in peak zone (%)", "Value": round(100.0 * system_hourly["is_peak"].mean(), 2)},
            {"Metric": "Share of hours in low-load zone (%)", "Value": round(100.0 * system_hourly["is_low_load"].mean(), 2)},
            {"Metric": "Top stressed grid area", "Value": f"{top_area['GridArea']} - {top_area['GridCompanyName']}"},
            {"Metric": "Top area cumulative pressure", "Value": round(float(top_area["cumulative_maintenance_pressure"]), 2)},
            {"Metric": "30-day maintenance attention threshold", "Value": round(maintenance_attention_threshold, 2)},
        ]
    )

    report: dict[str, Any] = {
        "raw_consumption": consumption_df,
        "system_hourly": system_hourly,
        "area_hourly": area_hourly,
        "daily_system": daily_system,
        "yearly_system": yearly_system,
        "area_summary": area_summary,
        "key_metrics": key_metrics,
        "system_metrics": {
            "low_load_threshold_MWh": low_threshold_kWh / 1000.0,
            "peak_threshold_MWh": peak_threshold_kWh / 1000.0,
            "overload_threshold_MWh": system_capacity_proxy_kWh / 1000.0,
            "critical_threshold_MWh": critical_multiplier * system_capacity_proxy_kWh / 1000.0,
            "maintenance_attention_threshold": maintenance_attention_threshold,
        },
    }
    report["definition_table"] = build_definition_table(report)

    return report


def plot_resilience_baseline(
    report: dict[str, Any],
    *,
    top_n_areas: int = 10,
    figsize: tuple[int, int] = (16, 11),
) -> tuple[plt.Figure, np.ndarray]:
    """Create a combined resilience figure for the notebook and reports."""

    system_hourly = report["system_hourly"]
    daily_system = report["daily_system"]
    area_summary = report["area_summary"]
    metrics = report["system_metrics"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_duration, ax_heatmap, ax_pressure, ax_areas = axes.ravel()

    sorted_load = system_hourly["ConsumptionkWh"].sort_values(ascending=False).to_numpy() / 1000.0
    load_share = np.linspace(0, 100, num=len(sorted_load), endpoint=False)

    ax_duration.plot(load_share, sorted_load, color="#0b4f6c", linewidth=2.0)
    ax_duration.axhline(metrics["peak_threshold_MWh"], color="#f4a259", linestyle="--", linewidth=1.6, label="Peak threshold (p95)")
    ax_duration.axhline(metrics["overload_threshold_MWh"], color="#d1495b", linestyle="--", linewidth=1.6, label="Overload threshold (p99)")
    ax_duration.axhline(metrics["low_load_threshold_MWh"], color="#5fa8d3", linestyle=":", linewidth=1.6, label="Low-load threshold (p05)")
    ax_duration.set_title("System Load Duration Curve")
    ax_duration.set_xlabel("Share of observed hours (%)")
    ax_duration.set_ylabel("Hourly system load (MWh)")
    ax_duration.legend(frameon=False, fontsize=9)

    heatmap = (
        system_hourly.assign(month=system_hourly["TimeDK"].dt.month, hour=system_hourly["TimeDK"].dt.hour)
        .pivot_table(index="month", columns="hour", values="ConsumptionkWh", aggfunc="mean")
        .sort_index()
        / 1000.0
    )
    im = ax_heatmap.imshow(heatmap.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax_heatmap.set_title("Average System Load by Month and Hour")
    ax_heatmap.set_xlabel("Hour of day")
    ax_heatmap.set_ylabel("Month")
    ax_heatmap.set_xticks(np.arange(0, 24, 3))
    ax_heatmap.set_yticks(np.arange(12))
    ax_heatmap.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Average load (MWh)")

    ax_pressure.plot(
        daily_system["Date"],
        daily_system["rolling_30d_pressure"],
        color="#1b998b",
        linewidth=2.0,
        label="30-day rolling maintenance pressure",
    )
    ax_pressure.axhline(
        metrics["maintenance_attention_threshold"],
        color="#d1495b",
        linestyle="--",
        linewidth=1.6,
        label="Maintenance attention threshold",
    )
    ax_pressure.set_title("Long-Horizon Maintenance Pressure")
    ax_pressure.set_xlabel("Date")
    ax_pressure.set_ylabel("30-day rolling pressure index")
    ax_pressure.tick_params(axis="x", rotation=30)

    ax_pressure_b = ax_pressure.twinx()
    ax_pressure_b.plot(
        daily_system["Date"],
        daily_system["rolling_30d_overloaded_hours"],
        color="#edae49",
        linewidth=1.4,
        alpha=0.85,
        label="30-day rolling overloaded hours",
    )
    ax_pressure_b.set_ylabel("Overloaded hours in past 30 days")

    lines_a, labels_a = ax_pressure.get_legend_handles_labels()
    lines_b, labels_b = ax_pressure_b.get_legend_handles_labels()
    ax_pressure.legend(lines_a + lines_b, labels_a + labels_b, frameon=False, fontsize=9, loc="upper left")

    top_areas = area_summary.head(top_n_areas).sort_values("cumulative_maintenance_pressure", ascending=True)
    labels = [f"{row.GridArea}: {row.GridCompanyName}" for row in top_areas.itertuples()]
    ax_areas.barh(labels, top_areas["cumulative_maintenance_pressure"], color="#4f6d7a")
    ax_areas.set_title(f"Top {top_n_areas} Grid Areas by Cumulative Maintenance Pressure")
    ax_areas.set_xlabel("Cumulative pressure index")
    ax_areas.set_ylabel("")

    fig.suptitle(
        "DK1 Baseline Grid Resilience: Peaks, Overload Proxy, and Maintenance Pressure",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout()

    return fig, axes


def save_resilience_outputs(
    report: dict[str, Any],
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    figure_name: str = "energy_congestion_resilience_baseline.png",
) -> dict[str, Path]:
    """Persist figure and summary tables for reuse in the notebook/report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_resilience_baseline(report)
    figure_path = output_dir / figure_name
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    key_metrics_path = output_dir / "energy_congestion_key_metrics.csv"
    definitions_path = output_dir / "energy_congestion_definition_table.csv"
    yearly_path = output_dir / "energy_congestion_yearly_system_summary.csv"
    area_path = output_dir / "energy_congestion_area_summary.csv"
    daily_path = output_dir / "energy_congestion_daily_system_summary.csv"

    report["key_metrics"].to_csv(key_metrics_path, index=False)
    report["definition_table"].to_csv(definitions_path, index=False)
    report["yearly_system"].to_csv(yearly_path, index=False)
    report["area_summary"].to_csv(area_path, index=False)
    report["daily_system"].to_csv(daily_path, index=False)

    return {
        "figure": figure_path,
        "key_metrics": key_metrics_path,
        "definitions": definitions_path,
        "yearly_system": yearly_path,
        "area_summary": area_path,
        "daily_system": daily_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline DK1 energy congestion analysis.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CONSUMPTION_PATH, help="Path to raw consumption CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary files and combined figure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_resilience_baseline_report(args.csv)
    outputs = save_resilience_outputs(report, output_dir=args.output_dir)

    print("Energy congestion baseline analysis completed.")
    print(f"Observed period: {report['system_hourly']['TimeDK'].min()} -> {report['system_hourly']['TimeDK'].max()}")
    print(
        "Overload threshold (historical p99): "
        f"{report['system_metrics']['overload_threshold_MWh']:.2f} MWh"
    )
    print(
        "Maintenance attention threshold (30-day rolling): "
        f"{report['system_metrics']['maintenance_attention_threshold']:.2f}"
    )
    print(f"Saved combined figure to {outputs['figure']}")
    print(f"Saved key metrics to {outputs['key_metrics']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
