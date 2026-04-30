from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CONSUMPTION_PATH = Path("data/consumption_dk1_raw.csv")


def build_grid_state_summary(
    csv_path: Path | str = DEFAULT_CONSUMPTION_PATH,
    *,
    peak_quantile: float = 0.95,
) -> dict[str, Any]:
    """Read the raw consumption CSV and derive simple system-level peak summaries."""

    csv_path = Path(csv_path)
    consumption_df = pd.read_csv(csv_path, parse_dates=["TimeDK", "TimeUTC"])

    required_columns = {
        "TimeDK",
        "GridArea",
        "GridCompanyName",
        "ConsumptionkWh",
    }
    missing = required_columns.difference(consumption_df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {csv_path}: {sorted(missing)}")

    consumption_df = consumption_df.copy()
    consumption_df["ConsumptionkWh"] = pd.to_numeric(consumption_df["ConsumptionkWh"], errors="coerce")
    consumption_df = consumption_df.dropna(
        subset=["TimeDK", "GridArea", "GridCompanyName", "ConsumptionkWh"]
    )
    consumption_df["GridArea"] = consumption_df["GridArea"].astype(str).str.zfill(3)
    consumption_df = consumption_df.sort_values(["TimeDK", "GridArea"]).reset_index(drop=True)

    system_hourly = (
        consumption_df.groupby("TimeDK", as_index=False)["ConsumptionkWh"]
        .sum()
        .sort_values("TimeDK")
        .reset_index(drop=True)
    )

    peak_threshold_kWh = float(system_hourly["ConsumptionkWh"].quantile(peak_quantile))
    system_hourly["is_peak"] = system_hourly["ConsumptionkWh"] >= peak_threshold_kWh

    monthly_consumption = (
        system_hourly.assign(Month=system_hourly["TimeDK"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)["ConsumptionkWh"]
        .sum()
        .rename(columns={"ConsumptionkWh": "monthly_consumption_kWh"})
    )
    monthly_consumption["monthly_consumption_MWh"] = monthly_consumption["monthly_consumption_kWh"] / 1000.0

    yearly_peak_counts = (
        system_hourly.assign(Year=system_hourly["TimeDK"].dt.year)
        .groupby("Year", as_index=False)["is_peak"]
        .sum()
        .rename(columns={"is_peak": "peak_hours"})
    )

    return {
        "raw_consumption": consumption_df,
        "system_hourly": system_hourly,
        "peak_threshold_kWh": peak_threshold_kWh,
        "peak_threshold_MWh": peak_threshold_kWh / 1000.0,
        "monthly_consumption": monthly_consumption,
        "yearly_peak_counts": yearly_peak_counts,
    }


def plot_historic_consumption_with_p95(
    grid_state: dict[str, Any] | None = None,
    csv_path: Path | str = DEFAULT_CONSUMPTION_PATH,
    *,
    peak_quantile: float = 0.95,
    figsize: tuple[int, int] = (14, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the historic hourly system consumption with the p95 threshold."""

    grid_state = grid_state or build_grid_state_summary(csv_path=csv_path, peak_quantile=peak_quantile)
    system_hourly = grid_state["system_hourly"]
    peak_threshold_MWh = float(grid_state["peak_threshold_MWh"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        system_hourly["TimeDK"],
        system_hourly["ConsumptionkWh"] / 1000.0,
        color="#9aa0a6",
        linewidth=0.9,
        alpha=0.9,
    )
    ax.axhline(
        peak_threshold_MWh,
        color="#c1121f",
        linewidth=2.0,
        label=f"Historical p95 = {peak_threshold_MWh:.1f} MWh",
    )
    ax.set_title("Historic DK1 Electricity Consumption and Peak Threshold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hourly system consumption (MWh)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    return fig, ax


def plot_consumption_trend_and_peak_hours(
    grid_state: dict[str, Any] | None = None,
    csv_path: Path | str = DEFAULT_CONSUMPTION_PATH,
    *,
    peak_quantile: float = 0.95,
    figsize: tuple[int, int] = (16, 5),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot monthly consumption development and yearly count of hours above p95."""

    grid_state = grid_state or build_grid_state_summary(csv_path=csv_path, peak_quantile=peak_quantile)
    monthly_consumption = grid_state["monthly_consumption"]
    yearly_peak_counts = grid_state["yearly_peak_counts"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_monthly, ax_peaks = axes

    grey_scale = np.linspace(0.80, 0.35, num=max(len(monthly_consumption), 1))
    monthly_colors = [str(level) for level in grey_scale]
    ax_monthly.bar(
        monthly_consumption["Month"],
        monthly_consumption["monthly_consumption_MWh"],
        width=25,
        color=monthly_colors,
        edgecolor="none",
    )
    ax_monthly.set_title("Monthly Aggregated DK1 Consumption Over Time")
    ax_monthly.set_xlabel("Month")
    ax_monthly.set_ylabel("Monthly consumption (MWh)")
    ax_monthly.tick_params(axis="x", rotation=45)
    ax_monthly.grid(axis="y", alpha=0.2)

    peak_values = yearly_peak_counts["peak_hours"].to_numpy(dtype=float)
    if len(peak_values) > 1 and peak_values.max() > peak_values.min():
        norm = (peak_values - peak_values.min()) / (peak_values.max() - peak_values.min())
    else:
        norm = np.zeros(len(peak_values))
    peak_colors = [plt.cm.YlOrRd(0.25 + 0.75 * value) for value in norm]

    ax_peaks.bar(
        yearly_peak_counts["Year"].astype(str),
        yearly_peak_counts["peak_hours"],
        color=peak_colors,
        edgecolor="none",
    )
    ax_peaks.set_title("Yearly Count of Hours Above the Historical p95")
    ax_peaks.set_xlabel("Year")
    ax_peaks.set_ylabel("Peak hours")
    ax_peaks.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    return fig, axes
