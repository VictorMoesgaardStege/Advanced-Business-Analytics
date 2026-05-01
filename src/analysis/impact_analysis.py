from pathlib import Path
from typing import Dict, Literal, Mapping, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from src.models.energy_shift_heuristic import (
    DECISION_WEIGHTS,
    FORECAST_HORIZON_HOURS,
    HOUSEHOLD_SHARE_OF_SYSTEM,
    SEGMENT_ASSUMPTIONS,
    SEGMENT_COLORS,
    assemble_window_for_issue_time,
    build_historical_segment_capacity,
    build_issue_catalog,
    build_shift_plan,
    load_predictions,
    load_system_load,
    run_rolling_impact_simulation,
)


Frequency = Literal["hourly", "daily"]

SEGMENT_LABELS = {
    "inflexible": "Inflexible load",
    "wet_loads": "Wet loads",
    "thermal": "Thermal load",
    "ev": "EV charging",
}

DOW_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def load_system_consumption_df(
    n_days: int = 60,
    csv_path: Optional[Path] = None,
    frequency: Frequency = "hourly",
) -> pd.DataFrame:
    if n_days <= 0:
        raise ValueError("n_days must be positive.")

    frequency = frequency.lower()
    if frequency not in {"hourly", "daily"}:
        raise ValueError("frequency must be either 'hourly' or 'daily'.")

    if csv_path is None:
        csv_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "consumption_dk1_raw.csv"
        )

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find consumption file: {csv_path}")

    raw = pd.read_csv(csv_path)

    datetime_col = next((col for col in ["TimeDK"] if col in raw.columns), None)
    if datetime_col is None:
        raise ValueError(
            f"Could not find a datetime column in {csv_path}. "
            f"Available columns: {list(raw.columns)}"
        )

    raw[datetime_col] = pd.to_datetime(raw[datetime_col])
    raw = raw.sort_values(datetime_col)

    consumption_col = next((col for col in ["ConsumptionkWh"] if col in raw.columns), None)
    if consumption_col is None:
        numeric_cols = raw.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) == 1:
            consumption_col = numeric_cols[0]
        else:
            raise ValueError(
                f"Could not find a consumption column in {csv_path}. "
                f"Available columns: {list(raw.columns)}"
            )

    df = (
        raw[[datetime_col, consumption_col]]
        .rename(
            columns={
                datetime_col: "datetime",
                consumption_col: "consumption_kwh",
            }
        )
        .assign(system_consumption_mwh=lambda data: data["consumption_kwh"] / 1000.0)
    )

    if frequency == "hourly":
        out = (
            df.assign(datetime=lambda data: data["datetime"].dt.floor("h"))
            .groupby("datetime", as_index=False)["system_consumption_mwh"]
            .sum()
            .sort_values("datetime")
        )
        out["date"] = out["datetime"].dt.floor("D")
        cutoff = out["datetime"].max() - pd.Timedelta(days=n_days)
        return out[out["datetime"] > cutoff].reset_index(drop=True)

    df["date"] = df["datetime"].dt.floor("D")
    obs_per_day = df.groupby("date").size().median()

    if obs_per_day > 1:
        out = (
            df.groupby("date", as_index=False)["system_consumption_mwh"]
            .sum()
            .sort_values("date")
        )
    else:
        out = (
            df.groupby("date", as_index=False)["system_consumption_mwh"]
            .first()
            .sort_values("date")
        )

    if len(out) < n_days:
        raise ValueError(
            f"Requested {n_days} days, but only found "
            f"{len(out)} daily values in {csv_path}"
        )

    return out.tail(n_days).reset_index(drop=True)


def add_household_load_split(
    df: pd.DataFrame,
    household_share: float = 0.32,
    system_col: str = "system_consumption_mwh",
) -> pd.DataFrame:
    _require_columns(df, [system_col])

    if not 0 <= household_share <= 1:
        raise ValueError("household_share must be between 0 and 1.")

    out = df.copy()
    out["household_load_mwh"] = out[system_col] * household_share
    out["other_system_load_mwh"] = out[system_col] - out["household_load_mwh"]
    return out


def get_central_segment_assumptions() -> Dict[str, Dict[str, float]]:
    return {
        "ev_charging": {
            "load_share": 0.050,
            "max_shiftable_share": 0.70,
            "max_wait_h": 24,
            "wait_penalty": 0.25,
        },
        "home_battery": {
            "load_share": 0.002,
            "max_shiftable_share": 1.00,
            "max_wait_h": 120,
            "wait_penalty": 0.05,
        },
        "thermal_heating": {
            "load_share": 0.100,
            "max_shiftable_share": 0.20,
            "max_wait_h": 6,
            "wait_penalty": 0.80,
        },
        "laundry": {
            "load_share": 0.060,
            "max_shiftable_share": 0.50,
            "max_wait_h": 24,
            "wait_penalty": 1.20,
        },
        "dishwasher": {
            "load_share": 0.030,
            "max_shiftable_share": 0.60,
            "max_wait_h": 12,
            "wait_penalty": 1.00,
        },
        "other_flexible": {
            "load_share": 0.050,
            "max_shiftable_share": 0.25,
            "max_wait_h": 12,
            "wait_penalty": 1.30,
        },
    }


def make_segment_assumption_table(
    segment_assumptions: Mapping[str, Mapping[str, float]]
) -> pd.DataFrame:
    table = pd.DataFrame(segment_assumptions).T.reset_index()
    table = table.rename(columns={"index": "segment"})
    table["share_of_household_load_pct"] = table["load_share"] * 100
    table["max_shiftable_share_pct"] = table["max_shiftable_share"] * 100
    return table


def make_load_share_summary(
    segment_assumptions: Mapping[str, Mapping[str, float]]
) -> pd.DataFrame:
    flexible_share = sum(params["load_share"] for params in segment_assumptions.values())
    inflexible_share = 1 - flexible_share

    if inflexible_share < -1e-9:
        raise ValueError(
            "Segment load shares sum to more than 1. "
            f"Total flexible share: {flexible_share:.3f}"
        )

    return pd.DataFrame(
        [
            {
                "load_type": "Flexible household load",
                "share_of_household_load": flexible_share,
                "share_of_household_load_pct": flexible_share * 100,
            },
            {
                "load_type": "Inflexible household load",
                "share_of_household_load": inflexible_share,
                "share_of_household_load_pct": inflexible_share * 100,
            },
        ]
    )


def add_flexible_segment_loads(
    df: pd.DataFrame,
    segment_assumptions: Mapping[str, Mapping[str, float]],
    household_col: str = "household_load_mwh",
) -> pd.DataFrame:
    if household_col not in df.columns:
        raise ValueError(
            f"Expected column '{household_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    flexible_household_share = sum(params["load_share"] for params in segment_assumptions.values())

    if flexible_household_share > 1:
        raise ValueError(
            "Flexible segment shares cannot sum to more than 1. "
            f"Current sum: {flexible_household_share:.3f}"
        )

    out["flexible_household_load_mwh"] = out[household_col] * flexible_household_share
    out["inflexible_household_load_mwh"] = out[household_col] * (1 - flexible_household_share)

    baseline_cols = []
    flexible_cols = []

    for segment, params in segment_assumptions.items():
        baseline_col = f"{segment}_baseline_mwh"
        flexible_col = f"{segment}_flexible_mwh"
        out[baseline_col] = out[household_col] * params["load_share"]
        out[flexible_col] = out[baseline_col] * params["max_shiftable_share"]
        baseline_cols.append(baseline_col)
        flexible_cols.append(flexible_col)

    out["total_segment_baseline_mwh"] = out[baseline_cols].sum(axis=1)
    out["total_shiftable_energy_mwh"] = out[flexible_cols].sum(axis=1)
    return out


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _resolve_time_col(df: pd.DataFrame, date_col: str = "date") -> str:
    if (
        date_col == "date"
        and "datetime" in df.columns
        and "date" in df.columns
        and df["datetime"].nunique() > df["date"].nunique()
    ):
        return "datetime"
    return date_col


def _time_axis_labels(date_col: str) -> tuple[str, str]:
    if date_col == "datetime":
        return "Time", "Consumption (MWh/hour)"
    return "Date", "Consumption (MWh/day)"


def plot_system_consumption(
    df: pd.DataFrame,
    date_col: str = "date",
    system_col: str = "system_consumption_mwh",
) -> None:
    date_col = _resolve_time_col(df, date_col)
    xlabel, ylabel = _time_axis_labels(date_col)
    _require_columns(df, [date_col, system_col])

    plt.figure(figsize=(12, 5))
    plt.plot(df[date_col], df[system_col])
    plt.title("Total DK1 Electricity Consumption")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_household_split(
    df: pd.DataFrame,
    date_col: str = "date",
) -> None:
    date_col = _resolve_time_col(df, date_col)
    xlabel, ylabel = _time_axis_labels(date_col)
    _require_columns(
        df,
        [date_col, "system_consumption_mwh", "household_load_mwh", "other_system_load_mwh"],
    )

    plt.figure(figsize=(12, 5))
    plt.plot(df[date_col], df["system_consumption_mwh"], label="Total system load")
    plt.plot(df[date_col], df["household_load_mwh"], label="Estimated household load")
    plt.plot(df[date_col], df["other_system_load_mwh"], label="Other system load")
    plt.title("Estimated Household Share of DK1 Consumption")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_flexible_vs_inflexible(
    df: pd.DataFrame,
    date_col: str = "date",
) -> None:
    date_col = _resolve_time_col(df, date_col)
    xlabel, ylabel = _time_axis_labels(date_col)
    _require_columns(
        df,
        [
            date_col,
            "household_load_mwh",
            "flexible_household_load_mwh",
            "inflexible_household_load_mwh",
        ],
    )

    plt.figure(figsize=(12, 5))
    plt.plot(df[date_col], df["household_load_mwh"], label="Estimated household load")
    plt.plot(
        df[date_col],
        df["inflexible_household_load_mwh"],
        label="Inflexible household load",
    )
    plt.plot(
        df[date_col],
        df["flexible_household_load_mwh"],
        label="Flexible household load",
    )
    plt.title("Flexible vs. Inflexible Household Consumption")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_segment_shares(segment_table: pd.DataFrame) -> None:
    _require_columns(segment_table, ["segment", "share_of_household_load_pct"])

    plt.figure(figsize=(8, 5))
    plt.bar(segment_table["segment"], segment_table["share_of_household_load_pct"])
    plt.title("Flexible Load Segments as Share of Household Electricity")
    plt.xlabel("Segment")
    plt.ylabel("Share of household load (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_segment_loads(
    df: pd.DataFrame,
    segment_assumptions: Mapping[str, Mapping[str, float]],
    date_col: str = "date",
) -> None:
    date_col = _resolve_time_col(df, date_col)
    xlabel, ylabel = _time_axis_labels(date_col)
    baseline_cols = [f"{segment}_baseline_mwh" for segment in segment_assumptions.keys()]
    _require_columns(df, [date_col] + baseline_cols)

    plt.figure(figsize=(12, 5))
    for segment in segment_assumptions.keys():
        plt.plot(df[date_col], df[f"{segment}_baseline_mwh"], label=segment)

    plt.title("Estimated Flexible Household Load Segments")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_shiftable_energy(
    df: pd.DataFrame,
    date_col: str = "date",
) -> None:
    date_col = _resolve_time_col(df, date_col)
    xlabel, ylabel = _time_axis_labels(date_col)
    _require_columns(
        df,
        [
            date_col,
            "household_load_mwh",
            "total_segment_baseline_mwh",
            "total_shiftable_energy_mwh",
        ],
    )

    plt.figure(figsize=(12, 5))
    plt.plot(df[date_col], df["household_load_mwh"], label="Estimated household load")
    plt.plot(df[date_col], df["total_segment_baseline_mwh"], label="Flexible segment baseline")
    plt.plot(df[date_col], df["total_shiftable_energy_mwh"], label="Actually shiftable energy")
    plt.title("From Household Load to Shiftable Energy")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_segment_shift_windows(segment_table: pd.DataFrame) -> None:
    _require_columns(segment_table, ["segment", "max_wait_h"])

    plt.figure(figsize=(8, 5))
    plt.bar(segment_table["segment"], segment_table["max_wait_h"])
    plt.title("Maximum Shift Window by Flexible Segment")
    plt.xlabel("Segment")
    plt.ylabel("Maximum wait (hours)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_segment_shiftable_shares(segment_table: pd.DataFrame) -> None:
    _require_columns(segment_table, ["segment", "max_shiftable_share_pct"])

    plt.figure(figsize=(8, 5))
    plt.bar(segment_table["segment"], segment_table["max_shiftable_share_pct"])
    plt.title("Maximum Shiftable Share by Flexible Segment")
    plt.xlabel("Segment")
    plt.ylabel("Maximum shiftable share (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def make_heuristic_assumption_table(
    segment_assumptions: Mapping[str, Mapping[str, float]] = SEGMENT_ASSUMPTIONS,
) -> pd.DataFrame:
    records = []
    for segment, params in segment_assumptions.items():
        share = float(params["share_of_household_load"])
        max_shiftable = float(params["max_shiftable_share"])
        records.append(
            {
                "segment": segment,
                "label": SEGMENT_LABELS.get(segment, segment.replace("_", " ").title()),
                "share_of_household_load_pct": 100.0 * share,
                "max_shiftable_share_pct": 100.0 * max_shiftable,
                "max_shiftable_share_of_household_pct": 100.0 * share * max_shiftable,
                "max_wait_h": int(params["max_wait_h"]),
                "wait_penalty": float(params["wait_penalty"]),
            }
        )

    return pd.DataFrame(records)


def make_heuristic_assumption_summary(
    segment_assumptions: Mapping[str, Mapping[str, float]] = SEGMENT_ASSUMPTIONS,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
) -> pd.Series:
    flexible_segments = {
        segment: params
        for segment, params in segment_assumptions.items()
        if float(params["max_shiftable_share"]) > 0
    }
    flexible_share = sum(
        float(params["share_of_household_load"])
        for params in flexible_segments.values()
    )
    max_theoretical_shiftable_share = sum(
        float(params["share_of_household_load"]) * float(params["max_shiftable_share"])
        for params in flexible_segments.values()
    )

    return pd.Series(
        {
            "household_share_of_system_pct": 100.0 * household_share_of_system,
            "flexible_share_of_household_pct": 100.0 * flexible_share,
            "max_theoretical_shiftable_share_of_household_pct": (
                100.0 * max_theoretical_shiftable_share
            ),
            "max_theoretical_shiftable_share_of_system_pct": (
                100.0 * household_share_of_system * max_theoretical_shiftable_share
            ),
        }
    )


def make_heuristic_assumption_markdown_tables(
    segment_assumptions: Mapping[str, Mapping[str, float]] = SEGMENT_ASSUMPTIONS,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
) -> tuple[str, str]:
    assumption_summary = make_heuristic_assumption_summary(
        segment_assumptions=segment_assumptions,
        household_share_of_system=household_share_of_system,
    )
    assumption_table = make_heuristic_assumption_table(
        segment_assumptions=segment_assumptions
    )

    summary_rows = [
        (
            "Household share of total system load",
            f"{assumption_summary['household_share_of_system_pct']:.1f}%",
        ),
        (
            "Flexible share of household load",
            f"{assumption_summary['flexible_share_of_household_pct']:.1f}%",
        ),
        (
            "Maximum theoretical shiftable share of household load",
            (
                f"{assumption_summary['max_theoretical_shiftable_share_of_household_pct']:.1f}%"
            ),
        ),
        (
            "Maximum theoretical shiftable share of total system load",
            (
                f"{assumption_summary['max_theoretical_shiftable_share_of_system_pct']:.1f}%"
            ),
        ),
    ]

    summary_md = ["| Assumption | Value |", "|---|---:|"]
    for label, value in summary_rows:
        summary_md.append(f"| {label} | {value} |")

    segment_md = [
        (
            "| Segment | Share of household load | Max shiftable share within segment | "
            "Max shiftable share of household load | Max wait | Wait penalty |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in assumption_table.iterrows():
        segment_md.append(
            "| "
            f"{row['label']} | "
            f"{row['share_of_household_load_pct']:.1f}% | "
            f"{row['max_shiftable_share_pct']:.1f}% | "
            f"{row['max_shiftable_share_of_household_pct']:.1f}% | "
            f"{int(row['max_wait_h'])} h | "
            f"{row['wait_penalty']:.2f} |"
        )

    return "\n".join(summary_md), "\n".join(segment_md)


def display_heuristic_assumption_tables(
    segment_assumptions: Mapping[str, Mapping[str, float]] = SEGMENT_ASSUMPTIONS,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
) -> tuple[str, str]:
    summary_md, segment_md = make_heuristic_assumption_markdown_tables(
        segment_assumptions=segment_assumptions,
        household_share_of_system=household_share_of_system,
    )
    display(Markdown(summary_md))
    display(Markdown(segment_md))
    return summary_md, segment_md


def run_granular_shift_illustration(
    issue_time: Optional[pd.Timestamp] = None,
    preds: Optional[pd.DataFrame] = None,
    system_load: Optional[pd.DataFrame] = None,
    segment_assumptions: Mapping[str, Mapping[str, float]] = SEGMENT_ASSUMPTIONS,
    decision_weights: Mapping[str, float] = DECISION_WEIGHTS,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
    historical_segment_capacity: Optional[pd.DataFrame] = None,
    show_plot: bool = True,
) -> dict:
    if preds is None:
        preds = load_predictions()
    if system_load is None:
        system_load = load_system_load()

    issue_catalog = build_issue_catalog(preds)
    if issue_time is None:
        issue_time = pd.Timestamp(issue_catalog.iloc[-1]["issue_time"])
    else:
        issue_time = pd.Timestamp(issue_time)

    if historical_segment_capacity is None:
        historical_segment_capacity = build_historical_segment_capacity(
            system_load=system_load,
            segment_assumptions=dict(segment_assumptions),
            household_share_of_system=household_share_of_system,
        )

    window_df = assemble_window_for_issue_time(
        issue_time=issue_time,
        preds=preds,
        system_load=system_load,
        household_share_of_system=household_share_of_system,
        segment_assumptions=dict(segment_assumptions),
    )
    detail_df, shifts_df = build_shift_plan(
        window_df=window_df,
        segment_assumptions=dict(segment_assumptions),
        decision_weights=dict(decision_weights),
        historical_segment_capacity=historical_segment_capacity,
    )
    detail_df = detail_df.copy()
    detail_df["net_shift_mwh"] = detail_df["shifted_system_load_mwh"] - detail_df["system_load_mwh"]

    shifted_energy_mwh = float(shifts_df["shift_mwh"].sum()) if not shifts_df.empty else 0.0
    total_system_consumption_mwh = float(detail_df["system_load_mwh"].sum())
    total_household_consumption_mwh = float(detail_df["household_load_mwh"].sum())

    summary = pd.Series(
        {
            "selected_issue_time": issue_time,
            "window_start": detail_df["target_time"].min(),
            "window_end": detail_df["target_time"].max(),
            "horizon_hours": int(len(detail_df)),
            "n_shift_events": int(len(shifts_df)),
            "shifted_energy_mwh": shifted_energy_mwh,
            "modeled_total_system_consumption_mwh": total_system_consumption_mwh,
            "modeled_total_household_consumption_mwh": total_household_consumption_mwh,
            "shifted_share_of_system_consumption_pct": (
                100.0 * shifted_energy_mwh / total_system_consumption_mwh
            ),
            "shifted_share_of_household_consumption_pct": (
                100.0 * shifted_energy_mwh / total_household_consumption_mwh
            ),
            "realized_household_savings_eur": float(
                detail_df["baseline_household_cost_eur"].sum()
                - detail_df["shifted_household_cost_eur"].sum()
            ),
            "predicted_household_savings_eur": float(
                detail_df["baseline_predicted_cost_eur"].sum()
                - detail_df["shifted_predicted_cost_eur"].sum()
            ),
        }
    )

    fig = None
    axes = None
    if show_plot:
        fig, axes = plot_granular_shift_illustration(
            detail_df=detail_df,
            segment_assumptions=segment_assumptions,
            segment_colors=SEGMENT_COLORS,
        )

    return {
        "summary": summary,
        "profiles": detail_df,
        "shifts": shifts_df,
        "issue_catalog": issue_catalog,
        "historical_segment_capacity": historical_segment_capacity,
        "figure": fig,
        "axes": axes,
    }


def plot_granular_shift_illustration(
    detail_df: pd.DataFrame,
    segment_assumptions: Mapping[str, Mapping[str, float]] = SEGMENT_ASSUMPTIONS,
    segment_colors: Mapping[str, str] = SEGMENT_COLORS,
) -> tuple[plt.Figure, np.ndarray]:
    required_cols = [
        "target_time",
        "system_load_mwh",
        "shifted_system_load_mwh",
        "household_load_mwh",
        "shifted_household_load_mwh",
        "predicted",
        "DayAheadPriceEUR",
    ]
    _require_columns(detail_df, required_cols)

    plot_df = detail_df.copy()
    if "net_shift_mwh" not in plot_df.columns:
        plot_df["net_shift_mwh"] = plot_df["shifted_system_load_mwh"] - plot_df["system_load_mwh"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)

    axes[0].plot(
        plot_df["target_time"],
        plot_df["system_load_mwh"],
        label="Baseline system load",
        color="#90A4AE",
        linewidth=1.6,
    )
    axes[0].plot(
        plot_df["target_time"],
        plot_df["shifted_system_load_mwh"],
        label="Shifted system load",
        color="#1B5E20",
        linewidth=1.6,
    )
    axes[0].set_title("Hourly system load in the selected 120h window")
    axes[0].set_ylabel("MWh")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        plot_df["target_time"],
        plot_df["household_load_mwh"],
        label="Baseline household load",
        color="#B0BEC5",
        linewidth=1.5,
    )
    axes[1].plot(
        plot_df["target_time"],
        plot_df["shifted_household_load_mwh"],
        label="Shifted household load",
        color="#2E7D32",
        linewidth=1.5,
    )
    axes[1].set_title("Hourly household load in the selected 120h window")
    axes[1].set_ylabel("MWh")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    bar_colors = np.where(plot_df["net_shift_mwh"] >= 0, "#43A047", "#E53935")
    axes[2].bar(plot_df["target_time"], plot_df["net_shift_mwh"], color=bar_colors, width=0.03)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Net shift by hour in the selected 120h window")
    axes[2].set_ylabel("Shifted MWh")
    axes[2].grid(alpha=0.25)

    axes[3].plot(
        plot_df["target_time"],
        plot_df["predicted"],
        label="XGBoost predicted price",
        color="#1565C0",
        linewidth=1.4,
    )
    axes[3].plot(
        plot_df["target_time"],
        plot_df["DayAheadPriceEUR"],
        label="Actual price",
        color="black",
        linewidth=1.1,
        alpha=0.7,
    )
    axes[3].set_title("Price context for the same hours")
    axes[3].set_ylabel("EUR/MWh")
    axes[3].grid(alpha=0.25)
    axes[3].legend()

    axes[3].set_title("Shifted household energy by segment")
    axes[3].set_ylabel("MWh")
    axes[3].grid(alpha=0.25)
    axes[3].legend(ncol=3, fontsize=9)

    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig, axes


def run_90_day_impact_simulation(
    simulation_days: int = 90,
    issue_hour: int = 12,
    window_stride_days: Optional[int] = None,
    preds: Optional[pd.DataFrame] = None,
    system_load: Optional[pd.DataFrame] = None,
    segment_assumptions: Mapping[str, Mapping[str, float]] = SEGMENT_ASSUMPTIONS,
    decision_weights: Mapping[str, float] = DECISION_WEIGHTS,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
) -> dict:
    result = run_rolling_impact_simulation(
        simulation_days=simulation_days,
        issue_hour=issue_hour,
        window_stride_days=window_stride_days,
        preds=preds,
        system_load=system_load,
        household_share_of_system=household_share_of_system,
        segment_assumptions=dict(segment_assumptions),
        decision_weights=dict(decision_weights),
        forecast_horizon_hours=FORECAST_HORIZON_HOURS,
    )

    summary = result["summary"].copy()
    summary["avg_abs_system_net_shift_mwh"] = float(result["profiles"]["net_shift_mwh"].abs().mean())
    result["summary"] = summary
    return result


def build_average_net_shift_heatmap(
    simulation_profiles_df: pd.DataFrame,
    day_order: list[str] = DOW_ORDER,
) -> pd.DataFrame:
    _require_columns(simulation_profiles_df, ["day_of_week", "hour", "net_shift_mwh"])

    return (
        simulation_profiles_df.groupby(["day_of_week", "hour"], as_index=False)["net_shift_mwh"]
        .mean()
        .assign(
            day_of_week=lambda df: pd.Categorical(
                df["day_of_week"],
                categories=day_order,
                ordered=True,
            )
        )
        .sort_values(["day_of_week", "hour"])
        .pivot(index="day_of_week", columns="hour", values="net_shift_mwh")
    )


def plot_average_net_shift_heatmap(
    simulation_profiles_df: pd.DataFrame,
) -> tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    heatmap_df = build_average_net_shift_heatmap(simulation_profiles_df)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(heatmap_df.to_numpy(), aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_title("Average net shift by day of week and hour of day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_xticks(np.arange(24))
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index.tolist())
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Average net shift (MWh)")
    plt.tight_layout()
    return heatmap_df, fig, ax


def plot_sample_household_impact_windows(
    simulation_profiles_df: pd.DataFrame,
    n_windows: int = 3,
    window_hours: int = 48,
) -> tuple[plt.Figure, np.ndarray]:
    plot_df = simulation_profiles_df.copy()
    day_counts = (
        plot_df.assign(day=plot_df["target_time"].dt.normalize())
        .groupby("day")
        .size()
        .reset_index(name="n_hours")
    )
    full_days = day_counts.loc[day_counts["n_hours"] == 24, "day"].reset_index(drop=True)
    valid_window_starts = full_days[
        full_days.shift(-1) == (full_days + pd.Timedelta(days=1))
    ].reset_index(drop=True)

    if len(valid_window_starts) < n_windows:
        raise ValueError(
            f"Expected at least {n_windows} valid windows, found {len(valid_window_starts)}."
        )

    window_positions = np.linspace(0, len(valid_window_starts) - 1, n_windows, dtype=int)
    window_starts = valid_window_starts.iloc[window_positions].tolist()
    impact_color = "#2E7D32"

    fig, axes = plt.subplots(n_windows, 1, figsize=(18, 4 * n_windows), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, window_start in zip(axes, window_starts):
        window_start = pd.Timestamp(window_start)
        window_end = window_start + pd.Timedelta(hours=window_hours - 1)
        window_df = plot_df.loc[
            (plot_df["target_time"] >= window_start)
            & (plot_df["target_time"] <= window_end)
        ].copy()

        ax.plot(
            window_df["target_time"],
            window_df["household_load_mwh"],
            color="#9E9E9E",
            linewidth=1.8,
            label="Current household consumption",
        )
        ax.plot(
            window_df["target_time"],
            window_df["shifted_household_load_mwh"],
            color="#8E24AA",
            linewidth=1.8,
            label="Shifted household consumption",
        )
        ax.fill_between(
            window_df["target_time"],
            window_df["household_load_mwh"],
            window_df["shifted_household_load_mwh"],
            color=impact_color,
            alpha=0.45,
            label="Impact",
        )
        ax.set_title(
            f"Sampled {window_hours}-Hour Window: "
            f"{window_start:%Y-%m-%d %H:%M} to {window_end:%Y-%m-%d %H:%M}"
        )
        ax.set_ylabel("Household load (MWh)")
        ax.set_xlim(window_start, window_end)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Date and time")
    plt.tight_layout()
    return fig, axes


def build_peak_excess_summary(
    simulation_profiles_df: pd.DataFrame,
    system_load: pd.DataFrame,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
) -> pd.DataFrame:
    required_cols = [
        "household_load_mwh",
        "shifted_household_load_mwh",
        "system_load_mwh",
        "shifted_system_load_mwh",
    ]
    _require_columns(simulation_profiles_df, required_cols)
    _require_columns(system_load, ["system_load_mwh"])

    historic_p95_thresholds = {
        "Total household": float(
            (system_load["system_load_mwh"] * household_share_of_system).quantile(0.95)
        ),
        "Total system energy": float(system_load["system_load_mwh"].quantile(0.95)),
    }

    def compute_peak_excess_summary(
        df: pd.DataFrame,
        baseline_col: str,
        shifted_col: str,
        label: str,
        baseline_p95: float,
    ) -> dict:
        current_excess = (df[baseline_col] - baseline_p95).clip(lower=0)
        shifted_excess = (df[shifted_col] - baseline_p95).clip(lower=0)

        current_total = float(current_excess.sum())
        shifted_total = float(shifted_excess.sum())
        impact_total = current_total - shifted_total
        reduction_pct = 100.0 * impact_total / current_total if current_total > 0 else 0.0

        return {
            "case": label,
            "baseline_p95_mwh": baseline_p95,
            "current_above_p95_mwh": current_total,
            "impact_mwh": impact_total,
            "shifted_above_p95_mwh": shifted_total,
            "peak_reduction_pct": reduction_pct,
            "current_hours_above_p95": int((df[baseline_col] > baseline_p95).sum()),
            "shifted_hours_above_p95": int((df[shifted_col] > baseline_p95).sum()),
        }

    return pd.DataFrame(
        [
            compute_peak_excess_summary(
                simulation_profiles_df,
                baseline_col="household_load_mwh",
                shifted_col="shifted_household_load_mwh",
                label="Total household",
                baseline_p95=historic_p95_thresholds["Total household"],
            ),
            compute_peak_excess_summary(
                simulation_profiles_df,
                baseline_col="system_load_mwh",
                shifted_col="shifted_system_load_mwh",
                label="Total system energy",
                baseline_p95=historic_p95_thresholds["Total system energy"],
            ),
        ]
    )


def _plot_peak_excess_waterfall(ax: plt.Axes, row: pd.Series) -> None:
    current_color = "#9E9E9E"
    shifted_color = "#8E24AA"
    impact_color = "#2E7D32"

    current_total = float(row["current_above_p95_mwh"])
    impact_total = float(row["impact_mwh"])
    shifted_total = float(row["shifted_above_p95_mwh"])
    scale = max(current_total, shifted_total, 1.0)

    ax.bar(0, current_total, width=0.65, color=current_color)
    ax.bar(1, -impact_total, width=0.65, bottom=current_total, color=impact_color)
    ax.bar(2, shifted_total, width=0.65, color=shifted_color)

    ax.text(0, current_total + scale * 0.03, f"{current_total:.1f}", ha="center", va="bottom")
    ax.text(
        1,
        current_total - impact_total / 2,
        f"-{impact_total:.1f}",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    ax.text(2, shifted_total + scale * 0.03, f"{shifted_total:.1f}", ha="center", va="bottom")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Current > historic p95", "Impact", "Shifted > historic p95"])
    ax.set_ylabel("Cumulative load above historic p95 (MWh)")
    ax.set_title(
        f"{row['case']} - 90-Day Peak Waterfall\n"
        f"Historic p95 threshold: {row['baseline_p95_mwh']:.2f} MWh | "
        f"Peak demand reduction: {row['peak_reduction_pct']:.2f}%"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, scale * 1.25)


def plot_peak_excess_waterfalls(
    simulation_profiles_df: pd.DataFrame,
    system_load: pd.DataFrame,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
) -> tuple[pd.DataFrame, plt.Figure, np.ndarray]:
    peak_excess_summary_df = build_peak_excess_summary(
        simulation_profiles_df=simulation_profiles_df,
        system_load=system_load,
        household_share_of_system=household_share_of_system,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax, (_, row) in zip(axes, peak_excess_summary_df.iterrows()):
        _plot_peak_excess_waterfall(ax, row)

    plt.tight_layout()
    return peak_excess_summary_df, fig, axes


__all__ = [
    "DECISION_WEIGHTS",
    "DOW_ORDER",
    "FORECAST_HORIZON_HOURS",
    "HOUSEHOLD_SHARE_OF_SYSTEM",
    "SEGMENT_ASSUMPTIONS",
    "SEGMENT_COLORS",
    "SEGMENT_LABELS",
    "Frequency",
    "add_flexible_segment_loads",
    "add_household_load_split",
    "build_average_net_shift_heatmap",
    "build_peak_excess_summary",
    "display_heuristic_assumption_tables",
    "get_central_segment_assumptions",
    "load_system_consumption_df",
    "make_heuristic_assumption_markdown_tables",
    "make_heuristic_assumption_summary",
    "make_heuristic_assumption_table",
    "make_load_share_summary",
    "make_segment_assumption_table",
    "plot_average_net_shift_heatmap",
    "plot_flexible_vs_inflexible",
    "plot_granular_shift_illustration",
    "plot_household_split",
    "plot_peak_excess_waterfalls",
    "plot_sample_household_impact_windows",
    "plot_segment_loads",
    "plot_segment_shares",
    "plot_segment_shift_windows",
    "plot_segment_shiftable_shares",
    "plot_shiftable_energy",
    "plot_system_consumption",
    "run_90_day_impact_simulation",
    "run_granular_shift_illustration",
]
