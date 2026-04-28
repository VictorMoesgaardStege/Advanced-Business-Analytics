from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_daily_system_consumption_df(
    n_days: int = 60,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Loads raw DK1 consumption data from CSV and returns the last n_days
    as a dataframe with daily total system consumption [MWh/day].

    Expected raw columns:
    - TimeDK
    - ConsumptionkWh

    Works for both:
    - hourly/sub-daily data -> aggregated to daily totals
    - already daily data -> used directly
    """

    if csv_path is None:
        csv_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "consumption_dk1_raw.csv"
        )

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find consumption file: {csv_path}")

    df = pd.read_csv(csv_path)

    possible_datetime_cols = ["TimeDK"]
    datetime_col = next((col for col in possible_datetime_cols if col in df.columns), None)

    if datetime_col is None:
        raise ValueError(
            f"Could not find a datetime column in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    possible_consumption_cols = ["ConsumptionkWh"]
    consumption_col = next((col for col in possible_consumption_cols if col in df.columns), None)

    if consumption_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 1:
            consumption_col = numeric_cols[0]
        else:
            raise ValueError(
                f"Could not find a consumption column in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

    df = df[[datetime_col, consumption_col]].copy()
    df = df.rename(
        columns={
            datetime_col: "datetime",
            consumption_col: "consumption_kwh",
        }
    )

    df["consumption_mwh"] = df["consumption_kwh"] / 1000.0
    df["date"] = df["datetime"].dt.floor("D")

    obs_per_day = df.groupby("date").size().median()

    if obs_per_day > 1:
        daily_consumption = (
            df.groupby("date", as_index=False)["consumption_mwh"]
            .sum()
            .sort_values("date")
        )
    else:
        daily_consumption = (
            df.groupby("date", as_index=False)["consumption_mwh"]
            .first()
            .sort_values("date")
        )

    if len(daily_consumption) < n_days:
        raise ValueError(
            f"Requested {n_days} days, but only found "
            f"{len(daily_consumption)} daily values in {csv_path}"
        )

    daily_consumption = daily_consumption.tail(n_days).reset_index(drop=True)
    daily_consumption = daily_consumption.rename(
        columns={"consumption_mwh": "system_consumption_mwh"}
    )

    return daily_consumption


def load_daily_system_consumption(
    n_days: int = 60,
    csv_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Backwards-compatible wrapper returning only daily consumption values.
    """

    daily_df = load_daily_system_consumption_df(
        n_days=n_days,
        csv_path=csv_path,
    )

    return daily_df["system_consumption_mwh"].to_numpy()


def load_hourly_system_consumption_df(
    n_days: int = 60,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Loads raw DK1 consumption data from CSV and returns hourly system
    consumption for the last n_days.

    Expected raw columns:
    - TimeDK
    - ConsumptionkWh
    """

    if csv_path is None:
        csv_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "consumption_dk1_raw.csv"
        )

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find consumption file: {csv_path}")

    df = pd.read_csv(csv_path)

    possible_datetime_cols = ["TimeDK"]
    datetime_col = next((col for col in possible_datetime_cols if col in df.columns), None)

    if datetime_col is None:
        raise ValueError(
            f"Could not find a datetime column in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    possible_consumption_cols = ["ConsumptionkWh"]
    consumption_col = next((col for col in possible_consumption_cols if col in df.columns), None)

    if consumption_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 1:
            consumption_col = numeric_cols[0]
        else:
            raise ValueError(
                f"Could not find a consumption column in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

    hourly_consumption = (
        df[[datetime_col, consumption_col]]
        .rename(
            columns={
                datetime_col: "datetime",
                consumption_col: "consumption_kwh",
            }
        )
        .assign(
            datetime=lambda data: data["datetime"].dt.floor("h"),
            system_consumption_mwh=lambda data: data["consumption_kwh"] / 1000.0,
        )
        .groupby("datetime", as_index=False)["system_consumption_mwh"]
        .sum()
        .sort_values("datetime")
    )
    hourly_consumption["date"] = hourly_consumption["datetime"].dt.floor("D")

    cutoff = hourly_consumption["datetime"].max() - pd.Timedelta(days=n_days)
    hourly_consumption = hourly_consumption[
        hourly_consumption["datetime"] > cutoff
    ].reset_index(drop=True)

    return hourly_consumption


def load_hourly_system_consumption(
    n_days: int = 60,
    csv_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Backwards-compatible wrapper returning only hourly consumption values.
    """

    hourly_df = load_hourly_system_consumption_df(
        n_days=n_days,
        csv_path=csv_path,
    )

    return hourly_df["system_consumption_mwh"].to_numpy()


def add_household_load_split(
    df: pd.DataFrame,
    household_share: float = 0.32,
    system_col: str = "system_consumption_mwh",
) -> pd.DataFrame:
    """
    Adds estimated household and non-household system load columns.
    """

    system_col = _resolve_system_consumption_col(df, system_col)

    if not 0 <= household_share <= 1:
        raise ValueError("household_share must be between 0 and 1.")

    out = df.copy()

    out["household_load_mwh"] = out[system_col] * household_share
    out["other_system_load_mwh"] = out[system_col] - out["household_load_mwh"]

    return out


def get_central_segment_assumptions() -> Dict[str, Dict[str, float]]:
    """
    Central assumptions for flexible household load segments.

    load_share is stated as share of total household electricity demand.
    'other_flexible' is not the remaining inflexible load. It only represents
    additional shiftable household loads.
    """

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
    """
    Converts the segment assumption dictionary to a readable dataframe.
    """

    table = pd.DataFrame(segment_assumptions).T.reset_index()
    table = table.rename(columns={"index": "segment"})

    table["share_of_household_load_pct"] = table["load_share"] * 100
    table["max_shiftable_share_pct"] = table["max_shiftable_share"] * 100

    return table


def make_load_share_summary(
    segment_assumptions: Mapping[str, Mapping[str, float]]
) -> pd.DataFrame:
    """
    Summarises flexible and inflexible shares of household load.
    """

    flexible_share = sum(
        params["load_share"]
        for params in segment_assumptions.values()
    )

    inflexible_share = 1 - flexible_share

    if inflexible_share < -1e-9:
        raise ValueError(
            "Segment load shares sum to more than 1. "
            f"Total flexible share: {flexible_share:.3f}"
        )

    summary = pd.DataFrame(
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

    return summary


def add_flexible_segment_loads(
    df: pd.DataFrame,
    segment_assumptions: Mapping[str, Mapping[str, float]],
    household_col: str = "household_load_mwh",
) -> pd.DataFrame:
    """
    Adds baseline and flexible load columns for each flexible segment.

    For each segment:
    - {segment}_baseline_mwh = household_load * load_share
    - {segment}_flexible_mwh = segment_baseline * max_shiftable_share
    """

    if household_col not in df.columns:
        raise ValueError(
            f"Expected column '{household_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()

    flexible_household_share = sum(
        params["load_share"]
        for params in segment_assumptions.values()
    )

    if flexible_household_share > 1:
        raise ValueError(
            "Flexible segment shares cannot sum to more than 1. "
            f"Current sum: {flexible_household_share:.3f}"
        )

    out["flexible_household_load_mwh"] = (
        out[household_col] * flexible_household_share
    )
    out["inflexible_household_load_mwh"] = (
        out[household_col] * (1 - flexible_household_share)
    )

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


def _resolve_system_consumption_col(
    df: pd.DataFrame,
    system_col: str = "system_consumption_mwh",
) -> str:
    if system_col in df.columns:
        return system_col

    legacy_col = "consumption_mwh"
    if system_col == "system_consumption_mwh" and legacy_col in df.columns:
        return legacy_col

    raise ValueError(
        f"Expected column '{system_col}' in dataframe. "
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
    system_col = _resolve_system_consumption_col(df, system_col)
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
        [
            date_col,
            "system_consumption_mwh",
            "household_load_mwh",
            "other_system_load_mwh",
        ],
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


def plot_segment_shares(
    segment_table: pd.DataFrame,
) -> None:
    _require_columns(segment_table, ["segment", "share_of_household_load_pct"])

    plt.figure(figsize=(8, 5))
    plt.bar(
        segment_table["segment"],
        segment_table["share_of_household_load_pct"],
    )
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
    baseline_cols = [
        f"{segment}_baseline_mwh"
        for segment in segment_assumptions.keys()
    ]

    _require_columns(df, [date_col] + baseline_cols)

    plt.figure(figsize=(12, 5))

    for segment in segment_assumptions.keys():
        col = f"{segment}_baseline_mwh"
        plt.plot(df[date_col], df[col], label=segment)

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
    plt.plot(
        df[date_col],
        df["total_segment_baseline_mwh"],
        label="Flexible segment baseline",
    )
    plt.plot(
        df[date_col],
        df["total_shiftable_energy_mwh"],
        label="Actually shiftable energy",
    )
    plt.title("From Household Load to Shiftable Energy")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_segment_shift_windows(
    segment_table: pd.DataFrame,
) -> None:
    _require_columns(segment_table, ["segment", "max_wait_h"])

    plt.figure(figsize=(8, 5))
    plt.bar(
        segment_table["segment"],
        segment_table["max_wait_h"],
    )
    plt.title("Maximum Shift Window by Flexible Segment")
    plt.xlabel("Segment")
    plt.ylabel("Maximum wait (hours)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_segment_shiftable_shares(
    segment_table: pd.DataFrame,
) -> None:
    _require_columns(segment_table, ["segment", "max_shiftable_share_pct"])

    plt.figure(figsize=(8, 5))
    plt.bar(
        segment_table["segment"],
        segment_table["max_shiftable_share_pct"],
    )
    plt.title("Maximum Shiftable Share by Flexible Segment")
    plt.xlabel("Segment")
    plt.ylabel("Maximum shiftable share (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
