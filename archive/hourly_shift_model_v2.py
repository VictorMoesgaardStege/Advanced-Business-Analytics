from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRED_PATH = ROOT / "outputs" / "model" / "predictions.parquet"
CONSUMPTION_PATH = ROOT / "data" / "consumption_dk1_raw.csv"

HOUSEHOLD_SHARE_OF_SYSTEM = 0.33
FORECAST_HORIZON_HOURS = 120
ROLLING_BACKTEST_MAX_WINDOWS = 120
ANNUAL_HEATMAP_DAILY_ISSUES = 365
HISTORICAL_CAPACITY_QUANTILE = 0.94
HISTORICAL_CAPACITY_BY_HOUR_OF_DAY = True

SEGMENT_COLORS = {
    "inflexible": "#B0BEC5",
    "wet_loads": "#1E88E5",
    "thermal": "#43A047",
    "ev": "#FB8C00",
}

SEGMENT_ASSUMPTIONS = {
    "inflexible": {
        "share_of_household_load": 0.6,
        "max_shiftable_share": 0.00,
        "max_wait_h": 0,
        "wait_penalty": 0.00,
    },
    "wet_loads": {
        "share_of_household_load": 0.1,
        "max_shiftable_share": 0.35,
        "max_wait_h": 24 * 2,
        "wait_penalty": 0.90,
    },
    "thermal": {
        "share_of_household_load": 0.2,
        "max_shiftable_share": 0.15,
        "max_wait_h": 24 * 1,
        "wait_penalty": 1.20,
    },
    "ev": {
        "share_of_household_load": 0.1,
        "max_shiftable_share": 0.85,
        "max_wait_h": 24 * 5,
        "wait_penalty": 0.35,
    },
}

DECISION_WEIGHTS = {
    "price": 0.80,
    "wait": 0.20,
}


def load_predictions(pred_path: Path = PRED_PATH) -> pd.DataFrame:
    preds = pd.read_parquet(pred_path)
    preds["issue_time"] = pd.to_datetime(preds["issue_time"])
    preds["target_time"] = pd.to_datetime(preds["target_time"])
    preds["fold_end"] = pd.to_datetime(preds["fold_end"])
    return preds.sort_values(["fold", "issue_time", "horizon_h"]).reset_index(drop=True)


def load_system_load(consumption_path: Path = CONSUMPTION_PATH) -> pd.DataFrame:
    consumption = pd.read_csv(consumption_path, parse_dates=["TimeDK"])
    return (
        consumption.groupby(pd.Grouper(key="TimeDK", freq="h"))["ConsumptionkWh"]
        .sum()
        .div(1000.0)
        .rename("system_load_mwh")
        .reset_index()
        .rename(columns={"TimeDK": "target_time"})
    )


def build_issue_catalog(
    preds: pd.DataFrame,
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    coverage = (
        preds.groupby(["fold", "issue_time"], as_index=False)
        .agg(
            horizons=("horizon_h", "nunique"),
            min_h=("horizon_h", "min"),
            max_h=("horizon_h", "max"),
        )
    )
    return coverage[
        (coverage["horizons"] == forecast_horizon_hours)
        & (coverage["min_h"] == 1)
        & (coverage["max_h"] == forecast_horizon_hours)
    ].sort_values("issue_time").reset_index(drop=True)


def build_segment_assumptions(
    shares: dict,
    base_segment_assumptions: dict = SEGMENT_ASSUMPTIONS,
) -> dict:
    assumptions = deepcopy(base_segment_assumptions)
    for segment, share in shares.items():
        assumptions[segment]["share_of_household_load"] = float(share)

    assumptions["inflexible"]["share_of_household_load"] = 1.0 - sum(shares.values())

    total_share = sum(v["share_of_household_load"] for v in assumptions.values())
    if not np.isclose(total_share, 1.0):
        raise ValueError(f"Scenario shares must sum to 1.0, got {total_share:.6f}")

    return assumptions


def build_historical_segment_capacity(
    system_load: pd.DataFrame,
    segment_assumptions: dict,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
    historical_capacity_quantile: float = HISTORICAL_CAPACITY_QUANTILE,
    by_hour_of_day: bool = HISTORICAL_CAPACITY_BY_HOUR_OF_DAY,
) -> pd.DataFrame:
    historical_segment_capacity = system_load[["target_time", "system_load_mwh"]].copy()
    historical_segment_capacity["target_time"] = pd.to_datetime(
        historical_segment_capacity["target_time"]
    )
    historical_segment_capacity["hour_of_day"] = historical_segment_capacity["target_time"].dt.hour
    historical_segment_capacity["household_load_mwh"] = (
        historical_segment_capacity["system_load_mwh"] * household_share_of_system
    )

    for segment, params in segment_assumptions.items():
        segment_cap_col = f"{segment}_historical_cap_mwh"
        historical_segment_capacity[segment_cap_col] = (
            historical_segment_capacity["household_load_mwh"] * params["share_of_household_load"]
        )

    capacity_cols = [f"{segment}_historical_cap_mwh" for segment in segment_assumptions]

    if by_hour_of_day:
        return (
            historical_segment_capacity.groupby("hour_of_day", as_index=False)[capacity_cols]
            .quantile(historical_capacity_quantile)
            .sort_values("hour_of_day")
            .reset_index(drop=True)
        )

    return historical_segment_capacity[capacity_cols].quantile(historical_capacity_quantile).to_frame().T


def _resolve_segment_capacity(
    window_df: pd.DataFrame,
    historical_segment_capacity: pd.DataFrame,
    cap_col: str,
) -> np.ndarray:
    if "hour_of_day" not in historical_segment_capacity.columns:
        segment_capacity_value = float(historical_segment_capacity.iloc[0][cap_col])
        return np.full(len(window_df), segment_capacity_value, dtype=float)

    capacity_by_hour = historical_segment_capacity.set_index("hour_of_day")[cap_col]
    segment_capacity = (
        pd.to_datetime(window_df["target_time"])
        .dt.hour.map(capacity_by_hour)
        .to_numpy(dtype=float)
    )

    if np.isnan(segment_capacity).any():
        fallback_capacity = float(historical_segment_capacity[cap_col].max())
        segment_capacity = np.where(
            np.isnan(segment_capacity),
            fallback_capacity,
            segment_capacity,
        )

    return segment_capacity


def _normalize(series: pd.Series) -> pd.Series:
    s_min = float(series.min())
    s_max = float(series.max())
    if np.isclose(s_min, s_max):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


def assemble_window_for_issue_time(
    issue_time: pd.Timestamp,
    preds: pd.DataFrame,
    system_load: pd.DataFrame,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
    segment_assumptions: dict = SEGMENT_ASSUMPTIONS,
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    issue_time = pd.Timestamp(issue_time)
    forecast_window = (
        preds[preds["issue_time"] == issue_time]
        .sort_values("horizon_h")
        .reset_index(drop=True)
    )
    if len(forecast_window) != forecast_horizon_hours:
        raise ValueError(
            f"Issue time {issue_time} does not have a full {forecast_horizon_hours}h forecast window."
        )

    window_df = forecast_window.merge(system_load, on="target_time", how="left")
    window_df["system_load_mwh"] = window_df["system_load_mwh"].ffill().bfill()
    window_df["household_load_mwh"] = window_df["system_load_mwh"] * household_share_of_system

    for segment, params in segment_assumptions.items():
        base_col = f"{segment}_baseline_mwh"
        flex_col = f"{segment}_flexible_mwh"
        window_df[base_col] = window_df["household_load_mwh"] * params["share_of_household_load"]
        window_df[flex_col] = window_df[base_col] * params["max_shiftable_share"]

    return window_df


def build_actual_baseline_reference(
    issue_times: pd.Series,
    preds: pd.DataFrame,
    system_load: pd.DataFrame,
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
) -> pd.DataFrame:
    reference_profiles = []

    for issue_time in issue_times:
        issue_time = pd.Timestamp(issue_time)
        forecast_window = (
            preds[preds["issue_time"] == issue_time]
            .sort_values("horizon_h")
            .reset_index(drop=True)
        )
        if len(forecast_window) != forecast_horizon_hours:
            raise ValueError(
                f"Issue time {issue_time} does not have a full {forecast_horizon_hours}h forecast window."
            )

        reference_window = forecast_window[["target_time"]].merge(
            system_load,
            on="target_time",
            how="left",
        )
        reference_window["system_load_mwh"] = reference_window["system_load_mwh"].ffill().bfill()
        reference_window["household_load_mwh"] = (
            reference_window["system_load_mwh"] * household_share_of_system
        )
        reference_profiles.append(
            reference_window[["target_time", "system_load_mwh", "household_load_mwh"]]
        )

    return (
        pd.concat(reference_profiles, ignore_index=True)
        .sort_values("target_time")
        .drop_duplicates(subset=["target_time"], keep="first")
        .reset_index(drop=True)
    )


def summarize_actual_baseline(
    reference_df: pd.DataFrame,
    scenario_name: str = "Actual Baseline",
) -> dict:
    baseline_p95 = float(reference_df["system_load_mwh"].quantile(0.95))
    baseline_hours_above_p95 = int((reference_df["system_load_mwh"] > baseline_p95).sum())
    total_system_consumption_mwh = float(reference_df["system_load_mwh"].sum())
    total_household_consumption_mwh = float(reference_df["household_load_mwh"].sum())

    return {
        "scenario": scenario_name,
        "inflexible_share_pct": np.nan,
        "wet_loads_share_pct": np.nan,
        "thermal_share_pct": np.nan,
        "ev_share_pct": np.nan,
        "simulation_hours": int(len(reference_df)),
        "simulation_days": float(len(reference_df) / 24.0),
        "simulation_start": reference_df["target_time"].min(),
        "simulation_end": reference_df["target_time"].max(),
        "shift_events": 0,
        "total_shifted_energy_mwh": 0.0,
        "modeled_total_system_consumption_mwh": total_system_consumption_mwh,
        "modeled_total_household_consumption_mwh": total_household_consumption_mwh,
        "shifted_share_of_system_consumption_pct": 0.0,
        "shifted_share_of_household_consumption_pct": 0.0,
        "realized_household_savings_eur": 0.0,
        "baseline_system_load_p95_mwh": baseline_p95,
        "shifted_system_load_p95_mwh": baseline_p95,
        "baseline_hours_above_p95": baseline_hours_above_p95,
        "shifted_hours_above_p95": baseline_hours_above_p95,
        "p95_delta_mwh": 0.0,
    }


def build_shift_plan(
    window_df: pd.DataFrame,
    segment_assumptions: dict,
    decision_weights: dict,
    historical_segment_capacity: pd.DataFrame,
):
    df = window_df.copy().reset_index(drop=True)
    df["other_system_load_mwh"] = df["system_load_mwh"] - df["household_load_mwh"]

    shift_rows = []
    shifted_cols = []

    for segment, params in segment_assumptions.items():
        base_col = f"{segment}_baseline_mwh"
        flex_col = f"{segment}_flexible_mwh"
        cap_col = f"{segment}_historical_cap_mwh"
        shifted = df[base_col].to_numpy(dtype=float).copy()
        segment_capacity = _resolve_segment_capacity(
            window_df=df,
            historical_segment_capacity=historical_segment_capacity,
            cap_col=cap_col,
        )
        flexible = df[flex_col].to_numpy(dtype=float)

        if params["max_shiftable_share"] == 0 or params["max_wait_h"] == 0:
            df[f"{segment}_shifted_mwh"] = shifted
            shifted_cols.append(f"{segment}_shifted_mwh")
            continue

        for origin in range(len(df)):
            movable = flexible[origin]
            if movable <= 0:
                continue

            max_wait = int(params["max_wait_h"])
            end = min(len(df) - 1, origin + max_wait)
            candidates = df.loc[origin:end, ["target_time", "predicted"]].copy()
            candidates["wait_h"] = np.arange(len(candidates))
            candidates["price_norm"] = _normalize(candidates["predicted"])
            candidates["wait_norm"] = candidates["wait_h"] / max(max_wait, 1)

            candidates["score"] = (
                decision_weights["price"] * candidates["price_norm"]
                + decision_weights["wait"] * params["wait_penalty"] * candidates["wait_norm"]
            )

            origin_score = float(candidates.iloc[0]["score"])
            ranked_candidate_indices = candidates.sort_values("score").index.tolist()
            remaining_movable = movable

            for rank_position, candidate_idx in enumerate(ranked_candidate_indices, start=1):
                candidate_idx = int(candidate_idx)
                if candidate_idx == origin:
                    continue

                candidate_score = float(candidates.loc[candidate_idx, "score"])
                if candidate_score >= origin_score:
                    continue

                available_room = segment_capacity[candidate_idx] - shifted[candidate_idx]
                if available_room <= 0:
                    continue

                allocated_mwh = min(remaining_movable, available_room)
                if allocated_mwh <= 0:
                    continue

                score_gain = origin_score - candidate_score
                projected_target_segment_load = shifted[candidate_idx] + allocated_mwh

                shifted[origin] -= allocated_mwh
                shifted[candidate_idx] += allocated_mwh
                remaining_movable -= allocated_mwh

                shift_rows.append(
                    {
                        "segment": segment,
                        "origin_index": origin,
                        "target_index": candidate_idx,
                        "origin_time": df.loc[origin, "target_time"],
                        "target_time": df.loc[candidate_idx, "target_time"],
                        "wait_h": candidate_idx - origin,
                        "shift_mwh": allocated_mwh,
                        "origin_predicted_price": df.loc[origin, "predicted"],
                        "target_predicted_price": df.loc[candidate_idx, "predicted"],
                        "score_gain": score_gain,
                        "chosen_rank": rank_position,
                        "target_segment_capacity_mwh": segment_capacity[candidate_idx],
                        "available_room_before_allocation_mwh": available_room,
                        "projected_target_segment_load_mwh": projected_target_segment_load,
                        "remaining_origin_movable_mwh": remaining_movable,
                    }
                )

                if remaining_movable <= 1e-9:
                    break

        shifted_col = f"{segment}_shifted_mwh"
        df[shifted_col] = shifted
        shifted_cols.append(shifted_col)

    df["shifted_household_load_mwh"] = df[shifted_cols].sum(axis=1)
    df["shifted_system_load_mwh"] = df["other_system_load_mwh"] + df["shifted_household_load_mwh"]
    df["baseline_household_cost_eur"] = df["household_load_mwh"] * df["DayAheadPriceEUR"]
    df["shifted_household_cost_eur"] = df["shifted_household_load_mwh"] * df["DayAheadPriceEUR"]
    df["baseline_predicted_cost_eur"] = df["household_load_mwh"] * df["predicted"]
    df["shifted_predicted_cost_eur"] = df["shifted_household_load_mwh"] * df["predicted"]

    shifts = pd.DataFrame(
        shift_rows,
        columns=[
            "segment",
            "origin_index",
            "target_index",
            "origin_time",
            "target_time",
            "wait_h",
            "shift_mwh",
            "origin_predicted_price",
            "target_predicted_price",
            "score_gain",
            "chosen_rank",
            "target_segment_capacity_mwh",
            "available_room_before_allocation_mwh",
            "projected_target_segment_load_mwh",
            "remaining_origin_movable_mwh",
        ],
    )
    return df, shifts


def summarize_issue_time_run(
    issue_time: pd.Timestamp,
    preds: pd.DataFrame,
    system_load: pd.DataFrame,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
    segment_assumptions: dict = SEGMENT_ASSUMPTIONS,
    decision_weights: dict = DECISION_WEIGHTS,
    historical_segment_capacity: pd.DataFrame | None = None,
) -> dict:
    if historical_segment_capacity is None:
        historical_segment_capacity = build_historical_segment_capacity(
            system_load=system_load,
            segment_assumptions=segment_assumptions,
            household_share_of_system=household_share_of_system,
        )

    window_df = assemble_window_for_issue_time(
        issue_time=issue_time,
        preds=preds,
        system_load=system_load,
        household_share_of_system=household_share_of_system,
        segment_assumptions=segment_assumptions,
    )
    prototype_df, shifts_df = build_shift_plan(
        window_df,
        segment_assumptions,
        decision_weights,
        historical_segment_capacity,
    )

    shifted_energy_mwh = float(shifts_df["shift_mwh"].sum()) if not shifts_df.empty else 0.0
    total_system_consumption_mwh = float(prototype_df["system_load_mwh"].sum())
    total_household_consumption_mwh = float(prototype_df["household_load_mwh"].sum())

    return {
        "issue_time": pd.Timestamp(issue_time),
        "horizon_hours": int(len(prototype_df)),
        "n_shift_events": int(len(shifts_df)),
        "shifted_energy_mwh": shifted_energy_mwh,
        "modeled_total_system_consumption_mwh": total_system_consumption_mwh,
        "modeled_total_household_consumption_mwh": total_household_consumption_mwh,
        "shifted_share_of_system_consumption_pct": 100.0 * shifted_energy_mwh / total_system_consumption_mwh,
        "shifted_share_of_household_consumption_pct": 100.0 * shifted_energy_mwh / total_household_consumption_mwh,
        "realized_household_savings_eur": float(
            prototype_df["baseline_household_cost_eur"].sum()
            - prototype_df["shifted_household_cost_eur"].sum()
        ),
        "predicted_household_savings_eur": float(
            prototype_df["baseline_predicted_cost_eur"].sum()
            - prototype_df["shifted_predicted_cost_eur"].sum()
        ),
    }


def run_scenario(
    scenario_name: str,
    scenario_shares: dict,
    issue_times: pd.Series,
    preds: pd.DataFrame,
    system_load: pd.DataFrame,
    baseline_reference_p95_mwh: float,
    base_segment_assumptions: dict = SEGMENT_ASSUMPTIONS,
    decision_weights: dict = DECISION_WEIGHTS,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
) -> dict:
    segment_assumptions = build_segment_assumptions(
        shares=scenario_shares,
        base_segment_assumptions=base_segment_assumptions,
    )
    historical_segment_capacity = build_historical_segment_capacity(
        system_load=system_load,
        segment_assumptions=segment_assumptions,
        household_share_of_system=household_share_of_system,
    )

    scenario_profiles = []
    total_shifted_energy_mwh = 0.0
    total_shift_events = 0

    for issue_time in issue_times:
        window_df = assemble_window_for_issue_time(
            issue_time=issue_time,
            preds=preds,
            system_load=system_load,
            household_share_of_system=household_share_of_system,
            segment_assumptions=segment_assumptions,
        )
        profile_df, shifts_df = build_shift_plan(
            window_df=window_df,
            segment_assumptions=segment_assumptions,
            decision_weights=decision_weights,
            historical_segment_capacity=historical_segment_capacity,
        )
        profile_df = profile_df.copy()
        profile_df["scenario"] = scenario_name
        profile_df["issue_time"] = pd.Timestamp(issue_time)
        scenario_profiles.append(profile_df)

        if not shifts_df.empty:
            total_shifted_energy_mwh += float(shifts_df["shift_mwh"].sum())
            total_shift_events += int(len(shifts_df))

    scenario_profiles_df = (
        pd.concat(scenario_profiles, ignore_index=True)
        .sort_values("target_time")
        .drop_duplicates(subset=["target_time"], keep="first")
        .reset_index(drop=True)
    )

    baseline_p95 = float(baseline_reference_p95_mwh)
    shifted_p95 = float(scenario_profiles_df["shifted_system_load_mwh"].quantile(0.95))
    baseline_hours_above_p95 = int((scenario_profiles_df["system_load_mwh"] > baseline_p95).sum())
    shifted_hours_above_p95 = int(
        (scenario_profiles_df["shifted_system_load_mwh"] > baseline_p95).sum()
    )
    total_system_consumption_mwh = float(scenario_profiles_df["system_load_mwh"].sum())
    total_household_consumption_mwh = float(scenario_profiles_df["household_load_mwh"].sum())

    return {
        "scenario": scenario_name,
        "inflexible_share_pct": 100.0 * segment_assumptions["inflexible"]["share_of_household_load"],
        "wet_loads_share_pct": 100.0 * segment_assumptions["wet_loads"]["share_of_household_load"],
        "thermal_share_pct": 100.0 * segment_assumptions["thermal"]["share_of_household_load"],
        "ev_share_pct": 100.0 * segment_assumptions["ev"]["share_of_household_load"],
        "simulation_hours": int(len(scenario_profiles_df)),
        "simulation_days": float(len(scenario_profiles_df) / 24.0),
        "simulation_start": scenario_profiles_df["target_time"].min(),
        "simulation_end": scenario_profiles_df["target_time"].max(),
        "shift_events": total_shift_events,
        "total_shifted_energy_mwh": total_shifted_energy_mwh,
        "modeled_total_system_consumption_mwh": total_system_consumption_mwh,
        "modeled_total_household_consumption_mwh": total_household_consumption_mwh,
        "shifted_share_of_system_consumption_pct": (
            100.0 * total_shifted_energy_mwh / total_system_consumption_mwh
        ),
        "shifted_share_of_household_consumption_pct": (
            100.0 * total_shifted_energy_mwh / total_household_consumption_mwh
        ),
        "realized_household_savings_eur": float(
            scenario_profiles_df["baseline_household_cost_eur"].sum()
            - scenario_profiles_df["shifted_household_cost_eur"].sum()
        ),
        "baseline_system_load_p95_mwh": baseline_p95,
        "shifted_system_load_p95_mwh": shifted_p95,
        "baseline_hours_above_p95": baseline_hours_above_p95,
        "shifted_hours_above_p95": shifted_hours_above_p95,
        "p95_delta_mwh": shifted_p95 - baseline_p95,
    }


__all__ = [
    "ANNUAL_HEATMAP_DAILY_ISSUES",
    "CONSUMPTION_PATH",
    "DECISION_WEIGHTS",
    "FORECAST_HORIZON_HOURS",
    "HISTORICAL_CAPACITY_BY_HOUR_OF_DAY",
    "HISTORICAL_CAPACITY_QUANTILE",
    "HOUSEHOLD_SHARE_OF_SYSTEM",
    "PRED_PATH",
    "ROLLING_BACKTEST_MAX_WINDOWS",
    "ROOT",
    "SEGMENT_ASSUMPTIONS",
    "SEGMENT_COLORS",
    "assemble_window_for_issue_time",
    "build_actual_baseline_reference",
    "build_historical_segment_capacity",
    "build_issue_catalog",
    "build_segment_assumptions",
    "build_shift_plan",
    "load_predictions",
    "load_system_load",
    "run_scenario",
    "summarize_actual_baseline",
    "summarize_issue_time_run",
]
