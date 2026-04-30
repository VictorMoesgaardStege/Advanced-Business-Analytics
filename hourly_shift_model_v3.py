from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
PRED_PATH = ROOT / "outputs" / "model" / "predictions.parquet"
CONSUMPTION_PATH = ROOT / "data" / "consumption_dk1_raw.csv"

HOUSEHOLD_SHARE_OF_SYSTEM = 0.33
FORECAST_HORIZON_HOURS = 120
ROLLING_BACKTEST_MAX_WINDOWS = 120
ANNUAL_HEATMAP_DAILY_ISSUES = 365
HISTORICAL_CAPACITY_QUANTILE = 0.94
HISTORICAL_CAPACITY_BY_HOUR_OF_DAY = True
HISTORICAL_CAPACITY_BY_DAY_TYPE = True
EV_WEEKDAY_WORKING_HOUR_FACTOR = 0.10

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

SHIFT_EVENT_COLUMNS = [
    "segment",
    "obligation_id",
    "decision_issue_time",
    "origin_time",
    "target_time",
    "wait_h",
    "shift_mwh",
    "origin_actual_price",
    "target_actual_price",
    "target_predicted_price",
    "score",
    "chosen_rank",
    "forced_by_deadline",
]


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
            min_target_time=("target_time", "min"),
            max_target_time=("target_time", "max"),
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

    explicit_inflexible_share = shares.get("inflexible")
    flexible_shares = {
        segment: share for segment, share in shares.items() if segment != "inflexible"
    }

    for segment, share in flexible_shares.items():
        assumptions[segment]["share_of_household_load"] = float(share)

    if explicit_inflexible_share is None:
        assumptions["inflexible"]["share_of_household_load"] = 1.0 - sum(
            flexible_shares.values()
        )
    else:
        assumptions["inflexible"]["share_of_household_load"] = float(
            explicit_inflexible_share
        )

    total_share = sum(v["share_of_household_load"] for v in assumptions.values())
    if not np.isclose(total_share, 1.0):
        raise ValueError(f"Scenario shares must sum to 1.0, got {total_share:.6f}")

    return assumptions


def _normalize(series: pd.Series) -> pd.Series:
    s_min = float(series.min())
    s_max = float(series.max())
    if np.isclose(s_min, s_max):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


def _ev_profile_raw_weight(
    target_time: pd.Series,
    low_hour_factor: float = EV_WEEKDAY_WORKING_HOUR_FACTOR,
) -> pd.Series:
    target_time = pd.to_datetime(target_time)
    is_weekday = target_time.dt.dayofweek < 5
    is_working_hour = target_time.dt.hour.between(7, 15)
    weights = np.where(is_weekday & is_working_hour, low_hour_factor, 1.0)
    return pd.Series(weights, index=target_time.index, dtype=float)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    values = pd.Series(values, dtype=float)
    weights = pd.Series(weights, dtype=float).fillna(0.0)
    if np.isclose(float(weights.sum()), 0.0):
        return float(values.mean())
    return float(np.average(values, weights=weights))


def compute_ev_profile_normalizer(
    df: pd.DataFrame,
    low_hour_factor: float = EV_WEEKDAY_WORKING_HOUR_FACTOR,
    time_col: str = "target_time",
    weight_col: str = "household_load_mwh",
) -> float:
    raw_weight = _ev_profile_raw_weight(df[time_col], low_hour_factor)
    weights = df[weight_col] if weight_col in df.columns else pd.Series(1.0, index=df.index)
    normalizer = _weighted_average(raw_weight, weights)
    if normalizer <= 0:
        return 1.0
    return normalizer


def add_segment_baselines(
    df: pd.DataFrame,
    segment_assumptions: dict,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
    ev_low_hour_factor: float = EV_WEEKDAY_WORKING_HOUR_FACTOR,
    ev_profile_normalizer: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["target_time"] = pd.to_datetime(out["target_time"])

    if "household_load_mwh" not in out.columns:
        out["household_load_mwh"] = out["system_load_mwh"] * household_share_of_system

    if "ev" in segment_assumptions:
        if ev_profile_normalizer is None:
            ev_profile_normalizer = compute_ev_profile_normalizer(
                out,
                low_hour_factor=ev_low_hour_factor,
            )
        out["ev_profile_weight"] = _ev_profile_raw_weight(
            out["target_time"],
            low_hour_factor=ev_low_hour_factor,
        )
        out["ev_profile_multiplier"] = out["ev_profile_weight"] / ev_profile_normalizer

    profiled_segment_cols = []
    for segment, params in segment_assumptions.items():
        if segment == "inflexible":
            continue

        base_col = f"{segment}_baseline_mwh"
        if segment == "ev":
            out[base_col] = (
                out["household_load_mwh"]
                * params["share_of_household_load"]
                * out["ev_profile_multiplier"]
            )
        else:
            out[base_col] = (
                out["household_load_mwh"] * params["share_of_household_load"]
            )
        profiled_segment_cols.append(base_col)

    if "inflexible" in segment_assumptions:
        inflexible_col = "inflexible_baseline_mwh"
        if profiled_segment_cols:
            out[inflexible_col] = out["household_load_mwh"] - out[
                profiled_segment_cols
            ].sum(axis=1)
        else:
            out[inflexible_col] = out["household_load_mwh"]

        min_inflexible_load = float(out[inflexible_col].min())
        if min_inflexible_load < -1e-9:
            raise ValueError(
                "The EV time profile makes inferred inflexible load negative. "
                "Lower the EV share or reduce the EV profile contrast."
            )
        out[inflexible_col] = out[inflexible_col].clip(lower=0.0)

    for segment, params in segment_assumptions.items():
        base_col = f"{segment}_baseline_mwh"
        flex_col = f"{segment}_flexible_mwh"
        nonflex_col = f"{segment}_nonflexible_mwh"
        out[flex_col] = out[base_col] * params["max_shiftable_share"]
        out[nonflex_col] = out[base_col] - out[flex_col]

    return out


def build_historical_segment_capacity(
    system_load: pd.DataFrame,
    segment_assumptions: dict,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
    historical_capacity_quantile: float = HISTORICAL_CAPACITY_QUANTILE,
    by_hour_of_day: bool = HISTORICAL_CAPACITY_BY_HOUR_OF_DAY,
    by_day_type: bool = HISTORICAL_CAPACITY_BY_DAY_TYPE,
    ev_low_hour_factor: float = EV_WEEKDAY_WORKING_HOUR_FACTOR,
) -> pd.DataFrame:
    historical_segment_capacity = system_load[["target_time", "system_load_mwh"]].copy()
    historical_segment_capacity["target_time"] = pd.to_datetime(
        historical_segment_capacity["target_time"]
    )
    historical_segment_capacity["hour_of_day"] = historical_segment_capacity[
        "target_time"
    ].dt.hour
    historical_segment_capacity["is_weekend"] = (
        historical_segment_capacity["target_time"].dt.dayofweek >= 5
    )
    historical_segment_capacity["household_load_mwh"] = (
        historical_segment_capacity["system_load_mwh"] * household_share_of_system
    )

    ev_profile_normalizer = compute_ev_profile_normalizer(
        historical_segment_capacity,
        low_hour_factor=ev_low_hour_factor,
    )
    historical_segment_capacity = add_segment_baselines(
        historical_segment_capacity,
        segment_assumptions=segment_assumptions,
        household_share_of_system=household_share_of_system,
        ev_low_hour_factor=ev_low_hour_factor,
        ev_profile_normalizer=ev_profile_normalizer,
    )

    capacity_cols = []
    for segment in segment_assumptions:
        cap_col = f"{segment}_historical_cap_mwh"
        historical_segment_capacity[cap_col] = historical_segment_capacity[
            f"{segment}_baseline_mwh"
        ]
        capacity_cols.append(cap_col)

    if by_hour_of_day:
        group_cols = ["hour_of_day"]
        if by_day_type:
            group_cols = ["is_weekend", "hour_of_day"]
        return (
            historical_segment_capacity.groupby(group_cols, as_index=False)[capacity_cols]
            .quantile(historical_capacity_quantile)
            .sort_values(group_cols)
            .reset_index(drop=True)
        )

    return (
        historical_segment_capacity[capacity_cols]
        .quantile(historical_capacity_quantile)
        .to_frame()
        .T
    )


def _resolve_segment_capacity(
    window_df: pd.DataFrame,
    historical_segment_capacity: pd.DataFrame,
    cap_col: str,
) -> np.ndarray:
    if "hour_of_day" not in historical_segment_capacity.columns:
        segment_capacity_value = float(historical_segment_capacity.iloc[0][cap_col])
        return np.full(len(window_df), segment_capacity_value, dtype=float)

    target_time = pd.to_datetime(window_df["target_time"])

    if "is_weekend" in historical_segment_capacity.columns:
        capacity_by_slot = historical_segment_capacity.set_index(
            ["is_weekend", "hour_of_day"]
        )[cap_col]
        keys = list(zip(target_time.dt.dayofweek >= 5, target_time.dt.hour))
        segment_capacity = np.array(
            [capacity_by_slot.get(key, np.nan) for key in keys],
            dtype=float,
        )
    else:
        capacity_by_hour = historical_segment_capacity.set_index("hour_of_day")[cap_col]
        segment_capacity = target_time.dt.hour.map(capacity_by_hour).to_numpy(dtype=float)

    if np.isnan(segment_capacity).any():
        fallback_capacity = float(historical_segment_capacity[cap_col].max())
        segment_capacity = np.where(
            np.isnan(segment_capacity),
            fallback_capacity,
            segment_capacity,
        )

    return segment_capacity


def assemble_window_for_issue_time(
    issue_time: pd.Timestamp,
    preds: pd.DataFrame,
    system_load: pd.DataFrame,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
    segment_assumptions: dict = SEGMENT_ASSUMPTIONS,
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS,
    ev_low_hour_factor: float = EV_WEEKDAY_WORKING_HOUR_FACTOR,
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

    return add_segment_baselines(
        window_df,
        segment_assumptions=segment_assumptions,
        household_share_of_system=household_share_of_system,
        ev_low_hour_factor=ev_low_hour_factor,
    )


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


def _build_actual_price_reference(preds: pd.DataFrame) -> pd.DataFrame:
    if "DayAheadPriceEUR" not in preds.columns:
        raise ValueError("preds must contain a DayAheadPriceEUR column.")

    return (
        preds[["target_time", "DayAheadPriceEUR"]]
        .dropna(subset=["DayAheadPriceEUR"])
        .sort_values("target_time")
        .drop_duplicates(subset=["target_time"], keep="first")
        .reset_index(drop=True)
    )


def _prepare_allowed_forecasts(
    issue_times: pd.Series,
    preds: pd.DataFrame,
    forecast_horizon_hours: int,
) -> pd.DataFrame:
    issue_times = pd.Series(pd.to_datetime(issue_times)).drop_duplicates()
    allowed = preds[preds["issue_time"].isin(issue_times)].copy()
    if allowed.empty:
        raise ValueError("No predictions found for the provided issue_times.")

    coverage = (
        allowed.groupby("issue_time", as_index=False)
        .agg(
            horizons=("horizon_h", "nunique"),
            min_h=("horizon_h", "min"),
            max_h=("horizon_h", "max"),
            min_target_time=("target_time", "min"),
            max_target_time=("target_time", "max"),
        )
        .sort_values("issue_time")
        .reset_index(drop=True)
    )
    full_coverage = coverage[
        (coverage["horizons"] == forecast_horizon_hours)
        & (coverage["min_h"] == 1)
        & (coverage["max_h"] == forecast_horizon_hours)
    ]
    if full_coverage.empty:
        raise ValueError(
            "None of the provided issue_times has a full "
            f"{forecast_horizon_hours}h forecast window."
        )

    return allowed[allowed["issue_time"].isin(full_coverage["issue_time"])].copy()


def _build_simulation_timeline(
    allowed_preds: pd.DataFrame,
    system_load: pd.DataFrame,
    preds: pd.DataFrame,
    household_share_of_system: float,
    segment_assumptions: dict,
    ev_low_hour_factor: float,
    simulation_start: pd.Timestamp | None = None,
    simulation_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, float]:
    target_times = (
        allowed_preds["target_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if simulation_start is not None:
        target_times = target_times[target_times >= pd.Timestamp(simulation_start)]
    if simulation_end is not None:
        target_times = target_times[target_times <= pd.Timestamp(simulation_end)]

    if target_times.empty:
        raise ValueError("The selected forecasts do not cover any simulation hours.")

    actual_prices = _build_actual_price_reference(preds)
    timeline = pd.DataFrame({"target_time": target_times})
    timeline = timeline.merge(system_load, on="target_time", how="left")
    timeline = timeline.merge(actual_prices, on="target_time", how="left")
    timeline["system_load_mwh"] = timeline["system_load_mwh"].ffill().bfill()
    timeline["DayAheadPriceEUR"] = timeline["DayAheadPriceEUR"].ffill().bfill()
    timeline["household_load_mwh"] = (
        timeline["system_load_mwh"] * household_share_of_system
    )
    timeline["other_system_load_mwh"] = (
        timeline["system_load_mwh"] - timeline["household_load_mwh"]
    )

    ev_profile_normalizer = compute_ev_profile_normalizer(
        timeline,
        low_hour_factor=ev_low_hour_factor,
    )
    timeline = add_segment_baselines(
        timeline,
        segment_assumptions=segment_assumptions,
        household_share_of_system=household_share_of_system,
        ev_low_hour_factor=ev_low_hour_factor,
        ev_profile_normalizer=ev_profile_normalizer,
    )

    return timeline.reset_index(drop=True), ev_profile_normalizer


def _max_shiftable_wait_h(segment_assumptions: dict) -> int:
    waits = [
        int(params["max_wait_h"])
        for params in segment_assumptions.values()
        if params["max_shiftable_share"] > 0 and params["max_wait_h"] > 0
    ]
    return max(waits) if waits else 0


def _infer_flexible_origin_end(
    issue_spans: pd.DataFrame,
    timeline_df: pd.DataFrame,
    segment_assumptions: dict,
) -> pd.Timestamp:
    issue_times = pd.to_datetime(issue_spans["issue_time"]).sort_values()
    max_wait_h = _max_shiftable_wait_h(segment_assumptions)
    latest_settled_origin_time = (
        pd.Timestamp(timeline_df["target_time"].max()) - pd.Timedelta(hours=max_wait_h)
    )

    if len(issue_times) > 1:
        issue_step_h = (
            issue_times.diff().dropna().median() / pd.Timedelta(hours=1)
        )
        issue_step_h = max(int(round(float(issue_step_h))), 1)
        nominal_origin_end = pd.Timestamp(issue_times.max()) + pd.Timedelta(
            hours=issue_step_h - 1
        )
    else:
        nominal_origin_end = pd.Timestamp(issue_times.max())

    return min(nominal_origin_end, latest_settled_origin_time)


def _select_latest_issue_time(
    current_time: pd.Timestamp,
    issue_spans: pd.DataFrame,
) -> pd.Timestamp:
    eligible = issue_spans[
        (issue_spans["issue_time"] <= current_time)
        & (issue_spans["max_target_time"] >= current_time)
    ]
    if eligible.empty:
        raise ValueError(f"No available forecast covers simulation hour {current_time}.")
    return pd.Timestamp(eligible.iloc[-1]["issue_time"])


def _forecast_window_for_time(
    current_time: pd.Timestamp,
    issue_time: pd.Timestamp,
    forecast_by_issue: dict[pd.Timestamp, pd.DataFrame],
    current_actual_price: float,
) -> pd.DataFrame:
    forecast_window = forecast_by_issue[issue_time]
    forecast_window = forecast_window[
        forecast_window["target_time"] >= current_time
    ][["target_time", "predicted"]].copy()

    if forecast_window.empty or not (forecast_window["target_time"] == current_time).any():
        current_row = pd.DataFrame(
            {
                "target_time": [current_time],
                "predicted": [current_actual_price],
            }
        )
        forecast_window = pd.concat([current_row, forecast_window], ignore_index=True)

    return (
        forecast_window.sort_values("target_time")
        .drop_duplicates(subset=["target_time"], keep="first")
        .reset_index(drop=True)
    )


def _build_future_baseline_window(
    forecast_window: pd.DataFrame,
    system_load: pd.DataFrame,
    preds: pd.DataFrame,
    segment_assumptions: dict,
    household_share_of_system: float,
    ev_low_hour_factor: float,
    ev_profile_normalizer: float,
) -> pd.DataFrame:
    actual_prices = _build_actual_price_reference(preds)
    future_df = forecast_window[["target_time"]].copy()
    future_df = future_df.merge(system_load, on="target_time", how="left")
    future_df = future_df.merge(actual_prices, on="target_time", how="left")
    future_df = future_df.merge(forecast_window, on="target_time", how="left")
    future_df["system_load_mwh"] = future_df["system_load_mwh"].ffill().bfill()
    future_df["DayAheadPriceEUR"] = future_df["DayAheadPriceEUR"].ffill().bfill()
    future_df["household_load_mwh"] = (
        future_df["system_load_mwh"] * household_share_of_system
    )
    future_df["other_system_load_mwh"] = (
        future_df["system_load_mwh"] - future_df["household_load_mwh"]
    )
    return add_segment_baselines(
        future_df,
        segment_assumptions=segment_assumptions,
        household_share_of_system=household_share_of_system,
        ev_low_hour_factor=ev_low_hour_factor,
        ev_profile_normalizer=ev_profile_normalizer,
    )


def _candidate_scores(
    pending_item: dict,
    candidates: pd.DataFrame,
    decision_weights: dict,
) -> pd.DataFrame:
    scored = candidates[["target_time", "predicted"]].copy()
    scored["price_norm"] = _normalize(scored["predicted"])
    wait_h = (
        (pd.to_datetime(scored["target_time"]) - pending_item["origin_time"])
        / pd.Timedelta(hours=1)
    ).astype(float)
    scored["wait_h"] = wait_h
    scored["wait_norm"] = scored["wait_h"] / max(float(pending_item["max_wait_h"]), 1.0)
    scored["score"] = (
        decision_weights["price"] * scored["price_norm"]
        + decision_weights["wait"]
        * pending_item["wait_penalty"]
        * scored["wait_norm"]
    )
    return scored.sort_values("score").reset_index(drop=True)


def _schedule_pending_loads(
    current_time: pd.Timestamp,
    decision_issue_time: pd.Timestamp,
    pending_loads: list[dict],
    forecast_window: pd.DataFrame,
    future_baseline_df: pd.DataFrame,
    segment_assumptions: dict,
    decision_weights: dict,
    historical_segment_capacity: pd.DataFrame,
) -> list[dict]:
    plans = []
    target_times = pd.to_datetime(forecast_window["target_time"])
    current_time = pd.Timestamp(current_time)

    for segment, params in segment_assumptions.items():
        if params["max_shiftable_share"] == 0 or params["max_wait_h"] == 0:
            continue

        segment_pending = [
            item
            for item in pending_loads
            if item["segment"] == segment and item["remaining_mwh"] > 1e-9
        ]
        if not segment_pending:
            continue

        cap_col = f"{segment}_historical_cap_mwh"
        segment_capacity = _resolve_segment_capacity(
            window_df=future_baseline_df,
            historical_segment_capacity=historical_segment_capacity,
            cap_col=cap_col,
        )
        capacity_by_time = dict(zip(future_baseline_df["target_time"], segment_capacity))

        reserved_by_time = {}
        for _, future_row in future_baseline_df.iterrows():
            target_time = pd.Timestamp(future_row["target_time"])
            if target_time == current_time:
                reserved_by_time[target_time] = float(
                    future_row[f"{segment}_nonflexible_mwh"]
                )
            else:
                reserved_by_time[target_time] = float(
                    future_row[f"{segment}_baseline_mwh"]
                )

        planned_by_time = {pd.Timestamp(t): 0.0 for t in future_baseline_df["target_time"]}

        segment_pending = sorted(
            segment_pending,
            key=lambda item: (item["deadline_time"], item["origin_time"], item["id"]),
        )

        for item in segment_pending:
            deadline_time = pd.Timestamp(item["deadline_time"])
            candidate_mask = (target_times >= current_time) & (target_times <= deadline_time)
            candidates = forecast_window.loc[candidate_mask].copy()

            if candidates.empty:
                candidates = pd.DataFrame(
                    {
                        "target_time": [current_time],
                        "predicted": [forecast_window.iloc[0]["predicted"]],
                    }
                )

            if deadline_time <= current_time:
                candidates = candidates[candidates["target_time"] == current_time]
                if candidates.empty:
                    candidates = pd.DataFrame(
                        {
                            "target_time": [current_time],
                            "predicted": [forecast_window.iloc[0]["predicted"]],
                        }
                    )

            scored_candidates = _candidate_scores(
                pending_item=item,
                candidates=candidates,
                decision_weights=decision_weights,
            )

            remaining_to_plan = float(item["remaining_mwh"])
            for rank_position, candidate in enumerate(
                scored_candidates.itertuples(index=False),
                start=1,
            ):
                target_time = pd.Timestamp(candidate.target_time)
                capacity = float(capacity_by_time.get(target_time, np.inf))
                reserved = float(reserved_by_time.get(target_time, 0.0))
                planned = float(planned_by_time.get(target_time, 0.0))
                available_room = capacity - reserved - planned
                forced_by_deadline = deadline_time <= current_time and target_time == current_time

                if forced_by_deadline:
                    allocated_mwh = remaining_to_plan
                else:
                    if available_room <= 1e-9:
                        continue
                    allocated_mwh = min(remaining_to_plan, available_room)

                if allocated_mwh <= 1e-9:
                    continue

                planned_by_time[target_time] = planned_by_time.get(target_time, 0.0) + allocated_mwh
                plans.append(
                    {
                        "segment": segment,
                        "obligation_id": item["id"],
                        "decision_issue_time": decision_issue_time,
                        "origin_time": item["origin_time"],
                        "target_time": target_time,
                        "wait_h": float(candidate.wait_h),
                        "shift_mwh": float(allocated_mwh),
                        "origin_actual_price": item["origin_actual_price"],
                        "target_predicted_price": float(candidate.predicted),
                        "score": float(candidate.score),
                        "chosen_rank": int(rank_position),
                        "forced_by_deadline": bool(forced_by_deadline),
                    }
                )

                remaining_to_plan -= allocated_mwh
                if remaining_to_plan <= 1e-9:
                    break

    return [plan for plan in plans if pd.Timestamp(plan["target_time"]) == current_time]


def _add_new_pending_loads(
    pending_loads: list[dict],
    current_row: pd.Series,
    segment_assumptions: dict,
    current_predicted_price: float,
    next_obligation_id: int,
) -> int:
    current_time = pd.Timestamp(current_row["target_time"])

    for segment, params in segment_assumptions.items():
        if params["max_shiftable_share"] == 0 or params["max_wait_h"] == 0:
            continue

        flexible_mwh = float(current_row[f"{segment}_flexible_mwh"])
        if flexible_mwh <= 1e-9:
            continue

        max_wait_h = int(params["max_wait_h"])
        pending_loads.append(
            {
                "id": next_obligation_id,
                "segment": segment,
                "origin_time": current_time,
                "deadline_time": current_time + pd.Timedelta(hours=max_wait_h),
                "max_wait_h": max_wait_h,
                "wait_penalty": float(params["wait_penalty"]),
                "remaining_mwh": flexible_mwh,
                "origin_actual_price": float(current_row["DayAheadPriceEUR"]),
                "origin_predicted_price": float(current_predicted_price),
            }
        )
        next_obligation_id += 1

    return next_obligation_id


def _commit_current_allocations(
    pending_loads: list[dict],
    current_allocations: list[dict],
) -> None:
    pending_by_id = {item["id"]: item for item in pending_loads}
    for allocation in current_allocations:
        item = pending_by_id[allocation["obligation_id"]]
        item["remaining_mwh"] -= float(allocation["shift_mwh"])

    pending_loads[:] = [
        item for item in pending_loads if item["remaining_mwh"] > 1e-9
    ]


def simulate_stateful_scenario(
    scenario_name: str,
    scenario_shares: dict,
    issue_times: pd.Series,
    preds: pd.DataFrame,
    system_load: pd.DataFrame,
    base_segment_assumptions: dict = SEGMENT_ASSUMPTIONS,
    decision_weights: dict = DECISION_WEIGHTS,
    household_share_of_system: float = HOUSEHOLD_SHARE_OF_SYSTEM,
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS,
    ev_low_hour_factor: float = EV_WEEKDAY_WORKING_HOUR_FACTOR,
    simulation_start: pd.Timestamp | None = None,
    simulation_end: pd.Timestamp | None = None,
    flexible_origin_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_assumptions = build_segment_assumptions(
        shares=scenario_shares,
        base_segment_assumptions=base_segment_assumptions,
    )
    preds = preds.copy()
    preds["issue_time"] = pd.to_datetime(preds["issue_time"])
    preds["target_time"] = pd.to_datetime(preds["target_time"])
    system_load = system_load.copy()
    system_load["target_time"] = pd.to_datetime(system_load["target_time"])

    allowed_preds = _prepare_allowed_forecasts(
        issue_times=issue_times,
        preds=preds,
        forecast_horizon_hours=forecast_horizon_hours,
    )
    issue_spans = (
        allowed_preds.groupby("issue_time", as_index=False)
        .agg(
            min_target_time=("target_time", "min"),
            max_target_time=("target_time", "max"),
        )
        .sort_values("issue_time")
        .reset_index(drop=True)
    )
    forecast_by_issue = {
        pd.Timestamp(issue_time): issue_df.sort_values("target_time").reset_index(drop=True)
        for issue_time, issue_df in allowed_preds.groupby("issue_time")
    }

    historical_segment_capacity = build_historical_segment_capacity(
        system_load=system_load,
        segment_assumptions=segment_assumptions,
        household_share_of_system=household_share_of_system,
        ev_low_hour_factor=ev_low_hour_factor,
    )
    timeline_df, ev_profile_normalizer = _build_simulation_timeline(
        allowed_preds=allowed_preds,
        system_load=system_load,
        preds=preds,
        household_share_of_system=household_share_of_system,
        segment_assumptions=segment_assumptions,
        ev_low_hour_factor=ev_low_hour_factor,
        simulation_start=simulation_start,
        simulation_end=simulation_end,
    )
    if flexible_origin_end is None:
        flexible_origin_end = _infer_flexible_origin_end(
            issue_spans=issue_spans,
            timeline_df=timeline_df,
            segment_assumptions=segment_assumptions,
        )
    flexible_origin_end = pd.Timestamp(flexible_origin_end)

    pending_loads = []
    next_obligation_id = 1
    profile_rows = []
    realized_shift_rows = []

    for _, current_row in timeline_df.iterrows():
        current_time = pd.Timestamp(current_row["target_time"])
        decision_issue_time = _select_latest_issue_time(
            current_time=current_time,
            issue_spans=issue_spans,
        )
        forecast_window = _forecast_window_for_time(
            current_time=current_time,
            issue_time=decision_issue_time,
            forecast_by_issue=forecast_by_issue,
            current_actual_price=float(current_row["DayAheadPriceEUR"]),
        )
        current_predicted_price = float(forecast_window.iloc[0]["predicted"])
        flexible_origin_window_active = current_time <= flexible_origin_end

        local_flexible_consumed_by_segment = {
            segment: 0.0 for segment in segment_assumptions
        }
        if flexible_origin_window_active:
            next_obligation_id = _add_new_pending_loads(
                pending_loads=pending_loads,
                current_row=current_row,
                segment_assumptions=segment_assumptions,
                current_predicted_price=current_predicted_price,
                next_obligation_id=next_obligation_id,
            )
        else:
            for segment, params in segment_assumptions.items():
                if params["max_shiftable_share"] > 0 and params["max_wait_h"] > 0:
                    local_flexible_consumed_by_segment[segment] = float(
                        current_row[f"{segment}_flexible_mwh"]
                    )

        future_baseline_df = _build_future_baseline_window(
            forecast_window=forecast_window,
            system_load=system_load,
            preds=preds,
            segment_assumptions=segment_assumptions,
            household_share_of_system=household_share_of_system,
            ev_low_hour_factor=ev_low_hour_factor,
            ev_profile_normalizer=ev_profile_normalizer,
        )
        current_allocations = _schedule_pending_loads(
            current_time=current_time,
            decision_issue_time=decision_issue_time,
            pending_loads=pending_loads,
            forecast_window=forecast_window,
            future_baseline_df=future_baseline_df,
            segment_assumptions=segment_assumptions,
            decision_weights=decision_weights,
            historical_segment_capacity=historical_segment_capacity,
        )
        _commit_current_allocations(
            pending_loads=pending_loads,
            current_allocations=current_allocations,
        )

        allocation_by_segment = {
            segment: 0.0 for segment in segment_assumptions
        }
        for allocation in current_allocations:
            allocation_by_segment[allocation["segment"]] += float(allocation["shift_mwh"])
            if abs(float(allocation["wait_h"])) > 1e-9:
                realized_shift_rows.append(allocation)

        profile_row = current_row.to_dict()
        profile_row["scenario"] = scenario_name
        profile_row["decision_issue_time"] = decision_issue_time
        profile_row["current_predicted_price"] = current_predicted_price
        profile_row["flexible_origin_window_active"] = flexible_origin_window_active

        shifted_segment_cols = []
        for segment, params in segment_assumptions.items():
            shifted_col = f"{segment}_shifted_mwh"
            realized_flex_col = f"{segment}_realized_flexible_mwh"

            if params["max_shiftable_share"] == 0 or params["max_wait_h"] == 0:
                profile_row[realized_flex_col] = 0.0
                profile_row[shifted_col] = float(profile_row[f"{segment}_baseline_mwh"])
            else:
                profile_row[realized_flex_col] = (
                    local_flexible_consumed_by_segment[segment]
                    + allocation_by_segment[segment]
                )
                profile_row[shifted_col] = (
                    float(profile_row[f"{segment}_nonflexible_mwh"])
                    + local_flexible_consumed_by_segment[segment]
                    + allocation_by_segment[segment]
                )
            shifted_segment_cols.append(shifted_col)

        profile_row["shifted_household_load_mwh"] = sum(
            float(profile_row[col]) for col in shifted_segment_cols
        )
        profile_row["shifted_system_load_mwh"] = (
            float(profile_row["other_system_load_mwh"])
            + profile_row["shifted_household_load_mwh"]
        )
        profile_row["baseline_household_cost_eur"] = (
            float(profile_row["household_load_mwh"])
            * float(profile_row["DayAheadPriceEUR"])
        )
        profile_row["shifted_household_cost_eur"] = (
            profile_row["shifted_household_load_mwh"]
            * float(profile_row["DayAheadPriceEUR"])
        )
        profile_row["pending_flexible_mwh_after_commit"] = sum(
            float(item["remaining_mwh"]) for item in pending_loads
        )
        profile_rows.append(profile_row)

    profiles_df = pd.DataFrame(profile_rows)
    shifts_df = pd.DataFrame(realized_shift_rows, columns=SHIFT_EVENT_COLUMNS)
    if not shifts_df.empty:
        actual_price_by_time = profiles_df.set_index("target_time")["DayAheadPriceEUR"]
        shifts_df["target_actual_price"] = shifts_df["target_time"].map(actual_price_by_time)

    return profiles_df, shifts_df


def summarize_stateful_scenario(
    scenario_name: str,
    scenario_profiles_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    segment_assumptions: dict,
    baseline_reference_p95_mwh: float,
) -> dict:
    shifted_energy_mwh = float(shifts_df["shift_mwh"].sum()) if not shifts_df.empty else 0.0
    baseline_p95 = float(baseline_reference_p95_mwh)
    shifted_p95 = float(scenario_profiles_df["shifted_system_load_mwh"].quantile(0.95))
    baseline_hours_above_p95 = int(
        (scenario_profiles_df["system_load_mwh"] > baseline_p95).sum()
    )
    shifted_hours_above_p95 = int(
        (scenario_profiles_df["shifted_system_load_mwh"] > baseline_p95).sum()
    )
    total_system_consumption_mwh = float(scenario_profiles_df["system_load_mwh"].sum())
    total_household_consumption_mwh = float(
        scenario_profiles_df["household_load_mwh"].sum()
    )

    return {
        "scenario": scenario_name,
        "inflexible_share_pct": 100.0
        * segment_assumptions["inflexible"]["share_of_household_load"],
        "wet_loads_share_pct": 100.0
        * segment_assumptions["wet_loads"]["share_of_household_load"],
        "thermal_share_pct": 100.0
        * segment_assumptions["thermal"]["share_of_household_load"],
        "ev_share_pct": 100.0 * segment_assumptions["ev"]["share_of_household_load"],
        "simulation_hours": int(len(scenario_profiles_df)),
        "simulation_days": float(len(scenario_profiles_df) / 24.0),
        "simulation_start": scenario_profiles_df["target_time"].min(),
        "simulation_end": scenario_profiles_df["target_time"].max(),
        "shift_events": int(len(shifts_df)),
        "total_shifted_energy_mwh": shifted_energy_mwh,
        "modeled_total_system_consumption_mwh": total_system_consumption_mwh,
        "modeled_total_household_consumption_mwh": total_household_consumption_mwh,
        "shifted_share_of_system_consumption_pct": (
            100.0 * shifted_energy_mwh / total_system_consumption_mwh
        ),
        "shifted_share_of_household_consumption_pct": (
            100.0 * shifted_energy_mwh / total_household_consumption_mwh
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
        "pending_flexible_mwh_end": float(
            scenario_profiles_df["pending_flexible_mwh_after_commit"].iloc[-1]
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
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS,
    ev_low_hour_factor: float = EV_WEEKDAY_WORKING_HOUR_FACTOR,
    simulation_start: pd.Timestamp | None = None,
    simulation_end: pd.Timestamp | None = None,
    flexible_origin_end: pd.Timestamp | None = None,
    return_profiles: bool = False,
) -> dict | tuple[dict, pd.DataFrame, pd.DataFrame]:
    segment_assumptions = build_segment_assumptions(
        shares=scenario_shares,
        base_segment_assumptions=base_segment_assumptions,
    )
    scenario_profiles_df, shifts_df = simulate_stateful_scenario(
        scenario_name=scenario_name,
        scenario_shares=scenario_shares,
        issue_times=issue_times,
        preds=preds,
        system_load=system_load,
        base_segment_assumptions=base_segment_assumptions,
        decision_weights=decision_weights,
        household_share_of_system=household_share_of_system,
        forecast_horizon_hours=forecast_horizon_hours,
        ev_low_hour_factor=ev_low_hour_factor,
        simulation_start=simulation_start,
        simulation_end=simulation_end,
        flexible_origin_end=flexible_origin_end,
    )
    summary = summarize_stateful_scenario(
        scenario_name=scenario_name,
        scenario_profiles_df=scenario_profiles_df,
        shifts_df=shifts_df,
        segment_assumptions=segment_assumptions,
        baseline_reference_p95_mwh=baseline_reference_p95_mwh,
    )

    if return_profiles:
        return summary, scenario_profiles_df, shifts_df

    return summary


__all__ = [
    "ANNUAL_HEATMAP_DAILY_ISSUES",
    "CONSUMPTION_PATH",
    "DECISION_WEIGHTS",
    "EV_WEEKDAY_WORKING_HOUR_FACTOR",
    "FORECAST_HORIZON_HOURS",
    "HISTORICAL_CAPACITY_BY_DAY_TYPE",
    "HISTORICAL_CAPACITY_BY_HOUR_OF_DAY",
    "HISTORICAL_CAPACITY_QUANTILE",
    "HOUSEHOLD_SHARE_OF_SYSTEM",
    "PRED_PATH",
    "ROLLING_BACKTEST_MAX_WINDOWS",
    "ROOT",
    "SEGMENT_ASSUMPTIONS",
    "SEGMENT_COLORS",
    "SHIFT_EVENT_COLUMNS",
    "add_segment_baselines",
    "assemble_window_for_issue_time",
    "build_actual_baseline_reference",
    "build_historical_segment_capacity",
    "build_issue_catalog",
    "build_segment_assumptions",
    "compute_ev_profile_normalizer",
    "load_predictions",
    "load_system_load",
    "run_scenario",
    "simulate_stateful_scenario",
    "summarize_actual_baseline",
    "summarize_stateful_scenario",
]
