"""
Electricity Price Forecasting App — Resilience Impact Simulation v2
====================================================================
Simulates how much electricity demand could be shifted when users see
5-day-ahead forecasted prices. Flexible load shifts to cheaper future
periods, weighted by price attractiveness and waiting aversion.

Run directly:  python sim_2.py

Uses the same real DK1 data files as simulate_impact.py:
  - data/consumption_dk1_raw.csv      (hourly, multi-area rows → summed)
  - data/day_ahead_prices_dk1_raw.csv (15-min sub-hourly → averaged to hourly)
Falls back to synthetic data if files are missing.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from pathlib import Path

# Force UTF-8 output on Windows so Unicode in print() doesn't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
np.random.seed(42)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

SIMULATION_DAYS   = 45           # how many hours of the available data to use
FORECAST_HORIZON  = 5 * 24      # hours the app looks ahead (5 days)
STRESS_PERCENTILE = 95           # top X% prices define "stress hours"
OUTPUT_DIR        = Path("simulation_outputs")

# Paths to the same raw data files used by simulate_impact.py
_DATA_DIR         = Path(__file__).resolve().parents[2] / "data"
CONSUMPTION_CSV   = _DATA_DIR / "consumption_dk1_raw.csv"
PRICES_CSV        = _DATA_DIR / "day_ahead_prices_dk1_raw.csv"


# -----------------------------------------------------------------------------
# 1. DATA LOADING  (same sources as simulate_impact.py, hourly resolution)
# -----------------------------------------------------------------------------

def load_real_prices(n_hours: int) -> pd.Series:
    """
    Reads day-ahead prices from PRICES_CSV, resamples 15-min → hourly mean,
    and returns the last n_hours as a Series indexed by timestamp (DKK/MWh).
    """
    df = pd.read_csv(PRICES_CSV, parse_dates=["TimeDK"])
    df = df.sort_values("TimeDK")
    hourly = (
        df.set_index("TimeDK")["DayAheadPriceDKK"]
        .resample("h")
        .mean()
        .dropna()
    )
    hourly = hourly.tail(n_hours)
    hourly.name = "actual_price_dkk_mwh"
    return hourly


def load_real_baseline_load(timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Reads DK1 consumption from CONSUMPTION_CSV, sums all grid areas per hour,
    converts kWh → MW (1 kWh per hour = 0.001 MW), and aligns to timestamps.
    """
    df = pd.read_csv(CONSUMPTION_CSV, parse_dates=["TimeDK"])
    df["hour"] = df["TimeDK"].dt.floor("h")
    hourly = (
        df.groupby("hour")["ConsumptionkWh"]
        .sum()
        / 1_000           # kWh → MWh = MW for 1-hour intervals
    )
    hourly.index.name = "TimeDK"
    hourly.name       = "baseline_load_mw"
    # Align to the price timestamps; forward-fill any gaps
    aligned = hourly.reindex(timestamps).ffill().bfill()
    return aligned

def generate_forecast_prices(
    actual: pd.Series,
    noise_std: float = 55.0,
) -> pd.Series:
    """
    Creates synthetic forecasted prices from actual prices + Gaussian noise.
    Noise std=55 DKK/MWh matches simulate_impact.py's daily forecast error.
    Replace with model predictions from your forecasting pipeline.
    """
    noise    = np.random.normal(0, noise_std, len(actual))
    forecast = (actual.values + noise).clip(min=0)
    return pd.Series(forecast, index=actual.index, name="forecast_price_dkk_mwh")


# -----------------------------------------------------------------------------
# 2. FLEXIBILITY CATEGORIES & SCENARIOS
# -----------------------------------------------------------------------------

def define_scenarios() -> dict:
    """
    Returns three scenario parameter dicts.

    Each flexibility category has:
      share_of_load    : fraction of total baseline load in this bucket
      max_movable      : max fraction of that category that CAN be shifted
      max_wait_hours   : upper bound on how far ahead load can shift
      decay_rate       : λ in exp(−λ·wait_hours) — higher = more waiting-averse
      price_sensitivity: steepness of the logistic price-response function
    """
    windows = {
        "short_flex"    : 6,
        "day_flex"      : 24,
        "multi_day_flex": FORECAST_HORIZON,
    }

    scenarios = {
        "conservative": {
            "short_flex": dict(
                share_of_load=0.06, max_movable=0.25,
                max_wait_hours=windows["short_flex"],
                decay_rate=0.60, price_sensitivity=0.40,
            ),
            "day_flex": dict(
                share_of_load=0.08, max_movable=0.30,
                max_wait_hours=windows["day_flex"],
                decay_rate=0.18, price_sensitivity=0.45,
            ),
            "multi_day_flex": dict(
                share_of_load=0.04, max_movable=0.20,
                max_wait_hours=windows["multi_day_flex"],
                decay_rate=0.04, price_sensitivity=0.35,
            ),
        },
        "medium": {
            "short_flex": dict(
                share_of_load=0.10, max_movable=0.40,
                max_wait_hours=windows["short_flex"],
                decay_rate=0.40, price_sensitivity=0.60,
            ),
            "day_flex": dict(
                share_of_load=0.12, max_movable=0.45,
                max_wait_hours=windows["day_flex"],
                decay_rate=0.12, price_sensitivity=0.65,
            ),
            "multi_day_flex": dict(
                share_of_load=0.08, max_movable=0.35,
                max_wait_hours=windows["multi_day_flex"],
                decay_rate=0.025, price_sensitivity=0.55,
            ),
        },
        "optimistic": {
            "short_flex": dict(
                share_of_load=0.14, max_movable=0.55,
                max_wait_hours=windows["short_flex"],
                decay_rate=0.25, price_sensitivity=0.80,
            ),
            "day_flex": dict(
                share_of_load=0.16, max_movable=0.60,
                max_wait_hours=windows["day_flex"],
                decay_rate=0.07, price_sensitivity=0.85,
            ),
            "multi_day_flex": dict(
                share_of_load=0.12, max_movable=0.50,
                max_wait_hours=windows["multi_day_flex"],
                decay_rate=0.015, price_sensitivity=0.70,
            ),
        },
    }
    return scenarios


# -----------------------------------------------------------------------------
# 3. CORE DECISION FUNCTIONS
# -----------------------------------------------------------------------------

def waiting_decay(wait_hours: float, decay_rate: float) -> float:
    """
    Exponential decay of willingness to shift as waiting time increases.
      decay(w) = exp(−λ · w)
    Returns value in (0, 1].
    """
    return float(np.exp(-decay_rate * wait_hours))


def price_response_strength(score: float, sensitivity: float, scale: float = 200.0) -> float:
    """
    Logistic price-response function.
    Maps a price-advantage score → fraction of max movable load [0, 1].

      raw    = sigmoid(sensitivity · score / scale)
      result = max(0,  2 · (raw − 0.5))   ← shifted so score=0 → response=0

    Higher sensitivity → sharper response to price differences.
    'scale' normalises the score to typical DKK/MWh units.
    """
    raw = 1.0 / (1.0 + np.exp(-sensitivity * (score / scale)))
    return max(0.0, (raw - 0.5) * 2.0)


def compute_shift_score(
    current_price: float,
    future_forecast: float,
    wait_hours: int,
    decay_rate: float,
) -> float:
    """
    score = max(0, price_advantage) × waiting_decay
    Returns 0 when the future period is not cheaper.
    """
    price_advantage = max(0.0, current_price - future_forecast)
    return price_advantage * waiting_decay(wait_hours, decay_rate)


def find_best_target(
    t: int,
    forecast_prices: np.ndarray,
    current_price: float,
    max_wait_hours: int,
    decay_rate: float,
) -> tuple:
    """
    Scans future hours in [t+1, t+max_wait_hours] and returns
    (best_future_index, best_score).
    Returns (-1, 0.0) if no beneficial future period exists.
    """
    n          = len(forecast_prices)
    best_idx   = -1
    best_score = 0.0

    for w in range(1, max_wait_hours + 1):
        future_t = t + w
        if future_t >= n:
            break
        score = compute_shift_score(current_price, forecast_prices[future_t], w, decay_rate)
        if score > best_score:
            best_score = score
            best_idx   = future_t

    return best_idx, best_score


# -----------------------------------------------------------------------------
# 4. MAIN SIMULATION
# -----------------------------------------------------------------------------

def run_simulation(
    timestamps     : pd.DatetimeIndex,
    baseline_load  : pd.Series,
    actual_prices  : pd.Series,
    forecast_prices: pd.Series,
    scenario_params: dict,
    scenario_name  : str,
) -> pd.DataFrame:
    """
    Core hourly simulation loop.

    For every hour t:
      1. Split baseline into inflexible + per-category flexible load.
      2. For each flex category find the best future target hour.
      3. Compute shifted amount via price-response function.
      4. Accumulate shifted-out (this hour) and shifted-in (target hour).

    Returns a detailed results DataFrame.
    """
    n        = len(timestamps)
    actual   = actual_prices.values
    forecast = forecast_prices.values
    load     = baseline_load.values

    total_flex_share = sum(p["share_of_load"] for p in scenario_params.values())
    inflexible_share = max(0.0, 1.0 - total_flex_share)

    shifted_out    = np.zeros(n)
    shifted_in     = np.zeros(n)
    target_hours   = np.full(n, -1, dtype=int)
    wait_hours_arr = np.zeros(n)
    shift_scores   = np.zeros(n)

    cat_shift_out  = {cat: np.zeros(n) for cat in scenario_params}

    for t in range(n):
        current_price = actual[t]
        hour_load     = load[t]

        for cat, params in scenario_params.items():
            cat_load = hour_load * params["share_of_load"]
            max_move = cat_load  * params["max_movable"]

            if max_move < 1e-6:
                continue

            best_idx, best_score = find_best_target(
                t, forecast, current_price,
                params["max_wait_hours"], params["decay_rate"],
            )

            if best_idx == -1 or best_score <= 0:
                continue

            response       = price_response_strength(best_score, params["price_sensitivity"])
            amount_shifted = max_move * response

            if amount_shifted < 1e-6:
                continue

            cat_shift_out[cat][t] += amount_shifted
            shifted_out[t]        += amount_shifted
            shifted_in[best_idx]  += amount_shifted

            # Record the dominant shift decision for this hour
            wait_h = best_idx - t
            if best_score > shift_scores[t]:
                wait_hours_arr[t] = wait_h
                shift_scores[t]   = best_score
                target_hours[t]   = best_idx

    shifted_total = load - shifted_out + shifted_in

    result = pd.DataFrame({
        "timestamp"          : timestamps,
        "scenario"           : scenario_name,
        "actual_price"       : actual,
        "forecast_price"     : forecast,
        "baseline_load"      : load,
        "inflexible_load"    : load * inflexible_share,
        "shifted_out"        : shifted_out,
        "shifted_in"         : shifted_in,
        "shifted_total_load" : shifted_total,
        "target_hour_idx"    : target_hours,
        "wait_hours"         : wait_hours_arr,
        "shift_score"        : shift_scores,
    })

    for cat, arr in cat_shift_out.items():
        result[f"flex_{cat}_load"] = load * scenario_params[cat]["share_of_load"]
        result[f"shift_out_{cat}"] = arr

    result["target_timestamp"] = result["target_hour_idx"].apply(
        lambda i: timestamps[i] if i >= 0 else pd.NaT
    )

    return result


# -----------------------------------------------------------------------------
# 5. KPI CALCULATION
# -----------------------------------------------------------------------------

def calculate_kpis(results: pd.DataFrame) -> dict:
    """
    Computes summary KPIs for a single scenario result DataFrame.
    Stress hours = top (100−STRESS_PERCENTILE)% most expensive hours.
    """
    df               = results.copy()
    stress_threshold = np.percentile(df["actual_price"], STRESS_PERCENTILE)
    stress_mask      = df["actual_price"] >= stress_threshold

    total_baseline   = df["baseline_load"].sum()
    total_shifted    = df["shifted_out"].sum()

    baseline_peak    = df["baseline_load"].max()
    shifted_peak     = df["shifted_total_load"].max()

    # Cost in kDKK (price DKK/MWh × load MW × 1 h = DKK; /1000 = kDKK)
    baseline_cost    = (df["actual_price"] * df["baseline_load"]).sum() / 1_000
    shifted_cost     = (df["actual_price"] * df["shifted_total_load"]).sum() / 1_000

    stress_baseline  = df.loc[stress_mask, "baseline_load"].sum()
    stress_shifted   = df.loc[stress_mask, "shifted_total_load"].sum()
    stress_reduction = stress_baseline - stress_shifted

    shifted_hours    = df[df["shifted_out"] > 0]
    avg_wait = (
        (shifted_hours["wait_hours"] * shifted_hours["shifted_out"]).sum()
        / shifted_hours["shifted_out"].sum()
        if len(shifted_hours) > 0 else 0.0
    )

    return {
        "scenario"                 : df["scenario"].iloc[0],
        "total_shifted_energy_mwh" : round(total_shifted, 1),
        "share_of_load_shifted_pct": round(100 * total_shifted / total_baseline, 2),
        "baseline_peak_mw"         : round(baseline_peak, 1),
        "shifted_peak_mw"          : round(shifted_peak, 1),
        "peak_reduction_pct"       : round(100 * (baseline_peak - shifted_peak) / baseline_peak, 2),
        "baseline_cost_dkk_k"      : round(baseline_cost, 0),
        "shifted_cost_dkk_k"       : round(shifted_cost, 0),
        "user_savings_dkk_k"       : round(baseline_cost - shifted_cost, 0),
        "savings_pct"              : round(100 * (baseline_cost - shifted_cost) / baseline_cost, 2),
        "stress_threshold_dkk_mwh" : round(stress_threshold, 1),
        "stress_load_reduced_mwh"  : round(stress_reduction, 1),
        "stress_reduction_pct"     : round(100 * stress_reduction / stress_baseline, 2)
                                     if stress_baseline > 0 else 0.0,
        "avg_wait_hours"           : round(avg_wait, 2),
        "n_shifting_hours"         : len(shifted_hours),
    }


# -----------------------------------------------------------------------------
# 6. PLOTTING
# -----------------------------------------------------------------------------

COLORS = {
    "conservative": "#5B8DB8",
    "medium"      : "#F4A261",
    "optimistic"  : "#2A9D8F",
    "baseline"    : "#6C757D",
    "stress"      : "#E63946",
}


def _fmt_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def plot_prices(result_df: pd.DataFrame) -> plt.Figure:
    """Plot 1 — Actual vs forecasted price over time."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(result_df["timestamp"], result_df["actual_price"],
            label="Actual price", lw=1.0, color="#264653")
    ax.plot(result_df["timestamp"], result_df["forecast_price"],
            label="Forecasted price", lw=0.8, alpha=0.7, color=COLORS["optimistic"])
    ax.set_title("Actual vs Forecasted Electricity Price", fontsize=14, fontweight="bold")
    ax.set_ylabel("DKK / MWh")
    ax.legend()
    _fmt_date_axis(ax)
    plt.tight_layout()
    return fig


def plot_load_comparison(all_results: dict) -> plt.Figure:
    """Plot 2 — Baseline vs shifted total load for each scenario."""
    n_sc  = len(all_results)
    fig, axes = plt.subplots(n_sc, 1, figsize=(14, 4 * n_sc), sharex=True)
    if n_sc == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, all_results.items()):
        ax.plot(df["timestamp"], df["baseline_load"],
                label="Baseline", lw=1.0, color=COLORS["baseline"], alpha=0.8)
        ax.plot(df["timestamp"], df["shifted_total_load"],
                label="Shifted", lw=1.0, color=COLORS[name])
        ax.fill_between(df["timestamp"],
                        df["baseline_load"], df["shifted_total_load"],
                        alpha=0.2, color=COLORS[name])
        ax.set_title(f"Baseline vs Shifted Load — {name.capitalize()}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("MW")
        ax.legend(loc="upper right")
        _fmt_date_axis(ax)
    plt.tight_layout()
    return fig


def plot_shift_flows(result_df: pd.DataFrame, scenario_name: str) -> plt.Figure:
    """Plots 3 & 4 — Shifted energy out and in per hour."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.bar(result_df["timestamp"], result_df["shifted_out"],
            width=0.04, color=COLORS["stress"], alpha=0.8, label="Shifted out")
    ax1.set_title(f"Load Shifted Out of Each Hour — {scenario_name.capitalize()}",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("MW")
    ax1.legend()

    ax2.bar(result_df["timestamp"], result_df["shifted_in"],
            width=0.04, color=COLORS["optimistic"], alpha=0.8, label="Shifted in")
    ax2.set_title(f"Load Shifted Into Each Hour — {scenario_name.capitalize()}",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("MW")
    ax2.legend()
    _fmt_date_axis(ax2)
    plt.tight_layout()
    return fig


def plot_stress_hours(all_results: dict, actual_prices: pd.Series) -> plt.Figure:
    """Plot 5 — Baseline vs shifted load during stress hours per scenario."""
    stress_threshold = np.percentile(actual_prices.values, STRESS_PERCENTILE)
    # Use .values to avoid index-alignment issues between price Series and result DataFrame
    stress_mask      = (actual_prices.values >= stress_threshold)

    scenarios   = list(all_results.keys())
    x           = np.arange(len(scenarios))
    bar_width   = 0.30

    baseline_stress = all_results[scenarios[0]]["baseline_load"].values[stress_mask].sum()
    shifted_vals    = [df["shifted_total_load"].values[stress_mask].sum()
                       for df in all_results.values()]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - bar_width / 2, [baseline_stress] * len(scenarios),
           width=bar_width, label="Baseline", color=COLORS["baseline"], alpha=0.8)
    ax.bar(x + bar_width / 2, shifted_vals, width=bar_width,
           color=[COLORS[s] for s in scenarios], alpha=0.9, label="After shift")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_title(
        f"Cumulative Load During Stress Hours (top {100 - STRESS_PERCENTILE}% price hours)\n"
        f"Threshold: {stress_threshold:.0f} DKK/MWh",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Cumulative MWh across stress hours")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_wait_time_distribution(result_df: pd.DataFrame, scenario_name: str):
    """Plot 6 — Distribution of waiting times for shifted load."""
    shifted_hours = result_df[result_df["shifted_out"] > 0]["wait_hours"]
    if shifted_hours.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(shifted_hours, bins=30, color=COLORS["medium"], alpha=0.85, edgecolor="white")
    ax.set_title(f"Distribution of Waiting Times — {scenario_name.capitalize()}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Wait hours")
    ax.set_ylabel("Count of shifting events")
    plt.tight_layout()
    return fig


def plot_waiting_decay_function() -> plt.Figure:
    """Plot 7 — How the waiting decay function behaves across λ values."""
    hours = np.linspace(0, 120, 500)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, lam, color in [
        ("Low aversion  λ=0.015", 0.015, COLORS["optimistic"]),
        ("Medium aversion λ=0.12", 0.12,  COLORS["medium"]),
        ("High aversion  λ=0.40", 0.40,  COLORS["conservative"]),
    ]:
        ax.plot(hours, np.exp(-lam * hours), label=label, color=color, lw=2)
    ax.axvline(24,  ls="--", color="grey", lw=0.8, alpha=0.7)
    ax.axvline(120, ls=":",  color="grey", lw=0.8, alpha=0.7)
    ax.text(25, 0.05, "24 h", color="grey", fontsize=8)
    ax.text(121, 0.05, "120 h", color="grey", fontsize=8)
    ax.set_title("Waiting Decay Function:  exp(−λ · wait_hours)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Wait time (hours)")
    ax.set_ylabel("Decay factor")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_price_response_function() -> plt.Figure:
    """Plot 8 — How the price-response function behaves across sensitivity values."""
    scores = np.linspace(0, 600, 500)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, sens, color in [
        ("Low sensitivity  0.35", 0.35, COLORS["conservative"]),
        ("Medium sensitivity 0.65", 0.65, COLORS["medium"]),
        ("High sensitivity  0.85", 0.85, COLORS["optimistic"]),
    ]:
        response = np.array([price_response_strength(s, sens) for s in scores])
        ax.plot(scores, response, label=label, color=color, lw=2)
    ax.set_title("Price Response Strength vs Shift Score",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Shift score  (DKK/MWh · decay factor)")
    ax.set_ylabel("Fraction of max movable load")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_scenario_comparison(kpi_list: list) -> plt.Figure:
    """Plot 9 — Side-by-side bar chart of key KPIs across scenarios."""
    df = pd.DataFrame(kpi_list).set_index("scenario")
    metrics = [
        ("total_shifted_energy_mwh",   "Shifted Energy\n(MWh)"),
        ("peak_reduction_pct",          "Peak Reduction\n(%)"),
        ("stress_reduction_pct",        "Stress-Hour\nReduction (%)"),
        ("savings_pct",                 "User Savings\n(%)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    for ax, (col, title) in zip(axes, metrics):
        vals   = df[col]
        colors = [COLORS[s] for s in df.index]
        bars   = ax.bar(df.index, vals, color=colors, alpha=0.9, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticklabels([s.capitalize() for s in df.index], rotation=15)
        upper = vals.max() * 1.3 if vals.max() > 0 else 1
        ax.set_ylim(0, upper)
    plt.suptitle("Scenario Comparison — Key KPIs", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_hourly_heatmap(result_df: pd.DataFrame, col: str, title: str) -> plt.Figure:
    """Plot 10 — Day × hour heatmap for any MW column (first 21 days shown)."""
    df          = result_df.copy()
    df["label"] = df["timestamp"].dt.strftime("%a %b %d")
    df["hour"]  = df["timestamp"].dt.hour

    pivot = df.pivot_table(index="label", columns="hour", values=col, aggfunc="mean")
    pivot = pivot.iloc[:21]

    fig, ax = plt.subplots(figsize=(16, max(5, len(pivot) * 0.40)))
    im      = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                        vmin=pivot.values.min(), vmax=pivot.values.max())
    plt.colorbar(im, ax=ax, label="MW")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=7)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Date")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 7. REPORTING
# -----------------------------------------------------------------------------

def print_assumption_overview(scenarios: dict) -> None:
    sep = "-" * 72
    print(f"\n{sep}")
    print("  SIMULATION ASSUMPTIONS OVERVIEW")
    print(sep)
    print(f"  Forecast horizon : {FORECAST_HORIZON} hours  ({FORECAST_HORIZON // 24} days)")
    print(f"  Simulation window: {SIMULATION_DAYS} days of hourly data")
    print(f"  Stress hours     : top {100 - STRESS_PERCENTILE}% most expensive hours")
    print()
    print("  Shifting logic:")
    print("    score     = max(0, current_price - forecast_price) * exp(-lambda * wait)")
    print("    response  = logistic(sensitivity * score / 200)  -> [0, 1]")
    print("    shifted   = flex_load * max_movable * response")
    print("    Best target = hour with highest positive score (not simply cheapest).")
    print("    Inflexible load is never shifted.")
    print()
    for sname, cats in scenarios.items():
        total_flex = sum(p["share_of_load"] for p in cats.values())
        print(f"  Scenario: {sname.upper()}")
        print(f"  {'Category':18s} {'share':>7}  {'max_move':>9}  "
              f"{'max_wait':>9}  {'decay_l':>9}  {'sensitivity':>11}")
        for cat, p in cats.items():
            print(f"    {cat:16s} {p['share_of_load']:>6.0%}  "
                  f"{p['max_movable']:>9.0%}  "
                  f"{p['max_wait_hours']:>8}h  "
                  f"{p['decay_rate']:>9.3f}  "
                  f"{p['price_sensitivity']:>11.2f}")
        print(f"    Total flexible: {total_flex:.0%}   Inflexible: {1 - total_flex:.0%}")
        print()
    print(sep)


def print_kpi_table(kpi_list: list) -> None:
    df = pd.DataFrame(kpi_list).set_index("scenario")
    print("\n  KEY PERFORMANCE INDICATORS")
    print(df.T.to_string())
    print()


def print_narrative_summary(kpi_list: list) -> None:
    sep = "-" * 72
    print(f"\n{sep}")
    print("  NARRATIVE SUMMARY")
    print(sep)
    for kpi in kpi_list:
        print(
            f"\n  [{kpi['scenario'].upper()}]  "
            f"The app shifted {kpi['total_shifted_energy_mwh']:,.0f} MWh "
            f"({kpi['share_of_load_shifted_pct']:.1f}% of total load), "
            f"reduced peak load by {kpi['peak_reduction_pct']:.1f}%, "
            f"cut stress-hour load by {kpi['stress_reduction_pct']:.1f}%, "
            f"and saved users ~{kpi['user_savings_dkk_k']:,.0f} kDKK "
            f"({kpi['savings_pct']:.2f}% of baseline electricity cost)."
        )
    print(f"\n{sep}\n")


# -----------------------------------------------------------------------------
# 8. MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load real data (same sources as simulate_impact.py) ----------------
    n_hours = SIMULATION_DAYS * 24
    print(f"Loading real DK1 data ({n_hours} hours) …")
    actual_prices   = load_real_prices(n_hours)
    timestamps      = actual_prices.index
    baseline_load   = load_real_baseline_load(timestamps)
    forecast_prices = generate_forecast_prices(actual_prices)

    scenarios = define_scenarios()

    # -- Run simulations ----------------------------------------------------
    all_results: dict = {}
    kpi_list: list    = []

    for scenario_name, scenario_params in scenarios.items():
        print(f"Running scenario: {scenario_name} …")
        result = run_simulation(
            timestamps, baseline_load, actual_prices,
            forecast_prices, scenario_params, scenario_name,
        )
        all_results[scenario_name] = result
        kpi_list.append(calculate_kpis(result))

    # -- Print reports ------------------------------------------------------
    print_assumption_overview(scenarios)
    print_kpi_table(kpi_list)
    print_narrative_summary(kpi_list)

    # -- Save CSVs ----------------------------------------------------------
    for name, df in all_results.items():
        path = OUTPUT_DIR / f"results_{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    kpi_df = pd.DataFrame(kpi_list)
    kpi_df.to_csv(OUTPUT_DIR / "kpi_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'kpi_summary.csv'}")

    # -- Plots --------------------------------------------------------------
    medium_df = all_results["medium"]
    print("Generating plots …")

    figs = {
        "01_prices"            : plot_prices(medium_df),
        "02_load_comparison"   : plot_load_comparison(all_results),
        "03_shift_flows"       : plot_shift_flows(medium_df, "medium"),
        "04_stress_hours"      : plot_stress_hours(all_results, actual_prices),
        "05_wait_distribution" : plot_wait_time_distribution(medium_df, "medium"),
        "06_decay_function"    : plot_waiting_decay_function(),
        "07_response_function" : plot_price_response_function(),
        "08_scenario_kpis"     : plot_scenario_comparison(kpi_list),
        "09_heatmap_baseline"  : plot_hourly_heatmap(
            medium_df, "baseline_load",
            "Hourly Baseline Load Heatmap — Medium scenario (MW)",
        ),
        "10_heatmap_shifted"   : plot_hourly_heatmap(
            medium_df, "shifted_total_load",
            "Hourly Shifted Load Heatmap — Medium scenario (MW)",
        ),
    }

    for fname, fig in figs.items():
        if fig is not None:
            path = OUTPUT_DIR / f"{fname}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
            plt.close(fig)

    print("\nDone. All outputs written to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
