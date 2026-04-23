from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_daily_system_consumption(
    n_days=60,
    csv_path=None,
):
    """
    Loads raw DK1 consumption data from CSV and returns the last n_days
    as daily total system consumption [MWh/day].

    Works for both:
    - hourly data -> aggregated to daily totals
    - already daily data -> used directly
    """

    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "consumption_dk1_raw.csv"

    df = pd.read_csv(csv_path)

    # ---- detect datetime column ----
    possible_datetime_cols = [
        "TimeDK"
    ]
    datetime_col = None
    for col in possible_datetime_cols:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col is None:
        raise ValueError(
            f"Could not find a datetime column in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    # ---- detect consumption column ----
    possible_consumption_cols = [
        "ConsumptionkWh"
    ]
    consumption_col = None
    for col in possible_consumption_cols:
        if col in df.columns:
            consumption_col = col
            break

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
    df = df.rename(columns={datetime_col: "datetime", consumption_col: "consumption"})

    # ---- decide whether data is hourly or already daily ----
    df["date"] = df["datetime"].dt.floor("D")

    obs_per_day = df.groupby("date").size().median()

    if obs_per_day > 1:
        # hourly/sub-daily data -> sum to daily total
        daily_consumption = (
            df.groupby("date", as_index=False)["consumption"]
            .sum()
            .sort_values("date")
        )
    else:
        # already daily data
        daily_consumption = (
            df.groupby("date", as_index=False)["consumption"]
            .first()
            .sort_values("date")
        )

    if len(daily_consumption) < n_days:
        raise ValueError(
            f"Requested {n_days} days, but only found {len(daily_consumption)} daily values in {csv_path}"
        )

    return daily_consumption["consumption"].tail(n_days).to_numpy()


def make_daily_prices(n_days=60, csv_path=None):

    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "day_ahead_prices_dk1_raw.csv"

    df = pd.read_csv(csv_path)

    df["TimeDK"] = pd.to_datetime(df["TimeDK"])

    df["date"] = df["TimeDK"].dt.floor("D")

    daily_prices = (
        df.groupby("date")["DayAheadPriceDKK"]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    daily_prices = daily_prices.tail(n_days)

    dates = daily_prices["date"].to_numpy()
    prices = daily_prices["DayAheadPriceDKK"].to_numpy()

    return dates, prices


def simulate_daily_shift(
    n_days=60,
    seed=42,
    model="logistic",                 # "threshold", "logistic", "elasticity"
    lookahead_days=5,                 # compare today to coming 1-5 days
    system_daily_energy_mwh=100000,   # total system energy use per day
    household_share_of_system=0.35,   # share of total electricity from households
    flexible_share_of_household=0.23, # share of household load that can be moved
    responsive_household_share=0.33,  # households that actually respond
    max_shift_fraction_of_flexible=0.70,
    elasticity_per_dkk_kwh=0.026,
    shift_not_shed_share=0.80,
    threshold_dkk_kwh=0.30,
    full_response_dkk_kwh=1.20,
    midpoint_dkk_kwh=0.60,
    steepness=4.0,
):
    """
    Simulation logic:
    - today's average price is known
    - the tool predicts average prices for the coming days
    - if a future day is cheaper, some flexible household demand is moved from today
      to the cheapest predicted day
    """

    rng = np.random.default_rng(seed)
    # Realised daily prices
    dates, price_today = make_daily_prices(
        n_days=n_days,
        csv_path=Path(__file__).resolve().parents[2] / "data" / "day_ahead_prices_dk1_raw.csv")
    # Forecasted daily prices (realised price + forecast error)
    forecast_error = rng.normal(0, 55, n_days)
    price_forecast = np.clip(price_today + forecast_error, 200, None)

    system_daily = load_daily_system_consumption(
        n_days=n_days,
        csv_path=Path(__file__).resolve().parents[2] / "data" / "consumption_dk1_raw.csv"
    )

    household_daily = system_daily * household_share_of_system
    other_daily = system_daily - household_daily
    flexible_daily = household_daily * flexible_share_of_household

    household_shifted = household_daily.copy()
    shifted_out = np.zeros(n_days)
    shifted_in = np.zeros(n_days)
    chosen_target_day = np.full(n_days, -1)
    best_spread_dkk_kwh = np.zeros(n_days)

    def response_threshold(diff_dkk_kwh):
        if diff_dkk_kwh <= threshold_dkk_kwh:
            return 0.0
        ramp = min(1.0, (diff_dkk_kwh - threshold_dkk_kwh) / (full_response_dkk_kwh - threshold_dkk_kwh))
        return responsive_household_share * max_shift_fraction_of_flexible * ramp

    def response_logistic(diff_dkk_kwh):
        sig = 1.0 / (1.0 + np.exp(-steepness * (diff_dkk_kwh - midpoint_dkk_kwh)))
        return responsive_household_share * max_shift_fraction_of_flexible * sig

    def response_elasticity(diff_dkk_kwh):
        reduction_share = elasticity_per_dkk_kwh * max(diff_dkk_kwh, 0.0) * shift_not_shed_share
        technical_cap = flexible_share_of_household * max_shift_fraction_of_flexible
        return min(reduction_share, technical_cap)

    response_fn = {
        "threshold": response_threshold,
        "logistic": response_logistic,
        "elasticity": response_elasticity,
    }[model]

    for t in range(n_days):
        end = min(n_days, t + 1 + lookahead_days)
        if t + 1 >= end:
            continue

        future_slice = slice(t + 1, end)

        # Decision is based on forecast of future days
        future_forecasts = price_forecast[future_slice]
        rel = np.argmin(future_forecasts)
        target = (t + 1) + rel

        # Spread between today's known price and cheapest forecasted coming-day price
        spread_dkk_kwh = (price_today[t] - price_forecast[target]) / 1000.0

        best_spread_dkk_kwh[t] = spread_dkk_kwh
        chosen_target_day[t] = target

        if spread_dkk_kwh <= 0: 
            continue

        shift_share = response_fn(spread_dkk_kwh)
        shift_mwh = flexible_daily[t] * shift_share

        household_shifted[t] -= shift_mwh
        household_shifted[target] += shift_mwh
        shifted_out[t] += shift_mwh
        shifted_in[target] += shift_mwh

    system_shifted = other_daily + household_shifted

    baseline_cost = np.sum(household_daily * price_today)
    shifted_cost = np.sum(household_shifted * price_today)
    savings = baseline_cost - shifted_cost

    summary = {
        "model": model,
        "days": n_days,
        "lookahead_days": lookahead_days,
        "avg_price_dkk_per_mwh": np.mean(price_today),
        "shifted_energy_mwh": np.sum(shifted_out),
        "baseline_peak_daily_mwh": np.max(system_daily),
        "shifted_peak_daily_mwh": np.max(system_shifted),
        "peak_reduction_daily_mwh": np.max(system_daily) - np.max(system_shifted),
        "residential_cost_savings_dkk": savings,
        "residential_cost_savings_pct": savings / baseline_cost if baseline_cost > 0 else 0,
    }

    df = pd.DataFrame({
        "date": dates,
        "price_today_dkk_per_mwh": price_today,
        "forecast_price_dkk_per_mwh": price_forecast,
        "system_daily_energy_baseline_mwh": system_daily,
        "household_daily_energy_baseline_mwh": household_daily,
        "household_daily_energy_shifted_mwh": household_shifted,
        "system_daily_energy_shifted_mwh": system_shifted,
        "flexible_daily_energy_mwh": flexible_daily,
        "shifted_out_mwh": shifted_out,
        "shifted_in_mwh": shifted_in,
        "best_spread_dkk_per_kwh": best_spread_dkk_kwh,
        "chosen_target_day_index": chosen_target_day,
    })

    return df, summary


def plot_shift_results(df):
    """Create the three main figures used to explain the shift simulation."""
    green_dark = "#1B5E20"
    green_mid = "#43A047"
    green_light = "#A5D6A7"
    green_fill = "#E8F5E9"

    fig, axes = plt.subplots(3, 1, figsize=(12, 13))

    axes[0].plot(
        df["date"],
        df["system_daily_energy_baseline_mwh"],
        color=green_light,
        linewidth=2,
        label="Baseline",
    )
    axes[0].plot(
        df["date"],
        df["system_daily_energy_shifted_mwh"],
        color=green_dark,
        linewidth=2,
        label="After tool",
    )
    axes[0].set_title("Daily system energy before and after shifting")
    axes[0].set_ylabel("Energy [MWh/day]")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].bar(df["date"], df["shifted_out_mwh"], color=green_mid)
    axes[1].set_title("Energy moved away from each day")
    axes[1].set_ylabel("Shifted energy [MWh]")
    axes[1].grid(axis="y", alpha=0.25)

    spreads = np.linspace(0, 1.5, 200)
    response = 0.33 * 0.70 * (1.0 / (1.0 + np.exp(-4.0 * (spreads - 0.60))))
    axes[2].plot(spreads, 100 * response, color=green_dark, linewidth=3)
    axes[2].fill_between(spreads, 0, 100 * response, color=green_fill)
    axes[2].set_title("Illustrative household response curve")
    axes[2].set_xlabel("Today's price minus cheapest coming-day forecast [DKK/kWh]")
    axes[2].set_ylabel("Shifted share of flexible household load [%]")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    return fig


def run_simulation_demo(save_outputs=False):
    """Run the default logistic shift simulation and optionally save the results."""
    df, summary = simulate_daily_shift(n_days=60, lookahead_days=5, model="logistic")
    fig = plot_shift_results(df)

    if save_outputs:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "daily_energy_shift_results.csv", index=False)

    return df, summary, fig


if __name__ == "__main__":
    demo_df, demo_summary, _ = run_simulation_demo(save_outputs=True)
    print("SUMMARY")
    for key, value in demo_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value}")
    plt.show()
