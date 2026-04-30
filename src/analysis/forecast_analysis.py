from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("data")


def plot_raw_price_and_weather(
    start: str = "2024-01-01",
    end: str = "2024-01-14",
    price_csv: str | Path = DATA_DIR / "day_ahead_prices_dk1_raw.csv",
    weather_csv: str | Path = DATA_DIR / "weather_actuals_raw.csv",
):
    """Plot scraped DK1 prices and actual weather for the same period.

    This is intended as a quick visual check of the raw scraped data before
    processing. It shows the electricity price together with temperature, wind,
    and solar radiation for the selected period.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    prices = pd.read_csv(price_csv, parse_dates=["TimeDK"])
    prices = (
        prices[prices["PriceArea"].astype(str).eq("DK1")]
        .sort_values("TimeDK")
        .set_index("TimeDK")
    )
    price_col = "DayAheadPriceDKK" if "DayAheadPriceDKK" in prices.columns else "DayAheadPriceEUR"
    prices[price_col] = pd.to_numeric(prices[price_col], errors="coerce")
    prices = prices.loc[start_ts:end_ts, [price_col]].resample("h").mean()

    weather = pd.read_csv(weather_csv, parse_dates=["TimeDK"])
    if "region" in weather.columns:
        weather = weather[weather["region"].astype(str).eq("DK1_west")]
    weather = weather.sort_values("TimeDK").set_index("TimeDK")
    weather = weather.loc[
        start_ts:end_ts,
        ["temperature_2m", "wind_speed_100m", "shortwave_radiation"],
    ].resample("h").mean()

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(prices.index, prices[price_col], color="#2563eb", linewidth=1.4)
    axes[0].set_ylabel("Price\nDKK/MWh" if price_col.endswith("DKK") else "Price\nEUR/MWh")
    axes[0].set_title(f"Scraped DK1 price and weather data ({start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d})")

    axes[1].plot(weather.index, weather["temperature_2m"], color="#ef4444", linewidth=1.4)
    axes[1].set_ylabel("Temp\nC")

    axes[2].plot(weather.index, weather["wind_speed_100m"], color="#0891b2", linewidth=1.4)
    axes[2].set_ylabel("Wind 100m\nm/s")

    axes[3].plot(weather.index, weather["shortwave_radiation"], color="#d97706", linewidth=1.4)
    axes[3].set_ylabel("Solar\nW/m2")
    axes[3].set_xlabel("Time")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig
