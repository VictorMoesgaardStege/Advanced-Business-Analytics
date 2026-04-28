import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── Consumption ───────────────────────────────────────────────────────────────
cons = pd.read_csv(Path("data/consumption_dk1_raw.csv"), parse_dates=["TimeDK"])
hourly = (
    cons.groupby("TimeDK", as_index=False)["ConsumptionkWh"]
    .sum()
    .rename(columns={"TimeDK": "_ts", "ConsumptionkWh": "MWh"})
)
hourly["MWh"] /= 1000

# ── Prices ────────────────────────────────────────────────────────────────────
prices = (
    pd.read_csv(Path("data/day_ahead_prices_dk1_raw.csv"), parse_dates=["TimeDK"])
    .pipe(lambda d: d[d["PriceArea"] == "DK1"])
    .groupby("TimeDK", as_index=False)["DayAheadPriceDKK"]
    .mean()
    .rename(columns={"TimeDK": "_ts", "DayAheadPriceDKK": "PriceDKK"})
)

# ── Last 7 days ───────────────────────────────────────────────────────────────
end   = hourly["_ts"].max()
start = end - pd.Timedelta(days=7)

week_cons  = hourly[(hourly["_ts"] >= start) & (hourly["_ts"] <= end)]
week_price = prices[(prices["_ts"] >= start) & (prices["_ts"] <= end)]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 4))

ax1.fill_between(week_cons["_ts"], week_cons["MWh"], alpha=0.15, color="steelblue")
ax1.plot(week_cons["_ts"], week_cons["MWh"], color="steelblue", linewidth=1.5, label="Consumption (MWh)")
ax1.set_ylabel("Consumption (MWh)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")

ax2 = ax1.twinx()
ax2.plot(week_price["_ts"], week_price["PriceDKK"], color="darkorange", linewidth=1.5,
         linestyle="--", label="Price (DKK/MWh)")
ax2.set_ylabel("Price (DKK/MWh)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")

# Day markers
for day in pd.date_range(start.normalize() + pd.Timedelta(days=1), end.normalize(), freq="D"):
    ax1.axvline(day, color="grey", linewidth=0.6, linestyle=":")
    ax1.text(day, ax1.get_ylim()[1], day.strftime("%a\n%d %b"),
             ha="center", va="top", fontsize=8, color="grey")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

ax1.set_title(f"DK1 hourly consumption & spot price — {start.strftime('%d %b')} to {end.strftime('%d %b %Y')}")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
fig.autofmt_xdate(rotation=0, ha="center")
fig.tight_layout()
plt.savefig("outputs/model/fig_consumption_week.png", dpi=150)
plt.show()
print("Saved → outputs/model/fig_consumption_week.png")
