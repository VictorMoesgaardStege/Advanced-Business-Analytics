import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(
    Path("data/private_consumption_heating_national_raw.csv"),
    parse_dates=["TimeDK"],
    encoding="utf-8",
)

df["hour"] = df["TimeDK"].dt.hour

# Total consumption per hour across all categories and days
hourly_total = df.groupby("hour")["ConsumptionkWh"].sum()

# Normalise so the 24 values sum to 100 %
hourly_pct = (hourly_total / hourly_total.sum() * 100).reindex(range(24))

# ── Tabular output ─────────────────────────────────────────────────────────────
table = pd.DataFrame({
    "hour":    hourly_pct.index,
    "pct":     hourly_pct.values.round(3),
})
table["cumulative_pct"] = table["pct"].cumsum().round(3)
print(table.to_string(index=False))
print(f"\nSum of pct column: {table['pct'].sum():.3f} %")

# ── Plot ───────────────────────────────────────────────────────────────────────
# Per-category breakdown for stacking, also normalised to the same total
hourly_by_cat = (
    df.groupby(["hour", "HeatingCategory"])["ConsumptionkWh"]
    .sum()
    .reset_index()
)
total_kwh = hourly_total.sum()
hourly_by_cat["pct"] = hourly_by_cat["ConsumptionkWh"] / total_kwh * 100

heating_cats = hourly_by_cat["HeatingCategory"].unique()
colors = {"Elvarme eller varmepumpe": "#E53935", "Andet": "#90A4AE", "-": "#CFD8DC"}

fig, ax = plt.subplots(figsize=(12, 5))

bottom = pd.Series(0.0, index=range(24))
for cat in heating_cats:
    vals = (
        hourly_by_cat[hourly_by_cat["HeatingCategory"] == cat]
        .set_index("hour")["pct"]
        .reindex(range(24), fill_value=0)
    )
    ax.bar(range(24), vals, bottom=bottom, label=cat,
           color=colors.get(cat, "#78909C"), alpha=0.9, width=0.7)
    bottom += vals

# Annotate each bar with its total %
for h in range(24):
    ax.text(h, hourly_pct[h] + 0.05, f"{hourly_pct[h]:.1f}%",
            ha="center", va="bottom", fontsize=6.5, color="black")

ax.set_xticks(range(24))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Share of daily consumption (%)")
ax.set_title("Private heating — hourly load profile, Denmark 2025\n(normalised: 24 bars sum to 100 %)")
ax.legend(title="Heating category", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.savefig("outputs/model/fig_heating_hourly_distribution.png", dpi=150)
plt.show()
print("Saved -> outputs/model/fig_heating_hourly_distribution.png")
