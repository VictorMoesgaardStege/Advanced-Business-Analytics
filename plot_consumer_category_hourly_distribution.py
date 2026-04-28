import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(
    Path("data/consumption_consumer_category_raw.csv"),
    parse_dates=["TimeDK"],
    encoding="utf-8",
)

df["hour"] = df["TimeDK"].dt.hour

# ── Aggregate per hour × category ─────────────────────────────────────────────
hourly_by_cat = (
    df.groupby(["hour", "ConsumerCategory3"])["ConsumptionkWh"]
    .sum()
    .reset_index()
)

# Normalise within each hour so each hour sums to 100 %
hourly_total = hourly_by_cat.groupby("hour")["ConsumptionkWh"].transform("sum")
hourly_by_cat["pct"] = hourly_by_cat["ConsumptionkWh"] / hourly_total * 100

hourly_total_pct = hourly_by_cat.groupby("hour")["pct"].sum().reindex(range(24))

# ── Tabular output ─────────────────────────────────────────────────────────────
cats = sorted(hourly_by_cat["ConsumerCategory3"].unique())
table = pd.DataFrame({"hour": range(24)})
for cat in cats:
    col = (
        hourly_by_cat[hourly_by_cat["ConsumerCategory3"] == cat]
        .set_index("hour")["pct"]
        .reindex(range(24), fill_value=0)
        .round(3)
    )
    table[cat] = col.values

table["total_pct"] = table[cats].sum(axis=1).round(3)
table["cumulative_pct"] = table["total_pct"].cumsum().round(3)

print(table.to_string(index=False))
print(f"\nSum of total_pct: {table['total_pct'].sum():.3f} %")

# ── Plot ───────────────────────────────────────────────────────────────────────
colors = {
    "Privat":                  "#1E88E5",
    "Erhverv":                 "#43A047",
    "Offentlige foretagender": "#FB8C00",
}

fig, ax = plt.subplots(figsize=(13, 5))

bottom = pd.Series(0.0, index=range(24))
for cat in cats:
    vals = (
        hourly_by_cat[hourly_by_cat["ConsumerCategory3"] == cat]
        .set_index("hour")["pct"]
        .reindex(range(24), fill_value=0)
    )
    ax.bar(range(24), vals, bottom=bottom, label=cat,
           color=colors.get(cat, "#90A4AE"), alpha=0.9, width=0.7)
    bottom += vals

for h in range(24):
    ax.text(h, hourly_total_pct[h] + 0.03, f"{hourly_total_pct[h]:.1f}%",
            ha="center", va="bottom", fontsize=6.5, color="black")

ax.set_xticks(range(24))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Share within hour (%)")
ax.set_title("Electricity consumption by consumer category — Denmark 2025\n(each hour normalised to 100 %)")
ax.legend(title="Consumer category", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.savefig("outputs/model/fig_consumer_category_hourly_distribution.png", dpi=150)
plt.show()
print("Saved -> outputs/model/fig_consumer_category_hourly_distribution.png")
