"""
model_uncertainty.py  -  Forecast accuracy and uncertainty plots for the DK1 XGBoost model
===========================================================================================
Callable functions for use in Jupyter notebooks.

Quick start
-----------
    import pandas as pd
    from src.analysis.model_uncertainty import (
        plot_actual_vs_predicted,
        plot_mae_by_horizon,
        plot_single_forecast,
    )

    preds_df = pd.read_parquet("outputs/model/predictions.parquet")

    plot_actual_vs_predicted(preds_df)
    plot_actual_vs_predicted(preds_df, start_date="2024-06-01", days=14)
    plot_mae_by_horizon(preds_df)
    plot_single_forecast(preds_df)
    plot_single_forecast(preds_df, issue_time="2024-08-15 00:00")
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.XG_Boost_full_Res import DAY_GROUPS, TARGET_COL, COLORS


def plot_actual_vs_predicted(
    preds_df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    days: int = 28,
) -> None:
    """Plot actual vs predicted prices for each day model.

    Parameters
    ----------
    preds_df   : predictions.parquet loaded as DataFrame
    start_date : start of the window to display (default: `days` before the last target_time)
    days       : number of days to show (default: 28)
    """
    if start_date is not None:
        start_t = pd.Timestamp(start_date)
    else:
        last_fold = preds_df["fold"].max()
        end_t     = preds_df[preds_df["fold"] == last_fold]["target_time"].max()
        start_t   = end_t - pd.Timedelta(days=days)

    end_t = start_t + pd.Timedelta(days=days)
    sub   = preds_df[
        preds_df["target_time"].between(start_t, end_t)
    ].copy()

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    for ax, (day, (h_lo, h_hi)), color in zip(axes, DAY_GROUPS.items(), COLORS):
        grp = (
            sub[sub["horizon_h"].between(h_lo, h_hi)]
            .groupby("target_time")
            .agg(actual=(TARGET_COL, "mean"), predicted=("predicted", "mean"))
        )
        if grp.empty:
            ax.set_title(f"Day {day}  (h{h_lo}-{h_hi})   no data", fontsize=10)
            continue
        ax.plot(grp.index, grp["actual"],    color="black", linewidth=1.2, label="Actual")
        ax.plot(grp.index, grp["predicted"], color=color,   linewidth=1.2,
                linestyle="--", label="Predicted")
        mae = mean_absolute_error(grp["actual"], grp["predicted"])
        ax.set_title(f"Day {day}  (h{h_lo}-{h_hi})   MAE = {mae:.1f} EUR/MWh", fontsize=10)
        ax.set_ylabel("EUR/MWh")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Target time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.suptitle(
        f"Actual vs Predicted  --  {start_t.strftime('%Y-%m-%d')} to {end_t.strftime('%Y-%m-%d')}",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


def plot_mae_by_horizon(preds_df: pd.DataFrame) -> None:
    """Plot MAE at each forecast horizon across all folds."""
    rows = []
    for h in range(1, 121):
        sub = preds_df[preds_df["horizon_h"] == h]
        if len(sub):
            rows.append({"h": h, "mae": mean_absolute_error(sub[TARGET_COL], sub["predicted"])})
    res = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(res["h"], res["mae"], linewidth=1.5, color="steelblue")
    for day, (h_lo, h_hi), color in zip(DAY_GROUPS.keys(), DAY_GROUPS.values(), COLORS):
        ax.axvspan(h_lo - 0.5, h_hi + 0.5, alpha=0.08, color=color, label=f"Day {day}")
    ax.set(xlabel="Horizon (hours ahead)", ylabel="MAE (EUR/MWh)",
           title="MAE by forecast horizon  (all folds)")
    ax.legend(ncol=5)
    fig.tight_layout()
    plt.show()


def plot_single_forecast(
    preds_df: pd.DataFrame,
    issue_time: str | pd.Timestamp | None = None,
) -> None:
    """Plot a single 120h forecast vs actuals.

    Parameters
    ----------
    preds_df   : predictions.parquet loaded as DataFrame
    issue_time : which forecast to show (default: latest issue_time in the last fold)
    """
    if issue_time is not None:
        issue = pd.Timestamp(issue_time)
        if issue not in preds_df["issue_time"].values:
            available = preds_df["issue_time"].drop_duplicates().sort_values()
            issue = available.iloc[(available - issue).abs().argsort().iloc[0]]
            print(f"  Exact issue_time not found — using nearest: {issue}")
    else:
        last_fold = preds_df["fold"].max()
        issue     = preds_df[preds_df["fold"] == last_fold]["issue_time"].max()

    single = preds_df[preds_df["issue_time"] == issue].sort_values("horizon_h").copy()

    fig, ax = plt.subplots(figsize=(14, 4))
    for day, (h_lo, h_hi), color in zip(DAY_GROUPS.keys(), DAY_GROUPS.values(), COLORS):
        rows_lo = single[single["horizon_h"] == h_lo]
        rows_hi = single[single["horizon_h"] == h_hi]
        if rows_lo.empty or rows_hi.empty:
            continue
        ax.axvspan(
            rows_lo["target_time"].values[0],
            rows_hi["target_time"].values[0],
            alpha=0.08, color=color, label=f"Day {day}",
        )
    ax.plot(single["target_time"], single[TARGET_COL],
            color="black", linewidth=1.5, label="Actual")
    ax.plot(single["target_time"], single["predicted"],
            color="steelblue", linewidth=1.5, linestyle="--", label="Predicted")
    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("Target time")
    ax.set_title(f"Full 120h forecast  --  issued at {issue.strftime('%Y-%m-%d %H:%M')}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.legend(ncol=7, fontsize=8)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    plt.show()
