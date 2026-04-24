"""
Residual diagnostics for the DK1 XGBoost walk-forward forecast outputs.

This script reads the predictions produced by XG_Boost_full_Res.py and
creates:

1. Residual ACF plots plus Ljung-Box statistics by day model
2. Residual breakdowns by price regime, with focus on negative prices
   and top-5-percent spike hours
3. Event-study plots around extreme price spikes to inspect lag/smoothing

Outputs are written to:
    outputs/model/model validation/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover - fallback is handled at runtime
    chi2 = None


TARGET_COL = "DayAheadPriceEUR"
MODEL_OUTPUT_DIR = Path("outputs/model")
VALIDATION_DIR = MODEL_OUTPUT_DIR / "model validation"
PREDICTIONS_PATH = MODEL_OUTPUT_DIR / "predictions.parquet"

DAY_GROUPS = {
    1: (1, 24),
    2: (25, 48),
    3: (49, 72),
    4: (73, 96),
    5: (97, 120),
}

MAX_ACF_LAG = 48


def ensure_output_dir() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}. "
            "Run XG_Boost_full_Res.py first to create walk-forward predictions."
        )

    preds = pd.read_parquet(PREDICTIONS_PATH).copy()
    preds["issue_time"] = pd.to_datetime(preds["issue_time"])
    preds["target_time"] = pd.to_datetime(preds["target_time"])
    preds["residual"] = preds["predicted"] - preds[TARGET_COL]
    preds["abs_error"] = np.abs(preds["residual"])
    return preds.sort_values(["issue_time", "target_time"]).reset_index(drop=True)


def residual_series_by_day(preds: pd.DataFrame, day: int) -> pd.Series:
    h_lo, h_hi = DAY_GROUPS[day]
    subset = preds[preds["horizon_h"].between(h_lo, h_hi)].copy()
    series = (
        subset.groupby("issue_time", as_index=True)["residual"]
        .mean()
        .sort_index()
    )
    return series


def compute_acf(series: pd.Series, max_lag: int) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.array([1.0])

    centered = values - values.mean()
    denom = np.dot(centered, centered)
    if denom == 0:
        return np.ones(min(max_lag, n - 1) + 1)

    acf_values = [1.0]
    usable_lag = min(max_lag, n - 1)
    for lag in range(1, usable_lag + 1):
        numer = np.dot(centered[:-lag], centered[lag:])
        acf_values.append(float(numer / denom))
    return np.asarray(acf_values)


def ljung_box(series: pd.Series, lag: int) -> tuple[float, float | None]:
    acf_values = compute_acf(series, lag)
    n = len(series)
    usable_lag = min(lag, len(acf_values) - 1)

    if usable_lag < 1 or n <= usable_lag:
        return np.nan, np.nan

    q_stat = n * (n + 2) * np.sum(
        [(acf_values[k] ** 2) / (n - k) for k in range(1, usable_lag + 1)]
    )

    if chi2 is None:
        return float(q_stat), np.nan

    p_value = float(chi2.sf(q_stat, usable_lag))
    return float(q_stat), p_value


def save_residual_acf_and_ljung_box(preds: pd.DataFrame) -> None:
    print("Creating residual ACF and Ljung-Box diagnostics...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    axes = axes.ravel()
    summary_rows = []

    for day, ax in zip(DAY_GROUPS, axes):
        series = residual_series_by_day(preds, day)
        acf_values = compute_acf(series, MAX_ACF_LAG)
        lags = np.arange(len(acf_values))
        conf = 1.96 / np.sqrt(len(series)) if len(series) else np.nan
        acf_tail = acf_values[1:] if len(acf_values) > 1 else np.array([0.0])

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(conf, color="#B22222", linewidth=0.8, linestyle=":")
        ax.axhline(-conf, color="#B22222", linewidth=0.8, linestyle=":")
        ax.vlines(lags[1:], 0, acf_values[1:], color="#1768AC", linewidth=1.5)
        ax.scatter(lags[1:], acf_values[1:], color="#1768AC", s=16)

        q24, p24 = ljung_box(series, lag=min(24, max(1, len(series) - 1)))
        q48, p48 = ljung_box(series, lag=min(48, max(1, len(series) - 1)))

        ax.set_title(
            (
                f"Day {day} residual ACF\n"
                f"LB(24) p={p24:.3g} | LB(48) p={p48:.3g}"
                if pd.notna(p24) and pd.notna(p48)
                else f"Day {day} residual ACF\nLB p-values unavailable"
            ),
            fontsize=10,
        )
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_xlim(0, MAX_ACF_LAG + 1)
        ax.set_ylim(
            min(-0.6, np.nanmin(acf_tail) - 0.05),
            max(0.6, np.nanmax(acf_tail) + 0.05),
        )

        summary_rows.append(
            {
                "day_model": day,
                "series_n": int(len(series)),
                "acf_lag_1": float(acf_values[1]) if len(acf_values) > 1 else np.nan,
                "acf_lag_24": float(acf_values[24]) if len(acf_values) > 24 else np.nan,
                "acf_lag_48": float(acf_values[48]) if len(acf_values) > 48 else np.nan,
                "ljung_box_q_24": q24,
                "ljung_box_p_24": p24,
                "ljung_box_q_48": q48,
                "ljung_box_p_48": p48,
            }
        )

    axes[-1].axis("off")
    fig.suptitle("Residual autocorrelation by day model", fontsize=13)
    fig.tight_layout()
    fig.savefig(VALIDATION_DIR / "residual_acf_ljung_box.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(VALIDATION_DIR / "residual_ljung_box_summary.csv", index=False)


def price_regime(value: float, spike_threshold: float) -> str:
    if value < 0:
        return "negative_price"
    if value >= spike_threshold:
        return "top_5pct_spike"
    return "other_hours"


def save_price_regime_breakdown(preds: pd.DataFrame) -> None:
    print("Creating residual breakdown by price regime...")
    spike_threshold = float(preds[TARGET_COL].quantile(0.95))

    rows = []
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        subset = preds[preds["horizon_h"].between(h_lo, h_hi)].copy()
        subset["regime"] = subset[TARGET_COL].apply(lambda value: price_regime(value, spike_threshold))

        for regime, grp in subset.groupby("regime"):
            rows.append(
                {
                    "day_model": day,
                    "regime": regime,
                    "n_rows": int(len(grp)),
                    "mean_residual": float(grp["residual"].mean()),
                    "mae": float(grp["abs_error"].mean()),
                    "median_abs_error": float(grp["abs_error"].median()),
                    "p90_abs_error": float(grp["abs_error"].quantile(0.90)),
                    "underprediction_share": float(np.mean(grp["residual"] < 0)),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["regime", "day_model"]).reset_index(drop=True)
    summary.to_csv(VALIDATION_DIR / "residual_price_regime_summary.csv", index=False)

    regime_order = ["negative_price", "top_5pct_spike", "other_hours"]
    mae_matrix = (
        summary.pivot(index="regime", columns="day_model", values="mae")
        .reindex(regime_order)
    )
    bias_matrix = (
        summary.pivot(index="regime", columns="day_model", values="mean_residual")
        .reindex(regime_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    im0 = axes[0].imshow(mae_matrix.to_numpy(), cmap="YlOrRd", aspect="auto")
    axes[0].set_title("MAE by price regime and day model")
    axes[0].set_xticks(range(len(mae_matrix.columns)), mae_matrix.columns)
    axes[0].set_yticks(range(len(mae_matrix.index)), mae_matrix.index)
    axes[0].set_xlabel("Day model")
    axes[0].set_ylabel("Price regime")
    for row in range(mae_matrix.shape[0]):
        for col in range(mae_matrix.shape[1]):
            axes[0].text(col, row, f"{mae_matrix.iloc[row, col]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="EUR/MWh")

    im1 = axes[1].imshow(bias_matrix.to_numpy(), cmap="RdBu_r", aspect="auto")
    axes[1].set_title("Mean residual by price regime and day model")
    axes[1].set_xticks(range(len(bias_matrix.columns)), bias_matrix.columns)
    axes[1].set_yticks(range(len(bias_matrix.index)), bias_matrix.index)
    axes[1].set_xlabel("Day model")
    axes[1].set_ylabel("Price regime")
    for row in range(bias_matrix.shape[0]):
        for col in range(bias_matrix.shape[1]):
            axes[1].text(col, row, f"{bias_matrix.iloc[row, col]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Predicted - actual")

    fig.suptitle(
        f"Residual regimes (spike threshold = {spike_threshold:.1f} EUR/MWh)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(VALIDATION_DIR / "residual_price_regime_heatmaps.png", dpi=150)
    plt.close(fig)


def aggregate_day_target_series(preds: pd.DataFrame, day: int) -> pd.DataFrame:
    h_lo, h_hi = DAY_GROUPS[day]
    subset = preds[preds["horizon_h"].between(h_lo, h_hi)].copy()
    return (
        subset.groupby("target_time", as_index=False)
        .agg(actual=(TARGET_COL, "mean"), predicted=("predicted", "mean"))
        .sort_values("target_time")
        .reset_index(drop=True)
    )


def select_spike_events(series_df: pd.DataFrame, percentile: float = 0.99, min_gap_hours: int = 24) -> pd.DataFrame:
    threshold = float(series_df["actual"].quantile(percentile))
    candidates = series_df[series_df["actual"] >= threshold].sort_values("actual", ascending=False).copy()

    selected_rows = []
    chosen_times: list[pd.Timestamp] = []
    for _, row in candidates.iterrows():
        target_time = row["target_time"]
        if all(abs((target_time - previous).total_seconds()) >= min_gap_hours * 3600 for previous in chosen_times):
            selected_rows.append(row)
            chosen_times.append(target_time)

    if not selected_rows:
        return pd.DataFrame(columns=series_df.columns)

    return pd.DataFrame(selected_rows).sort_values("target_time").reset_index(drop=True)


def event_window_rows(
    series_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_hours: int = 48,
) -> pd.DataFrame:
    rows = []
    indexed = series_df.set_index("target_time")

    for _, event in events_df.iterrows():
        event_time = event["target_time"]
        for rel_hour in range(-window_hours, window_hours + 1):
            timestamp = event_time + pd.Timedelta(hours=rel_hour)
            if timestamp not in indexed.index:
                continue

            match = indexed.loc[timestamp]
            rows.append(
                {
                    "event_time": event_time,
                    "relative_hour": rel_hour,
                    "actual": float(match["actual"]),
                    "predicted": float(match["predicted"]),
                }
            )

    return pd.DataFrame(rows)


def save_spike_event_study(preds: pd.DataFrame) -> None:
    print("Creating event-study plots around extreme spikes...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    axes = axes.ravel()
    summary_rows = []

    for day, ax in zip(DAY_GROUPS, axes):
        series_df = aggregate_day_target_series(preds, day)
        events_df = select_spike_events(series_df, percentile=0.99, min_gap_hours=24)
        event_rows = event_window_rows(series_df, events_df, window_hours=48)

        if event_rows.empty:
            ax.set_title(f"Day {day}: no spike events found")
            ax.axis("off")
            continue

        profile = (
            event_rows.groupby("relative_hour", as_index=False)
            .agg(actual=("actual", "mean"), predicted=("predicted", "mean"))
            .sort_values("relative_hour")
        )

        ax.plot(profile["relative_hour"], profile["actual"], color="black", linewidth=2, label="Actual")
        ax.plot(profile["relative_hour"], profile["predicted"], color="#1768AC", linewidth=2, linestyle="--", label="Predicted")
        ax.axvline(0, color="#B22222", linewidth=1, linestyle=":")
        ax.set_title(
            f"Day {day} | events={len(events_df)} | mean peak miss={(profile['predicted'].max() - profile['actual'].max()):.1f}",
            fontsize=10,
        )
        ax.set_xlabel("Hours relative to spike")
        ax.set_ylabel("EUR/MWh")
        ax.legend(fontsize=8, loc="upper right")

        summary_rows.append(
            {
                "day_model": day,
                "n_events": int(len(events_df)),
                "spike_threshold_actual": float(series_df["actual"].quantile(0.99)),
                "mean_peak_actual": float(profile["actual"].max()),
                "mean_peak_predicted": float(profile["predicted"].max()),
                "peak_gap_pred_minus_actual": float(profile["predicted"].max() - profile["actual"].max()),
            }
        )

        events_df.assign(day_model=day).to_csv(
            VALIDATION_DIR / f"residual_spike_events_day{day}.csv",
            index=False,
        )

    axes[-1].axis("off")
    fig.suptitle("Event study around extreme price spikes", fontsize=13)
    fig.tight_layout()
    fig.savefig(VALIDATION_DIR / "residual_spike_event_study.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(VALIDATION_DIR / "residual_spike_event_summary.csv", index=False)


def main() -> None:
    ensure_output_dir()

    print("=" * 70)
    print("Residual diagnostics for DK1 walk-forward predictions")
    print("=" * 70)

    preds = load_predictions()
    save_residual_acf_and_ljung_box(preds)
    save_price_regime_breakdown(preds)
    save_spike_event_study(preds)

    print(f"Residual diagnostics saved to {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
