"""
Behavioral validation diagnostics for the DK1 XGBoost price forecaster.

This script keeps the original training script untouched and rebuilds the
same feature matrix / five day-model setup to generate behavioral checks:

1. SHAP dependence plots for core economic / physical drivers
2. ALE curves for the same drivers
3. Counterfactual perturbation tests on sampled realistic rows
4. Boundary-jump analysis across the 24/25, 48/49, 72/73, 96/97 hour splits

Outputs are written to:
    outputs/model/model validation/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


REGION = "DK1_west"
TARGET_COL = "DayAheadPriceEUR"
DATA_DIR = Path("data")
MODEL_OUTPUT_DIR = Path("outputs/model")
VALIDATION_DIR = MODEL_OUTPUT_DIR / "model validation"

_XGB_BASE = dict(
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
XGB_FINAL = dict(**_XGB_BASE, n_estimators=500)

DAY_GROUPS = {
    1: (1, 24),
    2: (25, 48),
    3: (49, 72),
    4: (73, 96),
    5: (97, 120),
}

WEATHER_FEATURES = [
    "fcst_wind_speed_10m",
    "fcst_wind_direction_10m",
    "fcst_wind_speed_100m",
    "fcst_wind_direction_100m",
    "fcst_shortwave_radiation",
    "fcst_cloud_cover",
    "fcst_temperature_2m",
    "fcst_pressure_msl",
]
TIME_FEATURES = [
    "horizon_h",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
]
PRICE_LAG_FEATURES = [
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_rolling_24h_mean",
]
FEATURES = TIME_FEATURES + WEATHER_FEATURES + PRICE_LAG_FEATURES

FEATURE_LABELS = {
    "fcst_wind_speed_100m": "Wind speed 100m",
    "fcst_shortwave_radiation": "Shortwave radiation",
    "fcst_cloud_cover": "Cloud cover",
    "price_lag_24h": "Price lag 24h",
    "fcst_temperature_2m": "Temperature 2m",
}
SELECTED_FEATURES = list(FEATURE_LABELS.keys())

DAY_COLORS = {
    1: "#1768AC",
    2: "#2F9C95",
    3: "#F2B134",
    4: "#F07167",
    5: "#8E5572",
}

SHAP_SAMPLE_N = 1000
COUNTERFACTUAL_SAMPLE_N = 600
COUNTERFACTUAL_STEP_FRACTION = 0.25
ALE_BINS = 20


def load_hourly_prices() -> pd.Series:
    raw = (
        pd.read_csv(DATA_DIR / "day_ahead_prices_dk1_raw.csv", parse_dates=["TimeDK"])
        .set_index("TimeDK")
        .sort_index()
    )
    return raw["DayAheadPriceEUR"].resample("h").mean()


def build_price_lags(df: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    price_map = prices.to_dict()
    unique_ts = pd.to_datetime(df["issue_time"].unique())

    rows = {}
    for ts in unique_ts:
        past_24 = [price_map.get(ts - pd.Timedelta(hours=h), np.nan) for h in range(1, 25)]
        rows[ts] = {
            "price_lag_24h": price_map.get(ts - pd.Timedelta(hours=24), np.nan),
            "price_lag_48h": price_map.get(ts - pd.Timedelta(hours=48), np.nan),
            "price_lag_168h": price_map.get(ts - pd.Timedelta(hours=168), np.nan),
            "price_rolling_24h_mean": np.nanmean(past_24),
        }

    lag_df = pd.DataFrame.from_dict(rows, orient="index")
    lag_df.index.name = "issue_time"
    return df.merge(lag_df.reset_index(), on="issue_time", how="left")


def prepare_dataset() -> pd.DataFrame:
    print("Loading forecast dataset...")
    df = pd.read_parquet(DATA_DIR / "forsoeg_dataset.parquet")
    df = df[df["region"] == REGION].copy()
    print(f"  {len(df):,} rows | region={REGION}")

    print("Loading hourly prices...")
    prices = load_hourly_prices()

    price_df = (
        prices.rename(TARGET_COL)
        .reset_index()
        .rename(columns={"TimeDK": "target_time"})
    )
    df = df.merge(price_df, on="target_time", how="inner")

    print("Building issue-time lag features...")
    df = build_price_lags(df, prices)
    df = df.dropna(subset=[TARGET_COL] + PRICE_LAG_FEATURES).reset_index(drop=True)
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])

    print(f"  Ready for validation: {len(df):,} rows")
    return df.sort_values(["issue_time", "horizon_h"]).reset_index(drop=True)


def fit_day_models(train_df: pd.DataFrame, params: dict) -> dict[int, xgb.XGBRegressor]:
    models: dict[int, xgb.XGBRegressor] = {}
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        mask = train_df["horizon_h"].between(h_lo, h_hi)
        model = xgb.XGBRegressor(**params)
        model.fit(train_df.loc[mask, FEATURES], train_df.loc[mask, TARGET_COL])
        models[day] = model
    return models


def predict_day_models(models: dict[int, xgb.XGBRegressor], df: pd.DataFrame) -> np.ndarray:
    preds = np.full(len(df), np.nan)
    horizons = df["horizon_h"].to_numpy()
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        mask = (horizons >= h_lo) & (horizons <= h_hi)
        if mask.any():
            preds[mask] = models[day].predict(df.loc[mask, FEATURES])
    return preds


def ensure_output_dir() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def sample_day_rows(df: pd.DataFrame, day: int, sample_n: int) -> pd.DataFrame:
    h_lo, h_hi = DAY_GROUPS[day]
    subset = df[df["horizon_h"].between(h_lo, h_hi)].copy()
    if len(subset) > sample_n:
        subset = subset.sample(sample_n, random_state=42)
    return subset.sort_values("issue_time").reset_index(drop=True)


def compute_1d_ale(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    feature: str,
    bins: int = ALE_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = X[feature].to_numpy()
    quantiles = np.unique(np.nanquantile(values, np.linspace(0, 1, bins + 1)))

    if len(quantiles) < 3:
        return np.array([]), np.array([]), np.array([])

    mids: list[float] = []
    local_effects: list[float] = []
    counts: list[int] = []

    # Bin membership uses left-open, right-closed intervals for stability.
    bin_ids = np.searchsorted(quantiles, values, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, len(quantiles) - 2)

    for idx in range(len(quantiles) - 1):
        lower = quantiles[idx]
        upper = quantiles[idx + 1]
        mask = bin_ids == idx

        if mask.sum() < 10 or lower == upper:
            continue

        X_lower = X.loc[mask, FEATURES].copy()
        X_upper = X.loc[mask, FEATURES].copy()
        X_lower[feature] = lower
        X_upper[feature] = upper

        diff = model.predict(X_upper) - model.predict(X_lower)

        mids.append((lower + upper) / 2.0)
        local_effects.append(float(np.mean(diff)))
        counts.append(int(mask.sum()))

    if not mids:
        return np.array([]), np.array([]), np.array([])

    cumulative = np.cumsum(local_effects)
    centered = cumulative - np.average(cumulative, weights=np.asarray(counts))
    return np.asarray(mids), centered, np.asarray(counts)


def save_shap_dependence_plots(
    df: pd.DataFrame,
    models: dict[int, xgb.XGBRegressor],
) -> None:
    print("Creating SHAP dependence plots...")
    for feature in SELECTED_FEATURES:
        fig, axes = plt.subplots(1, 5, figsize=(24, 4.5), sharey=False)

        for ax, day in zip(axes, DAY_GROUPS):
            sample = sample_day_rows(df, day, SHAP_SAMPLE_N)
            X_day = sample[FEATURES]
            dmatrix = xgb.DMatrix(X_day, feature_names=FEATURES)
            # Use XGBoost's native SHAP contribution output to avoid the
            # SHAP/XGBoost base_score parsing bug present in some version pairs.
            shap_values = models[day].get_booster().predict(dmatrix, pred_contribs=True)
            feature_idx = FEATURES.index(feature)

            ax.scatter(
                sample[feature],
                shap_values[:, feature_idx],
                s=9,
                alpha=0.35,
                color=DAY_COLORS[day],
                edgecolors="none",
            )
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
            ax.set_title(f"Day {day} (h{DAY_GROUPS[day][0]}-{DAY_GROUPS[day][1]})", fontsize=10)
            ax.set_xlabel(FEATURE_LABELS[feature], fontsize=9)
            if day == 1:
                ax.set_ylabel("SHAP value (EUR/MWh)", fontsize=9)

        fig.suptitle(f"SHAP dependence: {FEATURE_LABELS[feature]}", fontsize=13)
        fig.tight_layout()
        fig.savefig(
            VALIDATION_DIR / f"behavioral_shap_dependence_{feature}.png",
            dpi=150,
        )
        plt.close(fig)


def save_ale_plots(df: pd.DataFrame, models: dict[int, xgb.XGBRegressor]) -> None:
    print("Creating ALE plots...")
    ale_rows: list[dict[str, float | int | str]] = []

    for feature in SELECTED_FEATURES:
        fig, axes = plt.subplots(1, 5, figsize=(24, 4.5), sharey=False)

        for ax, day in zip(axes, DAY_GROUPS):
            sample = sample_day_rows(df, day, SHAP_SAMPLE_N)
            mids, ale_values, counts = compute_1d_ale(models[day], sample, feature)

            if len(mids) == 0:
                ax.set_title(f"Day {day} (insufficient variation)")
                continue

            ax.plot(mids, ale_values, color=DAY_COLORS[day], linewidth=2)
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
            ax.set_title(f"Day {day} (h{DAY_GROUPS[day][0]}-{DAY_GROUPS[day][1]})", fontsize=10)
            ax.set_xlabel(FEATURE_LABELS[feature], fontsize=9)
            if day == 1:
                ax.set_ylabel("ALE (EUR/MWh)", fontsize=9)

            for mid, ale_value, count in zip(mids, ale_values, counts):
                ale_rows.append(
                    {
                        "day_model": day,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        "x_mid": float(mid),
                        "ale_value": float(ale_value),
                        "bin_count": int(count),
                    }
                )

        fig.suptitle(f"ALE: {FEATURE_LABELS[feature]}", fontsize=13)
        fig.tight_layout()
        fig.savefig(
            VALIDATION_DIR / f"behavioral_ale_{feature}.png",
            dpi=150,
        )
        plt.close(fig)

    pd.DataFrame(ale_rows).to_csv(VALIDATION_DIR / "behavioral_ale_points.csv", index=False)


def counterfactual_step_bounds(values: pd.Series) -> tuple[float, float, float]:
    q05, q95 = np.nanquantile(values, [0.05, 0.95])
    step = COUNTERFACTUAL_STEP_FRACTION * (q95 - q05)
    return float(q05), float(q95), float(step)


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, fmt: str = ".1f") -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                label = "NA"
            else:
                label = format(value, fmt)
            ax.text(col, row, label, ha="center", va="center", fontsize=9, color="black")


def save_counterfactual_analysis(
    df: pd.DataFrame,
    models: dict[int, xgb.XGBRegressor],
) -> None:
    print("Running counterfactual perturbation tests...")
    rows: list[dict[str, float | int | str]] = []

    for day in DAY_GROUPS:
        sample = sample_day_rows(df, day, COUNTERFACTUAL_SAMPLE_N)
        X_base = sample[FEATURES].copy()
        base_pred = models[day].predict(X_base)

        for feature in SELECTED_FEATURES:
            q05, q95, step = counterfactual_step_bounds(sample[feature])
            if step <= 0:
                continue

            X_plus = X_base.copy()
            X_minus = X_base.copy()
            X_plus[feature] = np.clip(X_plus[feature] + step, q05, q95)
            X_minus[feature] = np.clip(X_minus[feature] - step, q05, q95)

            delta_plus = models[day].predict(X_plus) - base_pred
            delta_minus = models[day].predict(X_minus) - base_pred

            rows.append(
                {
                    "day_model": day,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "step_size": step,
                    "q05": q05,
                    "q95": q95,
                    "mean_delta_plus": float(np.mean(delta_plus)),
                    "median_delta_plus": float(np.median(delta_plus)),
                    "share_positive_plus": float(np.mean(delta_plus > 0)),
                    "mean_delta_minus": float(np.mean(delta_minus)),
                    "median_delta_minus": float(np.median(delta_minus)),
                    "share_positive_minus": float(np.mean(delta_minus > 0)),
                    "sample_n": int(len(sample)),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["feature", "day_model"]).reset_index(drop=True)
    summary.to_csv(VALIDATION_DIR / "counterfactual_perturbation_summary.csv", index=False)

    mean_plus = (
        summary.pivot(index="feature_label", columns="day_model", values="mean_delta_plus")
        .reindex([FEATURE_LABELS[f] for f in SELECTED_FEATURES])
    )
    share_plus = (
        summary.pivot(index="feature_label", columns="day_model", values="share_positive_plus")
        .reindex([FEATURE_LABELS[f] for f in SELECTED_FEATURES])
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im0 = axes[0].imshow(mean_plus.to_numpy(), cmap="RdBu_r", aspect="auto")
    axes[0].set_title("Mean prediction delta after +step perturbation")
    axes[0].set_xticks(range(len(mean_plus.columns)), mean_plus.columns)
    axes[0].set_yticks(range(len(mean_plus.index)), mean_plus.index)
    axes[0].set_xlabel("Day model")
    annotate_heatmap(axes[0], mean_plus.to_numpy(), fmt=".1f")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="EUR/MWh")

    im1 = axes[1].imshow(share_plus.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Share of sampled rows with positive delta after +step")
    axes[1].set_xticks(range(len(share_plus.columns)), share_plus.columns)
    axes[1].set_yticks(range(len(share_plus.index)), share_plus.index)
    axes[1].set_xlabel("Day model")
    annotate_heatmap(axes[1], share_plus.to_numpy(), fmt=".2f")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Share")

    fig.suptitle("Counterfactual perturbation diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(VALIDATION_DIR / "behavioral_counterfactual_perturbations.png", dpi=150)
    plt.close(fig)


def boundary_jump_rows(
    df: pd.DataFrame,
    boundary_left: int,
    boundary_right: int,
) -> pd.DataFrame:
    subset = df[df["horizon_h"].isin([boundary_left, boundary_right])].copy()
    pivot = subset.pivot_table(
        index="issue_time",
        columns="horizon_h",
        values=[TARGET_COL, "predicted"],
        aggfunc="first",
    )
    pivot.columns = [
        f"{column_name}_{int(horizon)}" for column_name, horizon in pivot.columns.to_flat_index()
    ]
    pivot = pivot.dropna().reset_index()
    pivot["predicted_jump"] = pivot[f"predicted_{boundary_right}"] - pivot[f"predicted_{boundary_left}"]
    pivot["actual_jump"] = pivot[f"{TARGET_COL}_{boundary_right}"] - pivot[f"{TARGET_COL}_{boundary_left}"]
    pivot["jump_gap"] = pivot["predicted_jump"] - pivot["actual_jump"]
    pivot["abs_jump_gap"] = np.abs(pivot["predicted_jump"]) - np.abs(pivot["actual_jump"])
    pivot["boundary_label"] = f"h{boundary_left}/h{boundary_right}"
    return pivot


def save_boundary_jump_analysis(df: pd.DataFrame) -> None:
    print("Creating boundary-jump diagnostics...")
    boundaries = [(24, 25), (48, 49), (72, 73), (96, 97)]
    jump_frames = [boundary_jump_rows(df, left, right) for left, right in boundaries]
    all_jumps = pd.concat(jump_frames, ignore_index=True)

    summary_rows = []
    for jump_df, (left, right) in zip(jump_frames, boundaries):
        summary_rows.append(
            {
                "boundary": f"h{left}/h{right}",
                "n_pairs": int(len(jump_df)),
                "predicted_jump_mean": float(jump_df["predicted_jump"].mean()),
                "predicted_jump_abs_mean": float(jump_df["predicted_jump"].abs().mean()),
                "actual_jump_mean": float(jump_df["actual_jump"].mean()),
                "actual_jump_abs_mean": float(jump_df["actual_jump"].abs().mean()),
                "jump_gap_mean": float(jump_df["jump_gap"].mean()),
                "abs_jump_gap_mean": float(jump_df["abs_jump_gap"].mean()),
            }
        )

    pd.DataFrame(summary_rows).to_csv(VALIDATION_DIR / "behavioral_boundary_jump_summary.csv", index=False)
    all_jumps.to_csv(VALIDATION_DIR / "behavioral_boundary_jump_pairs.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, jump_df, (left, right) in zip(axes.ravel(), jump_frames, boundaries):
        ax.boxplot(
            [jump_df["actual_jump"], jump_df["predicted_jump"]],
            labels=["Actual", "Predicted"],
            patch_artist=True,
            boxprops=dict(facecolor="#D9E6F2"),
            medianprops=dict(color="#B22222", linewidth=1.8),
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(
            (
                f"Boundary h{left}/h{right}\n"
                f"mean |jump| actual={jump_df['actual_jump'].abs().mean():.1f}, "
                f"pred={jump_df['predicted_jump'].abs().mean():.1f}"
            ),
            fontsize=10,
        )
        ax.set_ylabel("Jump (EUR/MWh)")

    fig.suptitle("Boundary-jump analysis across adjacent day models", fontsize=13)
    fig.tight_layout()
    fig.savefig(VALIDATION_DIR / "behavioral_boundary_jump_boxplots.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_output_dir()

    print("=" * 70)
    print("Behavioral validation for DK1 XGBoost forecaster")
    print("=" * 70)

    df = prepare_dataset()

    print("Training final five day-models...")
    models = fit_day_models(df, XGB_FINAL)

    print("Generating full-dataset predictions for continuity checks...")
    df = df.copy()
    df["predicted"] = predict_day_models(models, df)

    save_shap_dependence_plots(df, models)
    save_ale_plots(df, models)
    save_counterfactual_analysis(df, models)
    save_boundary_jump_analysis(df)

    print(f"Behavioral validation outputs saved to {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
