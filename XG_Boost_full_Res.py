"""
FORSOEG_MODEL.py  -  XGBoost spot price forecasting for DK1
============================================================
5 XGBoost models, one per forecast day:
  Day 1: h=1-24    Day 2: h=25-48    Day 3: h=49-72
  Day 4: h=73-96   Day 5: h=97-120

Walk-forward cross-validation with expanding training window.

Price data covers 2021 onwards. Walk-forward uses a 12-month initial
training window with quarterly steps, giving ~10-15 folds across 4 years.

Features (all anchored to issue_time -- no leakage):
  - Simulated weather forecasts  (fcst_*)
  - Time features from target_time  (deterministic, known in advance)
  - horizon_h
  - Spot price lags at t-24h, t-48h, t-168h, rolling mean t-24h

Run:   python FORSOEG_MODEL.py
Output: outputs/model/
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from lime import lime_tabular

# ── Configuration ──────────────────────────────────────────────────────────────
REGION               = "DK1_west"
TARGET_COL           = "DayAheadPriceEUR"
DATA_DIR             = Path("data")
OUTPUT_DIR           = Path("outputs/model")

INITIAL_TRAIN_MONTHS = 12   # 1 year -- captures full seasonality before first test fold
STEP_MONTHS          = 3    # quarterly steps -- ~12-15 folds across 4 years of data

_XGB_BASE = dict(
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)
XGB_WF    = dict(**_XGB_BASE, n_estimators=300)
XGB_FINAL = dict(**_XGB_BASE, n_estimators=500)

_XGB_Q_BASE = dict(**_XGB_BASE, objective="reg:quantileerror", n_estimators=300)
XGB_Q40 = dict(**_XGB_Q_BASE, quantile_alpha=0.40)
XGB_Q60 = dict(**_XGB_Q_BASE, quantile_alpha=0.60)

DAY_GROUPS = {1: (1, 24), 2: (25, 48), 3: (49, 72), 4: (73, 96), 5: (97, 120)}

WEATHER_FEATURES = [
    "fcst_wind_speed_10m",     "fcst_wind_direction_10m",
    "fcst_wind_speed_100m",    "fcst_wind_direction_100m",
    "fcst_shortwave_radiation","fcst_cloud_cover",
    "fcst_temperature_2m",     "fcst_pressure_msl",
]
TIME_FEATURES = [
    "horizon_h",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
]
PRICE_LAG_FEATURES = [
    "price_lag_24h", "price_lag_48h", "price_lag_168h",
    "price_rolling_24h_mean",
]
FEATURES = TIME_FEATURES + WEATHER_FEATURES + PRICE_LAG_FEATURES

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


# ── Data preparation ───────────────────────────────────────────────────────────

def load_hourly_prices() -> pd.Series:
    raw = (
        pd.read_csv(DATA_DIR / "day_ahead_prices_dk1_raw.csv", parse_dates=["TimeDK"])
        .set_index("TimeDK")
        .sort_index()
    )
    return raw["DayAheadPriceEUR"].resample("h").mean()


def build_price_lags(df: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """
    Price lags are computed relative to issue_time, so they are always
    known when the forecast is issued regardless of the target horizon.
    """
    price_map  = prices.to_dict()
    unique_ts  = pd.to_datetime(df["issue_time"].unique())

    rows = {}
    for t in unique_ts:
        past_24 = [price_map.get(t - pd.Timedelta(hours=h), np.nan) for h in range(1, 25)]
        rows[t] = {
            "price_lag_24h":          price_map.get(t - pd.Timedelta(hours=24),  np.nan),
            "price_lag_48h":          price_map.get(t - pd.Timedelta(hours=48),  np.nan),
            "price_lag_168h":         price_map.get(t - pd.Timedelta(hours=168), np.nan),
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

    print("Loading prices (resampling to hourly)...")
    prices = load_hourly_prices()
    print(f"  Price range: {prices.index.min().date()} to {prices.index.max().date()}")

    # Join target price on target_time (inner = only keep hours with a known price)
    price_df = (
        prices
        .rename(TARGET_COL)
        .reset_index()
        .rename(columns={"TimeDK": "target_time"})
    )
    df = df.merge(price_df, on="target_time", how="inner")
    print(f"  After price join: {len(df):,} rows")

    print("Adding price lag features...")
    df = build_price_lags(df, prices)
    df = df.dropna(subset=[TARGET_COL] + PRICE_LAG_FEATURES).reset_index(drop=True)
    print(f"  After lag NaN drop: {len(df):,} rows")
    print(f"  Usable range: {df['issue_time'].min().date()} to {df['issue_time'].max().date()}")

    return df.sort_values("issue_time").reset_index(drop=True)


# ── Model training & prediction ────────────────────────────────────────────────

def fit_day_models(train_df: pd.DataFrame, params: dict) -> dict:
    models = {}
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        mask = train_df["horizon_h"].between(h_lo, h_hi)
        m = xgb.XGBRegressor(**params)
        m.fit(train_df.loc[mask, FEATURES], train_df.loc[mask, TARGET_COL])
        models[day] = m
    return models


def predict_day_models(models: dict, test_df: pd.DataFrame) -> np.ndarray:
    preds = np.full(len(test_df), np.nan)
    h     = test_df["horizon_h"].values
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        mask = (h >= h_lo) & (h <= h_hi)
        if mask.any():
            preds[mask] = models[day].predict(test_df[mask][FEATURES])
    return preds


# ── Walk-forward validation ────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame):
    min_t = df["issue_time"].min()
    max_t = df["issue_time"].max()

    folds, t = [], min_t + pd.DateOffset(months=INITIAL_TRAIN_MONTHS)
    while t + pd.DateOffset(months=STEP_MONTHS) <= max_t:
        folds.append((pd.Timestamp(t), pd.Timestamp(t + pd.DateOffset(months=STEP_MONTHS))))
        t += pd.DateOffset(months=STEP_MONTHS)
    print(f"  {len(folds)} folds | initial={INITIAL_TRAIN_MONTHS}m, step={STEP_MONTHS}m")

    all_preds    = []
    fold_metrics = []

    for i, (train_end, test_end) in enumerate(folds):
        train = df[df["issue_time"] <  train_end]
        test  = df[(df["issue_time"] >= train_end) & (df["issue_time"] < test_end)]
        if train.empty or test.empty:
            continue

        models     = fit_day_models(train, XGB_WF)
        models_q40 = fit_day_models(train, XGB_Q40)
        models_q60 = fit_day_models(train, XGB_Q60)
        test_r = test.reset_index(drop=True)
        preds     = predict_day_models(models,     test_r)
        preds_q40 = predict_day_models(models_q40, test_r)
        preds_q60 = predict_day_models(models_q60, test_r)

        result = test_r[["issue_time", "target_time", "horizon_h", TARGET_COL]].copy()
        result["predicted"]  = preds
        result["pred_q40"]   = preds_q40
        result["pred_q60"]   = preds_q60
        result["fold"]       = i
        result["fold_end"]   = train_end
        all_preds.append(result)

        row     = {"fold": i, "fold_end": train_end, "n_train": len(train)}
        summary = []
        for day, (h_lo, h_hi) in DAY_GROUPS.items():
            m = result["horizon_h"].between(h_lo, h_hi)
            if m.any():
                mae = mean_absolute_error(result.loc[m, TARGET_COL], result.loc[m, "predicted"])
                row[f"day{day}_mae"] = mae
                summary.append(f"d{day}={mae:.1f}")
        fold_metrics.append(row)
        print(f"  Fold {i+1} | train_end={train_end.date()} n={len(train):,} | {', '.join(summary)}")

    return pd.concat(all_preds, ignore_index=True), pd.DataFrame(fold_metrics)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_wf_mae(metrics_df, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    for day, color in zip(range(1, 6), COLORS):
        col = f"day{day}_mae"
        if col in metrics_df:
            ax.plot(metrics_df["fold_end"], metrics_df[col],
                    marker="o", label=f"Day {day} (h{(day-1)*24+1}-{day*24})", color=color)
    ax.set(xlabel="Training end", ylabel="MAE (EUR/MWh)",
           title="Walk-forward MAE by day model")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "fig1_walk_forward_mae.png", dpi=150)
    plt.close(fig)
    print("  fig1 saved")


def plot_actual_vs_predicted(preds_df, out):
    # Last fold, last 4 weeks visible
    last_fold = preds_df["fold"].max()
    sub = preds_df[preds_df["fold"] == last_fold].copy()
    end_t   = sub["target_time"].max()
    start_t = end_t - pd.Timedelta(weeks=4)
    sub = sub[sub["target_time"] >= start_t]

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    for ax, (day, (h_lo, h_hi)), color in zip(axes, DAY_GROUPS.items(), COLORS):
        grp = (
            sub[sub["horizon_h"].between(h_lo, h_hi)]
            .groupby("target_time")
            .agg(actual=(TARGET_COL, "mean"), predicted=("predicted", "mean"))
        )
        ax.plot(grp.index, grp["actual"],    color="black",  linewidth=1.2, label="Actual")
        ax.plot(grp.index, grp["predicted"], color=color,    linewidth=1.2,
                linestyle="--", label="Predicted")
        mae = mean_absolute_error(grp["actual"], grp["predicted"])
        ax.set_title(f"Day {day}  (h{h_lo}-{h_hi})   MAE = {mae:.1f} EUR/MWh", fontsize=10)
        ax.set_ylabel("EUR/MWh")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Target time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.suptitle("Actual vs Predicted  --  last validation fold (4-week window)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig2_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    print("  fig2 saved")


def plot_mae_by_horizon(preds_df, out):
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
    fig.savefig(out / "fig3_mae_by_horizon.png", dpi=150)
    plt.close(fig)
    print("  fig3 saved")


def plot_feature_importance(final_models, out):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (day, model), color in zip(axes, final_models.items(), COLORS):
        imp = (
            pd.Series(model.feature_importances_, index=FEATURES)
            .sort_values()
            .tail(12)
        )
        imp.plot(kind="barh", ax=ax, color=color)
        ax.set_title(f"Day {day}", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Importance")
    fig.suptitle("Feature importance  --  top 12 per day model", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig4_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  fig4 saved")


def plot_error_distribution(preds_df, out):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (day, (h_lo, h_hi)), color in zip(axes, DAY_GROUPS.items(), COLORS):
        sub    = preds_df[preds_df["horizon_h"].between(h_lo, h_hi)]
        errors = sub["predicted"] - sub[TARGET_COL]
        ax.hist(errors, bins=50, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"Day {day}\nMAE={errors.abs().mean():.1f}  bias={errors.mean():+.1f}")
        ax.set_xlabel("Error (EUR/MWh)")
    axes[0].set_ylabel("Count")
    fig.suptitle("Prediction error distributions  --  all folds", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig5_error_distribution.png", dpi=150)
    plt.close(fig)
    print("  fig5 saved")


def plot_lime(final_models, df, preds_df, out):
    """
    One subplot per day model. Each shows a LIME explanation for the
    midpoint prediction of the most recent issue time in the last fold.
    Midpoints: h=12 (day1), h=36 (day2), h=60 (day3), h=84 (day4), h=108 (day5).
    """
    last_fold = preds_df["fold"].max()
    issue     = preds_df[preds_df["fold"] == last_fold]["issue_time"].max()

    midpoints = {1: 12, 2: 36, 3: 60, 4: 84, 5: 108}

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    for ax, (day, (h_lo, h_hi)), color in zip(axes, DAY_GROUPS.items(), COLORS):
        model = final_models[day]

        # Background: all training rows for this day's horizon range
        mask       = df["horizon_h"].between(h_lo, h_hi)
        X_train_day = df.loc[mask, FEATURES].values

        explainer = lime_tabular.LimeTabularExplainer(
            training_data  = X_train_day,
            feature_names  = FEATURES,
            mode           = "regression",
            random_state   = 42,
        )

        # Instance to explain: midpoint horizon from the chosen issue time
        h_mid    = midpoints[day]
        instance = df[(df["issue_time"] == issue) & (df["horizon_h"] == h_mid)]
        if instance.empty:
            ax.set_title(f"Day {day}\n(no data)")
            continue

        X_instance = instance[FEATURES].values[0]
        actual_val = instance[TARGET_COL].values[0]
        pred_val   = model.predict(instance[FEATURES])[0]

        exp = explainer.explain_instance(
            data_row   = X_instance,
            predict_fn = model.predict,
            num_features = 10,
        )

        # Extract and sort by absolute contribution
        contributions = pd.Series(dict(exp.as_list())).sort_values()
        bar_colors    = [color if v >= 0 else "#bdbdbd" for v in contributions]

        contributions.plot(kind="barh", ax=ax, color=bar_colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"Day {day}  (h={h_mid})\n"
            f"Actual={actual_val:.1f}  Pred={pred_val:.1f} EUR/MWh",
            fontsize=9
        )
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Contribution (EUR/MWh)", fontsize=8)

    fig.suptitle(
        f"LIME explanations  --  issued at {issue.strftime('%Y-%m-%d %H:%M')}",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out / "fig7_lime.png", dpi=150)
    plt.close(fig)
    print("  fig7 saved")


def plot_single_forecast(preds_df, out):
    # Pick the most recent issue time in the last fold
    last_fold = preds_df["fold"].max()
    issue = preds_df[preds_df["fold"] == last_fold]["issue_time"].max()
    single = (
        preds_df[preds_df["issue_time"] == issue]
        .sort_values("horizon_h")
        .copy()
    )

    fig, ax = plt.subplots(figsize=(14, 4))

    # Shade day bands first so they sit behind the lines
    for day, (h_lo, h_hi), color in zip(DAY_GROUPS.keys(), DAY_GROUPS.values(), COLORS):
        ax.axvspan(
            single.loc[single["horizon_h"] == h_lo, "target_time"].values[0],
            single.loc[single["horizon_h"] == h_hi, "target_time"].values[0],
            alpha=0.08, color=color, label=f"Day {day}"
        )

    ax.plot(single["target_time"], single[TARGET_COL],
            color="black", linewidth=1.5, label="Actual")
    ax.plot(single["target_time"], single["predicted"],
            color="steelblue", linewidth=1.5, linestyle="--", label="Predicted")

    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("Target time (date and hour)")
    ax.set_title(f"Full 120h forecast  --  issued at {issue.strftime('%Y-%m-%d %H:%M')}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.legend(ncol=7, fontsize=8)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out / "fig6_single_forecast.png", dpi=150)
    plt.close(fig)
    print("  fig6 saved")


def plot_shap(final_models, df, out):
    """
    fig9  -- mean |SHAP| bar chart for all 5 day models in one figure.
             Bars are coloured by the direction of the mean effect
             (positive = pushes price up, negative = pushes price down).
    fig10 -- SHAP beeswarm per day model (one file each):
             fig10_shap_beeswarm_day1.png ... fig10_shap_beeswarm_day5.png
             Shows both magnitude and direction for every sample.
    """
    import shap
    SAMPLE_N = 1500   # rows per day model; enough for stable SHAP estimates

    # ── fig9: bar charts ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    for ax, (day, (h_lo, h_hi)), color in zip(axes, DAY_GROUPS.items(), COLORS):
        model   = final_models[day]
        mask    = df["horizon_h"].between(h_lo, h_hi)
        X_samp  = df.loc[mask, FEATURES].sample(
            min(SAMPLE_N, mask.sum()), random_state=42
        )

        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X_samp)          # (n, n_features)

        mean_abs    = np.abs(shap_vals).mean(axis=0)
        mean_signed = shap_vals.mean(axis=0)

        order       = np.argsort(mean_abs)[-12:]             # top 12
        feat_labels = [FEATURES[i] for i in order]
        bar_vals    = mean_abs[order]
        bar_colors  = [color if mean_signed[i] >= 0 else "#bdbdbd" for i in order]

        ax.barh(feat_labels, bar_vals, color=bar_colors)
        ax.set_title(f"Day {day} (h{h_lo}-{h_hi})", fontsize=10)
        ax.set_xlabel("Mean |SHAP| (EUR/MWh)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        "SHAP feature importance  --  colour = direction of mean effect  "
        "(solid = increases price / grey = decreases price)",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out / "fig9_shap_bar.png", dpi=150)
    plt.close(fig)
    print("  fig9 saved")

    # ── fig10: beeswarm per day model ─────────────────────────────────────────
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        model  = final_models[day]
        mask   = df["horizon_h"].between(h_lo, h_hi)
        X_samp = df.loc[mask, FEATURES].sample(
            min(SAMPLE_N, mask.sum()), random_state=42
        )

        explainer  = shap.TreeExplainer(model)
        shap_expl  = explainer(X_samp)                       # Explanation object

        fig_bees, ax_bees = plt.subplots(figsize=(9, 7))
        plt.sca(ax_bees)
        shap.summary_plot(
            shap_expl.values,
            X_samp,
            feature_names=FEATURES,
            max_display=12,
            show=False,
            plot_size=None,
        )
        ax_bees.set_title(
            f"SHAP beeswarm  --  Day {day} (h{h_lo}-{h_hi})", fontsize=11
        )
        ax_bees.set_xlabel("SHAP value (EUR/MWh impact on prediction)")
        fig_bees.tight_layout()
        fig_bees.savefig(out / f"fig10_shap_beeswarm_day{day}.png", dpi=150)
        plt.close(fig_bees)
        print(f"  fig10 day{day} saved")


def plot_weather_error_distributions(out):
    """
    For each weather variable, shows how forecast bias and spread grow
    with horizon (24, 48, 72, 96, 120 h).

    Bands:  p05-p95 (light)  and  p25-p75 (dark)
    Lines:  mean_error (solid)  and  p50_error / median (dashed)
    """
    err = pd.read_csv(DATA_DIR / "weather_error_distributions.csv")

    variables = err["forecast_variable"].unique()
    n         = len(variables)
    ncols     = 4
    nrows     = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    axes      = axes.flatten()

    for ax, var in zip(axes, variables):
        sub = err[err["forecast_variable"] == var].sort_values("horizon_hours")
        h   = sub["horizon_hours"].values

        # Percentile bands
        ax.fill_between(h, sub["p05_error"], sub["p95_error"],
                        alpha=0.15, color="steelblue", label="p05-p95")
        ax.fill_between(h, sub["p25_error"], sub["p75_error"],
                        alpha=0.35, color="steelblue", label="p25-p75")

        # Mean and median lines
        ax.plot(h, sub["mean_error"], color="steelblue",
                linewidth=2, marker="o", markersize=4, label="Mean error")
        ax.plot(h, sub["p50_error"], color="steelblue",
                linewidth=1.5, linestyle="--", marker="o", markersize=3,
                alpha=0.7, label="Median error")

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("Error")
        ax.set_xticks(h)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    # Single legend from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=2)

    fig.suptitle("Weather forecast error distributions by horizon", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig8_weather_error_distributions.png", dpi=150)
    plt.close(fig)
    print("  fig8 saved")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FORSOEG_MODEL  --  DK1 spot price forecasting")
    print("=" * 60)

    # 1. Prepare data
    df = prepare_dataset()
    print(f"\n  Features : {len(FEATURES)}")
    print(f"  Target   : {TARGET_COL}")

    # 2. Walk-forward CV
    print("\nWalk-forward validation...")
    preds_df, metrics_df = walk_forward(df)

    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    preds_df.to_parquet(OUTPUT_DIR / "predictions.parquet", index=False)

    print("\nOverall MAE (mean across folds):")
    for day in range(1, 6):
        col = f"day{day}_mae"
        if col in metrics_df:
            print(f"  Day {day} (h{(day-1)*24+1:3d}-{day*24}): "
                  f"{metrics_df[col].mean():.2f} EUR/MWh")

    # 3. Final models on all data
    print("\nTraining final models on full dataset...")
    final_models = {}
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        mask = df["horizon_h"].between(h_lo, h_hi)
        m = xgb.XGBRegressor(**XGB_FINAL)
        m.fit(df.loc[mask, FEATURES], df.loc[mask, TARGET_COL])
        final_models[day] = m
        print(f"  Day {day}: {mask.sum():,} rows")

    model_path = OUTPUT_DIR / "final_day_models.joblib"
    joblib.dump(final_models, model_path)
    print(f"  Final models saved → {model_path}")

    # Save SHAP data for Day 1 (consumed by the Streamlit dashboard)
    print("\nComputing SHAP values for Day 1 model (dashboard export)...")
    try:
        import shap as _shap
        mask_d1 = df["horizon_h"].between(1, 24)
        X_d1 = (
            df.loc[mask_d1, FEATURES]
            .sample(min(500, mask_d1.sum()), random_state=42)
            .reset_index(drop=True)
        )
        _explainer = _shap.TreeExplainer(final_models[1])
        _shap_vals = _explainer.shap_values(X_d1)
        pd.DataFrame(_shap_vals, columns=FEATURES).to_parquet(
            OUTPUT_DIR / "shap_day1_values.parquet", index=False
        )
        X_d1.to_parquet(OUTPUT_DIR / "shap_day1_features.parquet", index=False)
        print(f"  SHAP data saved → {OUTPUT_DIR}/shap_day1_*.parquet")
    except Exception as e:
        print(f"  SHAP export skipped: {e}")

    # 4. Plots
    print("\nGenerating plots...")
    plot_wf_mae(metrics_df, OUTPUT_DIR)
    plot_actual_vs_predicted(preds_df, OUTPUT_DIR)
    plot_mae_by_horizon(preds_df, OUTPUT_DIR)
    plot_feature_importance(final_models, OUTPUT_DIR)
    plot_error_distribution(preds_df, OUTPUT_DIR)
    plot_single_forecast(preds_df, OUTPUT_DIR)
    plot_lime(final_models, df, preds_df, OUTPUT_DIR)
    plot_shap(final_models, df, OUTPUT_DIR)
    plot_weather_error_distributions(OUTPUT_DIR)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
