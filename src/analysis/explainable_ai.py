"""
explainable_ai.py  -  XAI plots for the DK1 XGBoost price forecasting model
=============================================================================
Loads saved model outputs and generates all explainability figures.

Requires outputs from src/models/XG_Boost_full_Res.py:
  outputs/model/predictions.parquet
  outputs/model/metrics.csv
  outputs/model/final_day_models.joblib

Outputs (outputs/model/):
  fig1_walk_forward_mae.png
  fig2_actual_vs_predicted.png
  fig3_mae_by_horizon.png
  fig4_feature_importance.png
  fig5_error_distribution.png
  fig6_single_forecast.png
  fig7_lime.png
  fig8_weather_error_distributions.png
  fig9_shap_bar.png
  fig10_shap_beeswarm_day1-5.png

Run:   python src/analysis/explainable_ai.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from lime import lime_tabular

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.XG_Boost_full_Res import (
    FEATURES, DAY_GROUPS, TARGET_COL, DATA_DIR, OUTPUT_DIR, COLORS,
    prepare_dataset,
)

EUR_DKK = 7.46  # fixed conversion rate; model outputs EUR/MWh → DKK/MWh

# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_wf_mae(metrics_df: pd.DataFrame, out: Path) -> None:
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


def plot_actual_vs_predicted(preds_df: pd.DataFrame, out: Path) -> None:
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
    fig.suptitle("Actual vs Predicted  --  last validation fold (4-week window)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig2_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    print("  fig2 saved")


def plot_mae_by_horizon(preds_df: pd.DataFrame, out: Path) -> None:
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


def plot_feature_importance(final_models: dict, out: Path) -> None:
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


def plot_error_distribution(preds_df: pd.DataFrame, out: Path) -> None:
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


def plot_single_forecast(preds_df: pd.DataFrame, out: Path) -> None:
    last_fold = preds_df["fold"].max()
    issue = preds_df[preds_df["fold"] == last_fold]["issue_time"].max()
    single = (
        preds_df[preds_df["issue_time"] == issue]
        .sort_values("horizon_h")
        .copy()
    )

    fig, ax = plt.subplots(figsize=(14, 4))
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
    ax.set_xlabel("Target time")
    ax.set_title(f"Full 120h forecast  --  issued at {issue.strftime('%Y-%m-%d %H:%M')}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.legend(ncol=7, fontsize=8)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out / "fig6_single_forecast.png", dpi=150)
    plt.close(fig)
    print("  fig6 saved")


def plot_lime(final_models: dict, df: pd.DataFrame, preds_df: pd.DataFrame, out: Path) -> None:
    last_fold = preds_df["fold"].max()
    issue     = preds_df[preds_df["fold"] == last_fold]["issue_time"].max()
    midpoints = {1: 12, 2: 36, 3: 60, 4: 84, 5: 108}

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    for ax, (day, (h_lo, h_hi)), color in zip(axes, DAY_GROUPS.items(), COLORS):
        model = final_models[day]
        mask  = df["horizon_h"].between(h_lo, h_hi)

        explainer = lime_tabular.LimeTabularExplainer(
            training_data = df.loc[mask, FEATURES].values,
            feature_names = FEATURES,
            mode          = "regression",
            random_state  = 42,
        )

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


def plot_weather_error_distributions(out: Path) -> None:
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

        ax.fill_between(h, sub["p05_error"], sub["p95_error"],
                        alpha=0.15, color="steelblue", label="p05-p95")
        ax.fill_between(h, sub["p25_error"], sub["p75_error"],
                        alpha=0.35, color="steelblue", label="p25-p75")
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

    for ax in axes[n:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=2)
    fig.suptitle("Weather forecast error distributions by horizon", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig8_weather_error_distributions.png", dpi=150)
    plt.close(fig)
    print("  fig8 saved")


def plot_shap(final_models: dict, df: pd.DataFrame, out: Path) -> None:
    import shap
    SAMPLE_N = 1500

    # fig9: mean |SHAP| bar charts
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (day, (h_lo, h_hi)), color in zip(axes, DAY_GROUPS.items(), COLORS):
        model  = final_models[day]
        mask   = df["horizon_h"].between(h_lo, h_hi)
        X_samp = df.loc[mask, FEATURES].sample(min(SAMPLE_N, mask.sum()), random_state=42)

        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X_samp) * EUR_DKK
        mean_abs    = np.abs(shap_vals).mean(axis=0)
        mean_signed = shap_vals.mean(axis=0)
        order       = np.argsort(mean_abs)[-12:]
        feat_labels = [FEATURES[i] for i in order]
        bar_colors  = [color if mean_signed[i] >= 0 else "#bdbdbd" for i in order]

        ax.barh(feat_labels, mean_abs[order], color=bar_colors)
        ax.set_title(f"Day {day} (h{h_lo}-{h_hi})", fontsize=10)
        ax.set_xlabel("Mean |SHAP| (DKK/MWh)", fontsize=8)
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

    # fig10: all 5 beeswarm plots in one figure
    fig10, axes10 = plt.subplots(1, 5, figsize=(40, 8))
    for ax, (day, (h_lo, h_hi)) in zip(axes10, DAY_GROUPS.items()):
        model  = final_models[day]
        mask   = df["horizon_h"].between(h_lo, h_hi)
        X_samp = df.loc[mask, FEATURES].sample(min(SAMPLE_N, mask.sum()), random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_expl = explainer(X_samp)

        plt.sca(ax)
        shap.summary_plot(
            shap_expl.values * EUR_DKK, X_samp,
            feature_names=FEATURES,
            max_display=12,
            show=False,
            plot_size=None,
        )
        ax.set_title(f"Day {day} (h{h_lo}-{h_hi})", fontsize=11)
        ax.set_xlabel("SHAP value (DKK/MWh impact on prediction)", fontsize=8)

    fig10.suptitle("SHAP beeswarm  --  all day models", fontsize=13, y=1.01)
    fig10.tight_layout()
    fig10.savefig(out / "fig10_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig10)
    print("  fig10 saved")


def export_shap_for_dashboard(final_models: dict, df: pd.DataFrame, out: Path) -> None:
    """Export SHAP values for Day 1 to parquet — consumed by the Streamlit dashboard."""
    import shap as _shap
    mask_d1 = df["horizon_h"].between(1, 24)
    X_d1 = (
        df.loc[mask_d1, FEATURES]
        .sample(min(500, mask_d1.sum()), random_state=42)
        .reset_index(drop=True)
    )
    explainer  = _shap.TreeExplainer(final_models[1])
    shap_vals  = explainer.shap_values(X_d1)
    pd.DataFrame(shap_vals, columns=FEATURES).to_parquet(
        out / "shap_day1_values.parquet", index=False
    )
    X_d1.to_parquet(out / "shap_day1_features.parquet", index=False)
    print(f"  SHAP dashboard data saved -> {out}/shap_day1_*.parquet")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Explainable AI  --  DK1 XGBoost price forecasting")
    print("=" * 60)

    print("\nLoading model outputs...")
    preds_df     = pd.read_parquet(OUTPUT_DIR / "predictions.parquet")
    metrics_df   = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    final_models = joblib.load(OUTPUT_DIR / "final_day_models.joblib")
    print(f"  predictions: {len(preds_df):,} rows")
    print(f"  metrics: {len(metrics_df)} folds")
    print(f"  models: {list(final_models.keys())}")

    print("\nLoading dataset (needed for LIME and SHAP)...")
    df = prepare_dataset()

    print("\nGenerating plots...")
    plot_wf_mae(metrics_df, OUTPUT_DIR)
    plot_actual_vs_predicted(preds_df, OUTPUT_DIR)
    plot_mae_by_horizon(preds_df, OUTPUT_DIR)
    plot_feature_importance(final_models, OUTPUT_DIR)
    plot_error_distribution(preds_df, OUTPUT_DIR)
    plot_single_forecast(preds_df, OUTPUT_DIR)
    plot_lime(final_models, df, preds_df, OUTPUT_DIR)
    plot_weather_error_distributions(OUTPUT_DIR)
    plot_shap(final_models, df, OUTPUT_DIR)

    print("\nExporting SHAP data for dashboard...")
    export_shap_for_dashboard(final_models, df, OUTPUT_DIR)

    print(f"\nAll plots and exports saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
