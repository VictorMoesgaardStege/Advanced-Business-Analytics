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
    plot_feature_importance(final_models, OUTPUT_DIR)
    plot_error_distribution(preds_df, OUTPUT_DIR)
    plot_lime(final_models, df, preds_df, OUTPUT_DIR)
    plot_shap(final_models, df, OUTPUT_DIR)

    print("\nExporting SHAP data for dashboard...")
    export_shap_for_dashboard(final_models, df, OUTPUT_DIR)

    print(f"\nAll plots and exports saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
