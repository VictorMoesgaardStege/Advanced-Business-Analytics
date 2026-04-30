"""
explainable_ai.py  -  XAI plots for the DK1 XGBoost price forecasting model
=============================================================================
Notebook-callable functions for all explainability figures.
When called without `out`, each function displays the plot inline (plt.show()).
Pass out=Path("outputs/model") to save to PNG files instead.

Requires outputs from src/models/XG_Boost_full_Res.py:
  outputs/model/predictions.parquet
  outputs/model/metrics.csv
  outputs/model/final_day_models.joblib

Quick start (notebook)
----------------------
    import joblib, pandas as pd
    from src.analysis.explainable_ai import (
        plot_wf_mae, plot_feature_importance, plot_error_distribution,
        plot_lime, plot_shap, export_shap_for_dashboard,
        make_xai_forecast_summary, make_dashboard_shap_summary,
        display_xai_artifacts,
    )
    from src.models.XG_Boost_full_Res import OUTPUT_DIR, prepare_dataset

    preds_df     = pd.read_parquet(OUTPUT_DIR / "predictions.parquet")
    metrics_df   = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    final_models = joblib.load(OUTPUT_DIR / "final_day_models.joblib")
    df           = prepare_dataset()

    plot_wf_mae(metrics_df)
    plot_feature_importance(final_models)
    plot_error_distribution(preds_df)
    plot_lime(final_models, df, preds_df)
    plot_shap(final_models, df)

Run (CLI, saves all PNGs):
    python src/analysis/explainable_ai.py
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

EUR_DKK = 7.46  # model outputs EUR/MWh; multiply for DKK/MWh


# -- Notebook helpers -----------------------------------------------------------

def make_xai_forecast_summary(
    metrics_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    eur_dkk: float = EUR_DKK,
) -> pd.DataFrame:
    """Return a compact day-level forecast summary for the notebook storyline.

    The model is trained as five separate day models, so the summary keeps the
    same structure: one row per model/day. MAE is reported both in the native
    model unit (EUR/MWh) and in the user-facing dashboard unit (DKK/kWh).
    """
    rows = []

    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        sub = preds_df[preds_df["horizon_h"].between(h_lo, h_hi)].copy()
        sub = sub.dropna(subset=["predicted", TARGET_COL])
        errors = sub["predicted"] - sub[TARGET_COL]

        metric_col = f"day{day}_mae"
        mean_mae = (
            metrics_df[metric_col].mean()
            if metric_col in metrics_df
            else errors.abs().mean()
        )

        rows.append(
            {
                "day_model": f"Day {day}",
                "horizon_hours": f"{h_lo}-{h_hi}",
                "mean_walk_forward_mae_eur_mwh": mean_mae,
                "mean_walk_forward_mae_dkk_kwh": mean_mae * eur_dkk / 1000,
                "out_of_sample_bias_eur_mwh": errors.mean(),
                "n_predictions": int(len(sub)),
            }
        )

    return pd.DataFrame(rows)


def make_dashboard_shap_summary(
    out: Path = OUTPUT_DIR,
    top_n: int = 8,
    eur_dkk: float = EUR_DKK,
) -> pd.DataFrame:
    """Summarise the precomputed Day 1 SHAP export used by the dashboard."""
    out = Path(out)
    shap_path = out / "shap_day1_values.parquet"
    feature_path = out / "shap_day1_features.parquet"

    if not shap_path.exists() or not feature_path.exists():
        raise FileNotFoundError(
            "Missing SHAP dashboard files. Run export_shap_for_dashboard(...) "
            "or `python src/analysis/explainable_ai.py` first."
        )

    shap_values = pd.read_parquet(shap_path)
    feature_values = pd.read_parquet(feature_path)

    top_features = shap_values.abs().mean().sort_values(ascending=False).head(top_n)
    summary = pd.DataFrame(
        {
            "feature": top_features.index,
            "mean_abs_shap_dkk_kwh": top_features.values * eur_dkk / 1000,
            "mean_signed_shap_dkk_kwh": (
                shap_values[top_features.index].mean().values * eur_dkk / 1000
            ),
            "sample_mean_feature_value": feature_values[top_features.index].mean().values,
        }
    )
    return summary


def display_xai_artifacts(
    filenames: list[str] | tuple[str, ...] | None = None,
    out: Path = OUTPUT_DIR,
    width: int | None = None,
) -> None:
    """Display saved XAI PNG artifacts in a notebook without recomputing them."""
    from IPython.display import Image, display

    out = Path(out)
    if filenames is None:
        filenames = (
            "fig4_feature_importance.png",
            "fig7_lime.png",
            "fig9_shap_bar.png",
            "fig10_shap_beeswarm.png",
        )

    for filename in filenames:
        path = out / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python src/analysis/explainable_ai.py` first."
            )
        display(Image(filename=str(path), width=width))


def plot_shap_explanations(
    out: Path = OUTPUT_DIR,
    width: int | None = None,
) -> None:
    """Notebook wrapper for the saved SHAP bar and beeswarm figures."""
    display_xai_artifacts(
        ("fig9_shap_bar.png", "fig10_shap_beeswarm.png"),
        out=out,
        width=width,
    )


def plot_lime_explanations(
    out: Path = OUTPUT_DIR,
    width: int | None = None,
) -> None:
    """Notebook wrapper for the saved LIME local explanation figure."""
    display_xai_artifacts(("fig7_lime.png",), out=out, width=width)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_wf_mae(
    metrics_df: pd.DataFrame,
    out: Path | None = None,
) -> None:
    """Fig 1 — Walk-forward MAE per day model across folds.

    Parameters
    ----------
    metrics_df : metrics.csv loaded as DataFrame
    out        : directory to save fig1_walk_forward_mae.png; None = plt.show()
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fold_end = pd.to_datetime(metrics_df["fold_end"])
    for day, color in zip(range(1, 6), COLORS):
        col = f"day{day}_mae"
        if col in metrics_df:
            ax.plot(fold_end, metrics_df[col],
                    marker="o", label=f"Day {day} (h{(day-1)*24+1}-{day*24})", color=color)
    ax.set(xlabel="Training end", ylabel="MAE (EUR/MWh)",
           title="Walk-forward MAE by day model")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    if out is not None:
        fig.savefig(out / "fig1_walk_forward_mae.png", dpi=150)
        plt.close(fig)
        print("  fig1 saved")
    else:
        plt.show()


def plot_feature_importance(
    final_models: dict,
    out: Path | None = None,
) -> None:
    """Fig 4 — Top-12 XGBoost feature importances per day model.

    Parameters
    ----------
    final_models : dict {day: XGBRegressor} loaded from final_day_models.joblib
    out          : directory to save fig4_feature_importance.png; None = plt.show()
    """
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
    if out is not None:
        fig.savefig(out / "fig4_feature_importance.png", dpi=150)
        plt.close(fig)
        print("  fig4 saved")
    else:
        plt.show()


def plot_error_distribution(
    preds_df: pd.DataFrame,
    out: Path | None = None,
) -> None:
    """Fig 5 — Prediction error histograms per day model (all folds).

    Parameters
    ----------
    preds_df : predictions.parquet loaded as DataFrame
    out      : directory to save fig5_error_distribution.png; None = plt.show()
    """
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
    if out is not None:
        fig.savefig(out / "fig5_error_distribution.png", dpi=150)
        plt.close(fig)
        print("  fig5 saved")
    else:
        plt.show()


def plot_lime(
    final_models: dict,
    df: pd.DataFrame,
    preds_df: pd.DataFrame,
    out: Path | None = None,
    issue_time: str | pd.Timestamp | None = None,
) -> None:
    """Fig 7 — LIME local explanations for one forecast issue time.

    Parameters
    ----------
    final_models : dict {day: XGBRegressor}
    df           : full dataset from prepare_dataset()
    preds_df     : predictions.parquet loaded as DataFrame
    out          : directory to save fig7_lime.png; None = plt.show()
    issue_time   : which issue time to explain (default: latest in last fold)
    """
    if issue_time is not None:
        issue = pd.Timestamp(issue_time)
    else:
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
            data_row     = X_instance,
            predict_fn   = model.predict,
            num_features = 10,
        )

        contributions = pd.Series(dict(exp.as_list())).sort_values()
        bar_colors    = [color if v >= 0 else "#bdbdbd" for v in contributions]
        contributions.plot(kind="barh", ax=ax, color=bar_colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"Day {day}  (h={h_mid})\n"
            f"Actual={actual_val:.1f}  Pred={pred_val:.1f} EUR/MWh",
            fontsize=9,
        )
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Contribution (EUR/MWh)", fontsize=8)

    fig.suptitle(
        f"LIME explanations  --  issued at {issue.strftime('%Y-%m-%d %H:%M')}",
        fontsize=12,
    )
    fig.tight_layout()
    if out is not None:
        fig.savefig(out / "fig7_lime.png", dpi=150)
        plt.close(fig)
        print("  fig7 saved")
    else:
        plt.show()


def plot_shap(
    final_models: dict,
    df: pd.DataFrame,
    out: Path | None = None,
    sample_n: int = 1500,
) -> None:
    """Fig 9 + Fig 10 — SHAP bar and beeswarm plots for all day models.

    Parameters
    ----------
    final_models : dict {day: XGBRegressor}
    df           : full dataset from prepare_dataset()
    out          : directory to save fig9/fig10 PNGs; None = plt.show()
    sample_n     : number of rows to sample per day model (default: 1500)
    """
    import shap

    # fig9: mean |SHAP| bar charts
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (day, (h_lo, h_hi)), color in zip(axes, DAY_GROUPS.items(), COLORS):
        model  = final_models[day]
        mask   = df["horizon_h"].between(h_lo, h_hi)
        X_samp = df.loc[mask, FEATURES].sample(min(sample_n, mask.sum()), random_state=42)

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
        fontsize=10,
    )
    fig.tight_layout()
    if out is not None:
        fig.savefig(out / "fig9_shap_bar.png", dpi=150)
        plt.close(fig)
        print("  fig9 saved")
    else:
        plt.show()

    # fig10: all 5 beeswarm plots in one figure
    fig10, axes10 = plt.subplots(1, 5, figsize=(40, 8))
    for ax, (day, (h_lo, h_hi)) in zip(axes10, DAY_GROUPS.items()):
        model  = final_models[day]
        mask   = df["horizon_h"].between(h_lo, h_hi)
        X_samp = df.loc[mask, FEATURES].sample(min(sample_n, mask.sum()), random_state=42)

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
    if out is not None:
        fig10.savefig(out / "fig10_shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close(fig10)
        print("  fig10 saved")
    else:
        plt.show()


def export_shap_for_dashboard(
    final_models: dict,
    df: pd.DataFrame,
    out: Path = OUTPUT_DIR,
) -> None:
    """Export SHAP values for Day 1 to parquet — consumed by the Streamlit dashboard.

    Parameters
    ----------
    final_models : dict {day: XGBRegressor}
    df           : full dataset from prepare_dataset()
    out          : directory to write shap_day1_*.parquet (default: OUTPUT_DIR)
    """
    import shap as _shap
    mask_d1 = df["horizon_h"].between(1, 24)
    X_d1 = (
        df.loc[mask_d1, FEATURES]
        .sample(min(500, mask_d1.sum()), random_state=42)
        .reset_index(drop=True)
    )
    explainer = _shap.TreeExplainer(final_models[1])
    shap_vals = explainer.shap_values(X_d1)
    pd.DataFrame(shap_vals, columns=FEATURES).to_parquet(
        out / "shap_day1_values.parquet", index=False
    )
    X_d1.to_parquet(out / "shap_day1_features.parquet", index=False)
    print(f"  SHAP dashboard data saved -> {out}/shap_day1_*.parquet")


# ── Main (CLI — saves all PNGs) ────────────────────────────────────────────────

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
    plot_wf_mae(metrics_df, out=OUTPUT_DIR)
    plot_feature_importance(final_models, out=OUTPUT_DIR)
    plot_error_distribution(preds_df, out=OUTPUT_DIR)
    plot_lime(final_models, df, preds_df, out=OUTPUT_DIR)
    plot_shap(final_models, df, out=OUTPUT_DIR)

    print("\nExporting SHAP data for dashboard...")
    export_shap_for_dashboard(final_models, df, out=OUTPUT_DIR)

    print(f"\nAll plots and exports saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
