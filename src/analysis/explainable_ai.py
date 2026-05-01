"""
explainable_ai.py  -  Explainable AI plots for the DK1 XGBoost price forecasting model
=============================================================================
Notebook-callable functions for model explainability figures.
When called without `out`, each function displays the plot inline (plt.show()).
Pass out=Path("outputs/model") to save to PNG files instead.

Requires outputs from src/models/XG_Boost_full_Res.py:
  outputs/model/final_day_models.joblib

Quick start (notebook)
----------------------
    import joblib
    from src.analysis.explainable_ai import (
        plot_feature_importance,
        plot_lime, plot_shap, export_shap_for_dashboard,
        make_dashboard_shap_summary, summarize_shap_quantile_direction,
        explain_lime_for_instance, make_dashboard_lime_summary,
        display_xai_artifacts,
    )
    from src.models.XG_Boost_full_Res import OUTPUT_DIR, prepare_dataset

    final_models = joblib.load(OUTPUT_DIR / "final_day_models.joblib")
    df           = prepare_dataset()

    plot_feature_importance(final_models)
    plot_lime(final_models, df)
    plot_shap(final_models, df)

Run (CLI, saves all PNGs):
    python src/analysis/explainable_ai.py
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.XG_Boost_full_Res import (
    FEATURES, DAY_GROUPS, OUTPUT_DIR, COLORS,
    REGION, prepare_dataset,
)

EUR_DKK = 7.46  # model outputs EUR/MWh; multiply for DKK/MWh
MODEL_DATASET_FILE = ROOT / "data/model_dataset.parquet"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.parquet"
DEFAULT_LIME_ISSUE_TIME = pd.Timestamp("2025-12-15 00:00:00")


FEATURE_LABELS = {
    "horizon_h": "Forecast horizon",
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
    "month": "Month",
    "is_weekend": "Weekend flag",
    "sin_hour": "Hour cycle sine",
    "cos_hour": "Hour cycle cosine",
    "sin_doy": "Day-of-year cycle sine",
    "cos_doy": "Day-of-year cycle cosine",
    "fcst_wind_speed_10m": "Forecast wind speed 10m",
    "fcst_wind_dir_10m_sin": "Forecast wind direction 10m sine",
    "fcst_wind_dir_10m_cos": "Forecast wind direction 10m cosine",
    "fcst_wind_speed_100m": "Forecast wind speed 100m",
    "fcst_wind_dir_100m_sin": "Forecast wind direction 100m sine",
    "fcst_wind_dir_100m_cos": "Forecast wind direction 100m cosine",
    "fcst_shortwave_radiation": "Forecast solar radiation",
    "fcst_cloud_cover": "Forecast cloud cover",
    "fcst_temperature_2m": "Forecast temperature",
    "fcst_pressure_msl": "Forecast pressure",
    "price_lag_24h": "Price 24h ago",
    "price_lag_48h": "Price 48h ago",
    "price_lag_168h": "Price 168h ago",
    "price_rolling_24h_mean": "Recent 24h price average",
}


def feature_display_name(name: str) -> str:
    """Human-readable feature names for plots, notebooks, and LLM diagnostics."""
    return FEATURE_LABELS.get(
        name,
        name.replace("fcst_", "forecast_").replace("_", " ").title(),
    )


def _price_pressure(value: float, neutral_band: float = 0.0005) -> str:
    if pd.isna(value) or abs(value) < neutral_band:
        return "near-neutral"
    return "raises predicted prices" if value > 0 else "lowers predicted prices"


def summarize_shap_quantile_direction(
    shap_values: pd.DataFrame,
    feature_values: pd.DataFrame,
    top_n: int = 8,
    eur_dkk: float = EUR_DKK,
    quantile: float = 0.25,
) -> pd.DataFrame:
    """Summarise SHAP direction by comparing low and high feature-value quantiles.

    This avoids the misleading "mean signed SHAP" shortcut. For each globally
    important feature, it asks whether low feature values and high feature values
    tend to push predictions upward or downward.
    """
    if shap_values.empty or feature_values.empty:
        return pd.DataFrame()

    common = [c for c in shap_values.columns if c in feature_values.columns]
    if not common:
        return pd.DataFrame()

    top_features = (
        shap_values[common]
        .abs()
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    rows = []
    for feature in top_features:
        sv = pd.to_numeric(shap_values[feature], errors="coerce") * eur_dkk / 1000
        fv = pd.to_numeric(feature_values[feature], errors="coerce")
        valid = sv.notna() & fv.notna()
        if valid.sum() < 20:
            continue

        sv = sv[valid]
        fv = fv[valid]
        low_cut = fv.quantile(quantile)
        high_cut = fv.quantile(1 - quantile)
        low_sv = sv[fv <= low_cut]
        high_sv = sv[fv >= high_cut]
        if low_sv.empty or high_sv.empty:
            continue

        low_effect = float(low_sv.mean())
        high_effect = float(high_sv.mean())
        mean_abs = float(sv.abs().mean())
        directional_gap = abs(high_effect - low_effect)
        direction_summary = (
            "mixed directional pattern"
            if directional_gap < max(0.0005, mean_abs * 0.10)
            else (
                "low values push prices higher than high values"
                if low_effect > high_effect
                else "high values push prices higher than low values"
            )
        )

        rows.append(
            {
                "feature": feature,
                "display_feature": feature_display_name(feature),
                "mean_abs_shap_dkk_kwh": mean_abs,
                "low_feature_value_q": float(low_cut),
                "high_feature_value_q": float(high_cut),
                "low_value_mean_shap_dkk_kwh": low_effect,
                "high_value_mean_shap_dkk_kwh": high_effect,
                "low_value_pressure": _price_pressure(low_effect),
                "high_value_pressure": _price_pressure(high_effect),
                "direction_summary": direction_summary,
            }
        )

    return pd.DataFrame(rows)


def _day_for_horizon(horizon_h: int) -> int:
    for day, (h_lo, h_hi) in DAY_GROUPS.items():
        if h_lo <= int(horizon_h) <= h_hi:
            return day
    raise ValueError(f"horizon_h={horizon_h} is outside configured day groups")


def _format_lime_rule(rule: str) -> str:
    formatted = rule
    for feature in sorted(FEATURES, key=len, reverse=True):
        formatted = formatted.replace(feature, feature_display_name(feature))
    return formatted


def explain_lime_for_instance(
    final_models: dict,
    df: pd.DataFrame,
    issue_time: str | pd.Timestamp,
    horizon_h: int,
    top_n: int = 6,
    eur_dkk: float = EUR_DKK,
    sample_n: int = 5000,
    num_samples: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return compact local LIME contributions for one forecast issue/horizon.

    The dashboard uses this only for representative hours inside selected
    cheapest/most-expensive windows, not for every forecast hour.
    """
    if df.empty:
        return pd.DataFrame()

    horizon_h = int(horizon_h)
    day = _day_for_horizon(horizon_h)
    model = final_models.get(day)
    if model is None:
        return pd.DataFrame()

    issue_ts = pd.Timestamp(issue_time)
    issue_times = pd.to_datetime(df["issue_time"])
    instance = df[(issue_times == issue_ts) & (df["horizon_h"].astype(int) == horizon_h)]
    if instance.empty:
        return pd.DataFrame()

    h_lo, h_hi = DAY_GROUPS[day]
    train = df[df["horizon_h"].between(h_lo, h_hi)][FEATURES].dropna()
    if train.empty:
        return pd.DataFrame()
    if len(train) > sample_n:
        train = train.sample(sample_n, random_state=random_state)

    X_instance = instance[FEATURES].dropna().head(1)
    if X_instance.empty:
        return pd.DataFrame()

    from lime import lime_tabular

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=train.values,
        feature_names=FEATURES,
        mode="regression",
        random_state=random_state,
    )

    def predict_fn(values: np.ndarray) -> np.ndarray:
        X = pd.DataFrame(values, columns=FEATURES)
        return model.predict(X)

    exp = explainer.explain_instance(
        data_row=X_instance.iloc[0].values,
        predict_fn=predict_fn,
        num_features=top_n,
        num_samples=num_samples,
    )

    pred_eur_mwh = float(model.predict(X_instance)[0])
    rows = []
    for rule, contribution in exp.as_list():
        contribution = float(contribution)
        rows.append(
            {
                "day_model": day,
                "issue_time": issue_ts,
                "horizon_h": horizon_h,
                "target_time": pd.Timestamp(instance["target_time"].iloc[0]),
                "prediction_eur_mwh": pred_eur_mwh,
                "prediction_dkk_kwh": pred_eur_mwh * eur_dkk / 1000,
                "feature_rule": _format_lime_rule(rule),
                "contribution_eur_mwh": contribution,
                "contribution_dkk_kwh": contribution * eur_dkk / 1000,
                "direction": "raises forecast" if contribution >= 0 else "lowers forecast",
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.reindex(
            result["contribution_dkk_kwh"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    return result


def make_dashboard_hourly_forecast(
    predictions: pd.DataFrame,
    issue_time: str | pd.Timestamp | None = None,
    eur_dkk: float = EUR_DKK,
) -> pd.DataFrame:
    """Prepare the 120-hour forecast slice that the dashboard uses for LIME windows."""
    if predictions.empty:
        return pd.DataFrame()

    preds = predictions.copy()
    preds["issue_time"] = pd.to_datetime(preds["issue_time"])
    preds["target_time"] = pd.to_datetime(preds["target_time"])
    available = pd.to_datetime(preds["issue_time"].unique())
    if issue_time is None:
        issue_ts = pd.Timestamp(available.max())
    else:
        selected = pd.Timestamp(issue_time)
        issue_ts = min(available, key=lambda x: abs(x - selected))

    future = (
        preds[(preds["issue_time"] == issue_ts) & (preds["target_time"] > issue_ts)]
        .sort_values("target_time")
        .head(120)
        .copy()
    )
    if future.empty:
        return pd.DataFrame()

    future["pred_dkk"] = future["predicted"] * eur_dkk / 1000
    future["issue_time"] = issue_ts
    return future[["issue_time", "target_time", "horizon_h", "pred_dkk"]].reset_index(drop=True)


def select_extreme_forecast_windows(
    hourly_forecast: pd.DataFrame,
    window_hours: int = 3,
) -> list[dict]:
    """Select the cheapest and most expensive rolling windows used for local LIME."""
    required = {"issue_time", "target_time", "horizon_h", "pred_dkk"}
    if hourly_forecast.empty or not required.issubset(hourly_forecast.columns):
        return []

    df = hourly_forecast.dropna(subset=["pred_dkk"]).sort_values("target_time").reset_index(drop=True)
    if df.empty:
        return []

    window_hours = min(window_hours, len(df))
    df["_window_avg"] = df["pred_dkk"].rolling(window_hours, min_periods=window_hours).mean()
    valid = df.dropna(subset=["_window_avg"])
    if valid.empty:
        return []

    windows = []
    for label, idx in [
        ("Cheapest selected window", int(valid["_window_avg"].idxmin())),
        ("Most expensive selected window", int(valid["_window_avg"].idxmax())),
    ]:
        start_i = max(0, idx - window_hours + 1)
        end_i = idx
        mid_i = start_i + (end_i - start_i) // 2
        windows.append(
            {
                "window_label": label,
                "window_start": pd.Timestamp(df.loc[start_i, "target_time"]),
                "window_end": pd.Timestamp(df.loc[end_i, "target_time"]) + pd.Timedelta(hours=1),
                "window_avg_dkk_kwh": float(df.loc[start_i:end_i, "pred_dkk"].mean()),
                "issue_time": pd.Timestamp(df.loc[mid_i, "issue_time"]),
                "horizon_h": int(df.loc[mid_i, "horizon_h"]),
                "representative_time": pd.Timestamp(df.loc[mid_i, "target_time"]),
            }
        )
    return windows


def load_lime_dashboard_dataset(
    model_data_path: Path = MODEL_DATASET_FILE,
) -> pd.DataFrame:
    """Load the model-ready feature rows needed for dashboard-style LIME examples."""
    model_data_path = Path(model_data_path)
    if not model_data_path.exists():
        raise FileNotFoundError(f"Missing {model_data_path}")

    columns = list(dict.fromkeys(["region", "issue_time", "target_time", "horizon_h"] + FEATURES))
    df = pd.read_parquet(model_data_path, columns=columns)
    df = df[df["region"] == REGION].copy()
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def make_dashboard_lime_summary(
    final_models: dict,
    predictions_path: Path = PREDICTIONS_FILE,
    model_data_path: Path = MODEL_DATASET_FILE,
    issue_time: str | pd.Timestamp | None = None,
    window_hours: int = 3,
    top_n: int = 4,
    eur_dkk: float = EUR_DKK,
) -> pd.DataFrame:
    """Return local LIME contributions for the two windows passed to the dashboard LLM."""
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing {predictions_path}")

    predictions = pd.read_parquet(predictions_path)
    hourly_forecast = make_dashboard_hourly_forecast(
        predictions=predictions,
        issue_time=issue_time,
        eur_dkk=eur_dkk,
    )
    windows = select_extreme_forecast_windows(hourly_forecast, window_hours=window_hours)
    if not windows:
        return pd.DataFrame()

    lime_df = load_lime_dashboard_dataset(model_data_path)
    rows = []
    for window in windows:
        exp = explain_lime_for_instance(
            final_models=final_models,
            df=lime_df,
            issue_time=window["issue_time"],
            horizon_h=window["horizon_h"],
            top_n=top_n,
            eur_dkk=eur_dkk,
        )
        for _, row in exp.head(top_n).iterrows():
            item = row.to_dict()
            item.update(window)
            rows.append(item)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    first_cols = [
        "window_label",
        "window_start",
        "window_end",
        "window_avg_dkk_kwh",
        "representative_time",
        "horizon_h",
        "feature_rule",
        "direction",
        "contribution_dkk_kwh",
    ]
    ordered = first_cols + [c for c in result.columns if c not in first_cols]
    return result[ordered].reset_index(drop=True)


# -- Notebook helpers -----------------------------------------------------------

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

    return summarize_shap_quantile_direction(
        shap_values=shap_values,
        feature_values=feature_values,
        top_n=top_n,
        eur_dkk=eur_dkk,
    )


def display_xai_artifacts(
    filenames: list[str] | tuple[str, ...] | None = None,
    out: Path = OUTPUT_DIR,
    width: int | None = None,
) -> None:
    """Display saved Explainable AI PNG artifacts in a notebook without recomputing them."""
    from IPython.display import Image, display

    out = Path(out)
    if filenames is None:
        filenames = (
            "fig4_feature_importance.png",
            "fig7_lime.png",
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
    """Notebook wrapper for the saved SHAP beeswarm figures."""
    display_xai_artifacts(
        (
         "fig10_shap_beeswarm.png",),
        out=out,
        width=width,
    )


def plot_lime_explanations(
    out: Path = OUTPUT_DIR,
    width: int | None = None,
    issue_time: str | pd.Timestamp = DEFAULT_LIME_ISSUE_TIME,
) -> None:
    """Notebook wrapper that regenerates and displays the December 2025 LIME figure."""
    out = Path(out)
    final_models = joblib.load(out / "final_day_models.joblib")
    df = prepare_dataset()
    plot_lime(final_models, df, out=out, issue_time=issue_time)
    display_xai_artifacts(("fig7_lime.png",), out=out, width=width)


# ── Plots ──────────────────────────────────────────────────────────────────────

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
    import matplotlib.pyplot as plt

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


def plot_lime(
    final_models: dict,
    df: pd.DataFrame,
    out: Path | None = None,
    issue_time: str | pd.Timestamp | None = None,
) -> None:
    """Fig 7 — LIME local explanations for one forecast issue time.

    Parameters
    ----------
    final_models : dict {day: XGBRegressor}
    df           : full dataset from prepare_dataset()
    out          : directory to save fig7_lime.png; None = plt.show()
    issue_time   : which issue time to explain (default: latest in df)
    """
    import matplotlib.pyplot as plt
    from lime import lime_tabular

    issue_times = pd.to_datetime(df["issue_time"])
    if issue_time is not None:
        issue = pd.Timestamp(issue_time)
    else:
        issue = issue_times.max()

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
        instance = df[(issue_times == issue) & (df["horizon_h"] == h_mid)]
        if instance.empty:
            ax.set_title(f"Day {day}\n(no data)")
            continue

        X_instance = instance[FEATURES].values[0]
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
            f"Forecast={pred_val:.1f} EUR/MWh",
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
    import matplotlib.pyplot as plt
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
        order       = np.argsort(mean_abs)[-12:]
        feat_labels = [FEATURES[i] for i in order]
        bar_colors  = [color for _ in order]

        ax.barh(feat_labels, mean_abs[order], color=bar_colors)
        ax.set_title(f"Day {day} (h{h_lo}-{h_hi})", fontsize=10)
        ax.set_xlabel("Mean |SHAP| (DKK/MWh)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        "SHAP feature importance  --  bar length = mean absolute effect; "
        "use beeswarm plots for direction",
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
    out = Path(out)
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
    final_models = joblib.load(OUTPUT_DIR / "final_day_models.joblib")
    print(f"  models: {list(final_models.keys())}")

    print("\nLoading dataset (needed for LIME and SHAP)...")
    df = prepare_dataset()

    print("\nGenerating Explainable AI plots...")
    plot_feature_importance(final_models, out=OUTPUT_DIR)
    plot_lime(final_models, df, out=OUTPUT_DIR, issue_time=DEFAULT_LIME_ISSUE_TIME)
    plot_shap(final_models, df, out=OUTPUT_DIR)

    print("\nExporting SHAP data for dashboard...")
    export_shap_for_dashboard(final_models, df, out=OUTPUT_DIR)

    print(f"\nAll plots and exports saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
