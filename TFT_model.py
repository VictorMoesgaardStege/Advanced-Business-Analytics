"""
TFT_model.py  -  Temporal Fusion Transformer for DK1 spot price forecasting
============================================================================
Comparison model to XGBoost (XG_Boost_full_Res.py).
Output schema is identical to predictions.parquet so it can be swapped into
the Streamlit dashboard by changing one path constant.

Note on fairness: TFT uses *observed* weather at target_time as future
covariates (always available from weather_actuals_raw).  XGBoost uses
*simulated* NWP forecasts.  This gives TFT a slight information advantage;
keep that in mind when comparing accuracy.

Run:  python TFT_model.py
Outputs:
  outputs/model/predictions_tft.parquet   -- same schema as predictions.parquet
  outputs/model/fig_tft_*.png             -- comparison figures vs XGBoost
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# ── Config ────────────────────────────────────────────────────────────────────
REGION     = "DK1_west"
TARGET_COL = "DayAheadPriceEUR"
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs/model")

HORIZON        = 120    # 5-day forecast (hours)
ENCODER_LENGTH = 168    # 1 week of past prices fed to TFT encoder
MAX_STEPS      = 2000   # training budget — increase for GPU runs
N_WINDOWS      = 3      # walk-forward folds (fewer than XGBoost; training is slower)
STEP_SIZE      = 2160   # hours between folds (≈ 90 days = quarterly)

DAY_GROUPS = {1: (1, 24), 2: (25, 48), 3: (49, 72), 4: (73, 96), 5: (97, 120)}
XGB_COLOR  = "#2196F3"
TFT_COLOR  = "#FF9800"

FUTR_EXOG = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    "actual_wind_speed_100m", "actual_shortwave_radiation", "actual_temperature_2m",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prices() -> pd.Series:
    raw = (
        pd.read_csv(DATA_DIR / "day_ahead_prices_dk1_raw.csv", parse_dates=["TimeDK"])
        .set_index("TimeDK")
        .sort_index()
    )
    return raw[TARGET_COL].resample("h").mean()


def build_series(prices: pd.Series) -> pd.DataFrame:
    """
    Build a single hourly time series in neuralforecast long format.
    Weather actuals are extracted from forsoeg_dataset.parquet — each
    target_time has one unique observation regardless of issue_time.
    """
    fcst = pd.read_parquet(DATA_DIR / "forsoeg_dataset.parquet")
    fcst = fcst[fcst["region"] == REGION]

    actual_cols = [
        "actual_wind_speed_100m", "actual_shortwave_radiation", "actual_temperature_2m",
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    ]
    weather = (
        fcst.groupby("target_time")[actual_cols]
        .first()
        .reset_index()
        .rename(columns={"target_time": "ds"})
    )

    price_df = (
        prices.reset_index()
        .rename(columns={"TimeDK": "ds", TARGET_COL: "y"})
    )
    series = (
        weather.merge(price_df, on="ds", how="inner")
        .assign(unique_id="DK1")
        .sort_values("ds")
        .reset_index(drop=True)
    )
    series = series.dropna(subset=["y"] + FUTR_EXOG).reset_index(drop=True)
    return series[["unique_id", "ds", "y"] + FUTR_EXOG]


# ── Training & CV ─────────────────────────────────────────────────────────────

def train_and_cv(series: pd.DataFrame) -> pd.DataFrame:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT
    from neuralforecast.losses.pytorch import MQLoss

    model = TFT(
        h                         = HORIZON,
        input_size                = ENCODER_LENGTH,
        hidden_size               = 64,
        n_head                    = 4,
        learning_rate             = 1e-3,
        max_steps                 = MAX_STEPS,
        batch_size                = 64,
        futr_exog_list            = FUTR_EXOG,
        loss                      = MQLoss(quantiles=[0.1, 0.5, 0.9]),
        scaler_type               = "standard",
        early_stop_patience_steps = 100,
        random_seed               = 42,
        logger                    = False,   # TensorBoard incompatible with NumPy 2.x
    )

    nf = NeuralForecast(models=[model], freq="h")
    print(f"  Running {N_WINDOWS}-window walk-forward CV "
          f"(step={STEP_SIZE}h ≈ {STEP_SIZE//24}d, max_steps={MAX_STEPS})...")
    return nf.cross_validation(df=series, n_windows=N_WINDOWS, step_size=STEP_SIZE)


def reformat(cv: pd.DataFrame) -> pd.DataFrame:
    """Map neuralforecast CV output → XGBoost-compatible schema."""
    cv = cv.copy()
    cv["ds"]     = pd.to_datetime(cv["ds"])
    cv["cutoff"] = pd.to_datetime(cv["cutoff"])
    cv["horizon_h"] = (
        (cv["ds"] - cv["cutoff"]).dt.total_seconds() / 3600
    ).round().astype(int)

    # Locate quantile columns — neuralforecast naming varies by version
    tft_cols = [c for c in cv.columns if "TFT" in c]
    print(f"  TFT output columns: {tft_cols}")

    def find(needle: str):
        return next((c for c in tft_cols if needle in c), None)

    q10_col = find("0.1") or find("lo")
    q50_col = find("0.5") or find("median")
    q90_col = find("0.9") or find("hi")

    # Fallback: sort alphabetically and take first / mid / last
    if not all([q10_col, q50_col, q90_col]):
        s = sorted(tft_cols)
        q10_col, q50_col, q90_col = s[0], s[len(s) // 2], s[-1]

    # Sort row-wise to prevent quantile crossing
    q_arr = np.sort(
        np.column_stack([cv[q10_col], cv[q50_col], cv[q90_col]]),
        axis=1,
    )

    result = pd.DataFrame({
        "issue_time":  cv["cutoff"],
        "target_time": cv["ds"],
        "horizon_h":   cv["horizon_h"],
        TARGET_COL:    cv["y"],
        "predicted":   q_arr[:, 1],
        "pred_q10":    q_arr[:, 0],
        "pred_q90":    q_arr[:, 2],
        "fold":        pd.Categorical(cv["cutoff"]).codes,
    })
    return result[result["horizon_h"].between(1, HORIZON)].reset_index(drop=True)


# ── Comparison figures ────────────────────────────────────────────────────────

def _mae_by_day(df: pd.DataFrame) -> list[float]:
    return [
        mean_absolute_error(
            df.loc[df["horizon_h"].between(lo, hi), TARGET_COL],
            df.loc[df["horizon_h"].between(lo, hi), "predicted"],
        )
        for _, (lo, hi) in DAY_GROUPS.items()
    ]


def fig_mae_by_day(xgb: pd.DataFrame, tft: pd.DataFrame, out: Path) -> None:
    x, w = np.arange(5), 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, _mae_by_day(xgb), w, label="XGBoost (q50)", color=XGB_COLOR, alpha=0.85)
    ax.bar(x + w / 2, _mae_by_day(tft), w, label="TFT (q50)",     color=TFT_COLOR,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Day {d}\n(h{lo}–{hi})" for d, (lo, hi) in DAY_GROUPS.items()])
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("MAE by forecast day — XGBoost vs TFT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_tft_mae_by_day.png", dpi=150)
    plt.close(fig)
    print("  fig_tft_mae_by_day.png saved")


def fig_mae_by_horizon(xgb: pd.DataFrame, tft: pd.DataFrame, out: Path) -> None:
    def by_h(df):
        return df.groupby("horizon_h").apply(
            lambda g: mean_absolute_error(g[TARGET_COL], g["predicted"])
        )

    xgb_h = by_h(xgb)
    tft_h = by_h(tft)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(xgb_h.index, xgb_h.values, color=XGB_COLOR, linewidth=1.5, label="XGBoost")
    ax.plot(tft_h.index, tft_h.values, color=TFT_COLOR,  linewidth=1.5, label="TFT")
    for day, (lo, hi) in DAY_GROUPS.items():
        ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.05, color="grey")
    ylim = ax.get_ylim()
    for day, (lo, hi) in DAY_GROUPS.items():
        ax.text((lo + hi) / 2, ylim[1] * 0.96, f"D{day}",
                ha="center", fontsize=8, color="grey")
    ax.set_xlabel("Horizon (hours ahead)")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("MAE by forecast horizon — XGBoost vs TFT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_tft_mae_by_horizon.png", dpi=150)
    plt.close(fig)
    print("  fig_tft_mae_by_horizon.png saved")


def fig_interval_coverage(xgb: pd.DataFrame, tft: pd.DataFrame, out: Path) -> None:
    """Fraction of actuals inside [q10, q90] per horizon — target is 80%."""
    def coverage(df):
        return df.groupby("horizon_h").apply(
            lambda g: (
                (g[TARGET_COL] >= g["pred_q10"]) & (g[TARGET_COL] <= g["pred_q90"])
            ).mean()
        )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(coverage(xgb).index, coverage(xgb).values, color=XGB_COLOR, linewidth=1.5, label="XGBoost")
    ax.plot(coverage(tft).index, coverage(tft).values, color=TFT_COLOR,  linewidth=1.5, label="TFT")
    ax.axhline(0.80, color="black", linestyle="--", linewidth=0.8, label="Ideal 80%")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Horizon (hours ahead)")
    ax.set_ylabel("Coverage")
    ax.set_title("q10–q90 interval coverage — XGBoost vs TFT  (target: 80%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_tft_coverage.png", dpi=150)
    plt.close(fig)
    print("  fig_tft_coverage.png saved")


def fig_actual_vs_predicted(xgb: pd.DataFrame, tft: pd.DataFrame, out: Path) -> None:
    """Day 1 daily-average prices — last fold, 4-week window."""
    def daily_day1(df):
        return (
            df[df["horizon_h"].between(1, 24)]
            .assign(date=lambda d: d["target_time"].dt.normalize())
            .groupby("date")
            .agg(actual=(TARGET_COL, "mean"), predicted=("predicted", "mean"))
        )

    d_xgb = daily_day1(xgb[xgb["fold"] == xgb["fold"].max()])
    d_tft = daily_day1(tft[tft["fold"] == tft["fold"].max()])
    common = d_tft.index.intersection(d_xgb.index)
    end   = common.max()
    start = end - pd.Timedelta(weeks=4)
    common = common[(common >= start) & (common <= end)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(common, d_tft.loc[common, "actual"],    color="black",   linewidth=1.5, label="Actual")
    ax.plot(common, d_xgb.loc[common, "predicted"], color=XGB_COLOR, linewidth=1.2,
            linestyle="--", label="XGBoost")
    ax.plot(common, d_tft.loc[common, "predicted"], color=TFT_COLOR,  linewidth=1.2,
            linestyle="--", label="TFT")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Actual vs predicted — Day 1 daily avg — last fold (4-week window)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "fig_tft_actual_vs_pred.png", dpi=150)
    plt.close(fig)
    print("  fig_tft_actual_vs_pred.png saved")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TFT_model.py  --  DK1 spot price forecasting")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    prices = load_prices()
    series = build_series(prices)
    print(f"  Series: {len(series):,} hourly rows | "
          f"{series['ds'].min().date()} to {series['ds'].max().date()}")

    print("\n[2/4] Training TFT + walk-forward CV...")
    cv_raw = train_and_cv(series)

    print("\n[3/4] Reformatting predictions...")
    tft_preds = reformat(cv_raw)
    out_path  = OUTPUT_DIR / "predictions_tft.parquet"
    tft_preds.to_parquet(out_path, index=False)
    print(f"  {len(tft_preds):,} rows → {out_path}")

    print("\n[4/4] Generating comparison figures...")
    xgb_path = OUTPUT_DIR / "predictions.parquet"
    if not xgb_path.exists():
        print("  XGBoost predictions.parquet not found — run XG_Boost_full_Res.py first.")
        return

    xgb_preds = pd.read_parquet(xgb_path)
    # Align to the same target_time range as TFT
    xgb_preds = xgb_preds[xgb_preds["target_time"].isin(tft_preds["target_time"])].copy()

    fig_mae_by_day(xgb_preds, tft_preds, OUTPUT_DIR)
    fig_mae_by_horizon(xgb_preds, tft_preds, OUTPUT_DIR)
    fig_interval_coverage(xgb_preds, tft_preds, OUTPUT_DIR)
    fig_actual_vs_predicted(xgb_preds, tft_preds, OUTPUT_DIR)

    print("\nMAE summary (EUR/MWh):")
    print(f"  {'Day':<6} {'XGBoost':>10} {'TFT':>10} {'Δ (TFT−XGB)':>14}")
    for day, (lo, hi) in DAY_GROUPS.items():
        m_xgb = mean_absolute_error(
            xgb_preds.loc[xgb_preds["horizon_h"].between(lo, hi), TARGET_COL],
            xgb_preds.loc[xgb_preds["horizon_h"].between(lo, hi), "predicted"],
        )
        m_tft = mean_absolute_error(
            tft_preds.loc[tft_preds["horizon_h"].between(lo, hi), TARGET_COL],
            tft_preds.loc[tft_preds["horizon_h"].between(lo, hi), "predicted"],
        )
        print(f"  Day {day}  {m_xgb:>10.2f} {m_tft:>10.2f} {m_tft - m_xgb:>+14.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
