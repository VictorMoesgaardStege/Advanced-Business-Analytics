"""
data_processing.py  —  Build the input dataset for the XGBoost model
=============================================================
Reads weather actuals and error distributions, simulates NWP-style
forecasts by adding horizon-scaled Gaussian noise, joins prices and
price lags, and saves the model-ready result as model_dataset.parquet.

Quick start
-----------
    from src.data.data_processing import build_model_dataset

    build_model_dataset()

Run from the repository root:
    python -m src.data.data_processing

Direct script execution also works from the repository root:
    python src/data/data_processing.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"

FORECAST_HORIZON_HOURS = 120
ISSUE_INTERVAL_HOURS   = 12

FORECAST_VAR_MAP = {
    "wind_speed_10m":      "wind_speed_10m",
    "wind_direction_10m":  "wind_direction_10m",
    "wind_speed_100m":     "wind_speed_120m",
    "wind_direction_100m": "wind_direction_120m",
    "shortwave_radiation": "shortwave_radiation",
    "cloud_cover":         "cloud_cover",
    "temperature_2m":      "temperature_2m",
    "pressure_msl":        "pressure_msl",
}

VAR_BOUNDS = {
    "wind_speed_10m":      (0,    None),
    "wind_direction_10m":  (0,    360),
    "wind_speed_100m":     (0,    None),
    "wind_direction_100m": (0,    360),
    "shortwave_radiation": (0,    None),
    "cloud_cover":         (0,    100),
    "temperature_2m":      (None, None),
    "pressure_msl":        (None, None),
}

WEATHER_VARS = list(FORECAST_VAR_MAP.keys())
HORIZONS     = np.arange(1, FORECAST_HORIZON_HOURS + 1, dtype=int)
TARGET_COL   = "DayAheadPriceEUR"
PRICE_LAG_FEATURES = [
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_rolling_24h_mean",
]


def _build_error_params(err_dist: pd.DataFrame) -> dict:
    known_horizons = sorted(err_dist["horizon_hours"].unique())
    params = {}
    for actual_var, fcst_var in FORECAST_VAR_MAP.items():
        sub      = err_dist[err_dist["forecast_variable"] == fcst_var].set_index("horizon_hours")
        kn_means = [sub.loc[h, "mean_error"] for h in known_horizons]
        kn_stds  = [sub.loc[h, "std_error"]  for h in known_horizons]
        mean_arr = np.interp(HORIZONS, known_horizons, kn_means)
        std_arr  = np.interp(HORIZONS, known_horizons, kn_stds)
        first_h  = known_horizons[0]
        below    = HORIZONS < first_h
        mean_arr[below] = kn_means[0] * (HORIZONS[below] / first_h)
        std_arr [below] = kn_stds [0] * (HORIZONS[below] / first_h)
        params[actual_var] = {"mean": mean_arr, "std": std_arr}
    return params


def _time_features(ts: pd.Timestamp) -> dict:
    return {
        "hour_of_day": ts.hour,
        "day_of_week": ts.dayofweek,
        "month":       ts.month,
        "is_weekend":  int(ts.dayofweek >= 5),
        "sin_hour":    np.sin(2 * np.pi * ts.hour / 24),
        "cos_hour":    np.cos(2 * np.pi * ts.hour / 24),
        "sin_doy":     np.sin(2 * np.pi * ts.dayofyear / 365),
        "cos_doy":     np.cos(2 * np.pi * ts.dayofyear / 365),
    }


def build_forecast_dataset(
    actuals_csv:    str | Path = DATA_DIR / "weather_actuals_raw.csv",
    error_dist_csv: str | Path = DATA_DIR / "weather_error_distributions.csv",
    output_path:    str | Path = DATA_DIR / "forecast_dataset.parquet",
    seed: int = 42,
) -> pd.DataFrame:
    """Build the synthetic forecast dataset used by XGBoost.

    Reads weather actuals and error distributions, simulates NWP-style
    forecasts by adding horizon-scaled Gaussian noise, and saves the
    result as a parquet file. Also writes a slim dashboard parquet
    (weather_dk1_dashboard.parquet) as a side effect.

    Parameters
    ----------
    actuals_csv    : path to weather_actuals_raw.csv
    error_dist_csv : path to weather_error_distributions.csv
    output_path    : where to write the parquet (default: data/forecast_dataset.parquet)
    seed           : random seed for reproducibility (default: 42)

    Returns
    -------
    pd.DataFrame with columns region, issue_time, target_time, horizon_h,
                 fcst_*, actual_*, and time features.
    """
    actuals_csv    = Path(actuals_csv)
    error_dist_csv = Path(error_dist_csv)
    output_path    = Path(output_path)

    print("[build_forecast_dataset] Loading inputs...")
    actuals = (
        pd.read_csv(actuals_csv, parse_dates=["TimeDK"])
        .rename(columns={"TimeDK": "time"})
        .sort_values(["region", "time"])
        .reset_index(drop=True)
    )
    err_dist = pd.read_csv(error_dist_csv)
    print(f"  Actuals: {len(actuals):,} rows | regions: {actuals['region'].unique().tolist()}")

    print("[build_forecast_dataset] Interpolating error parameters...")
    err_params = _build_error_params(err_dist)

    print(f"[build_forecast_dataset] Simulating forecasts "
          f"(every {ISSUE_INTERVAL_HOURS}h, horizon 1-{FORECAST_HORIZON_HOURS}h)...")
    rng     = np.random.default_rng(seed)
    records = []

    for region, rdf in actuals.groupby("region"):
        rdf         = rdf.set_index("time").sort_index()
        all_times   = rdf.index
        cutoff      = all_times[-1] - pd.Timedelta(hours=int(FORECAST_HORIZON_HOURS))
        valid_times = all_times[all_times <= cutoff]
        issue_times = valid_times[::ISSUE_INTERVAL_HOURS]
        print(f"  {region}: {len(issue_times)} issue times x {len(HORIZONS)} horizons "
              f"= {len(issue_times) * len(HORIZONS):,} rows")

        for issue_time in issue_times:
            for h_idx, h in enumerate(HORIZONS):
                target_time = issue_time + pd.Timedelta(hours=int(h))
                if target_time not in rdf.index:
                    continue
                actual_row = rdf.loc[target_time]
                rec = {
                    "region":      region,
                    "issue_time":  issue_time,
                    "target_time": target_time,
                    "horizon_h":   int(h),
                    **_time_features(target_time),
                }
                for var in WEATHER_VARS:
                    actual_val = float(actual_row[var])
                    if var == "shortwave_radiation" and actual_val < 0.2:
                        rec[f"fcst_{var}"]   = 0.0
                        rec[f"actual_{var}"] = actual_val
                        continue
                    noise = rng.normal(
                        loc=float(err_params[var]["mean"][h_idx]),
                        scale=max(float(err_params[var]["std"][h_idx]), 1e-9),
                    )
                    fcst_val = actual_val + noise
                    lo, hi = VAR_BOUNDS[var]
                    if lo is not None and hi is not None and "wind_direction" in var:
                        fcst_val = fcst_val % 360
                    else:
                        if lo is not None:
                            fcst_val = max(fcst_val, lo)
                        if hi is not None:
                            fcst_val = min(fcst_val, hi)
                    rec[f"fcst_{var}"]   = round(fcst_val, 4)
                    rec[f"actual_{var}"] = actual_val
                records.append(rec)

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  {len(df):,} rows x {df.shape[1]} cols saved -> {output_path}")

    # Slim dashboard file (DK1_west only)
    dashboard_cols = [
        "region", "issue_time", "target_time", "horizon_h",
        "fcst_wind_speed_100m",     "actual_wind_speed_100m",
        "fcst_shortwave_radiation", "actual_shortwave_radiation",
        "fcst_temperature_2m",      "actual_temperature_2m",
    ]
    dash_df   = df.loc[df["region"] == "DK1_west", dashboard_cols].copy()
    dash_path = DATA_DIR / "weather_dk1_dashboard.parquet"
    dash_df.to_parquet(dash_path, index=False, compression="snappy")
    print(f"  Dashboard file: {dash_path}  ({len(dash_df):,} rows)")

    return df


def build_model_dataset(
    forecast_dataset: str | Path = DATA_DIR / "forecast_dataset.parquet",
    price_csv: str | Path = DATA_DIR / "day_ahead_prices_dk1_raw.csv",
    output_path: str | Path = DATA_DIR / "model_dataset.parquet",
    price_area: str = "DK1",
) -> pd.DataFrame:
    """Build the model-ready dataset with target prices and issue-time lags."""
    forecast_dataset = Path(forecast_dataset)
    output_path = Path(output_path)

    print("[build_model_dataset] Loading forecast dataset...")
    df = pd.read_parquet(forecast_dataset)
    print(f"  Forecast rows: {len(df):,}")

    print("[build_model_dataset] Loading hourly prices and adding price features...")
    prices = load_hourly_prices(price_csv=price_csv, price_area=price_area)
    df = add_price_target_and_lags(df, prices)

    for suffix in ("10m", "100m"):
        rad = np.deg2rad(df[f"fcst_wind_direction_{suffix}"])
        df[f"fcst_wind_dir_{suffix}_sin"] = np.sin(rad)
        df[f"fcst_wind_dir_{suffix}_cos"] = np.cos(rad)

    df = df.sort_values(["issue_time", "horizon_h"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  {len(df):,} rows x {df.shape[1]} cols saved -> {output_path}")
    return df


WEATHER_PLOT_VARS = {
    "Wind 100m":   ("fcst_wind_speed_100m",    "actual_wind_speed_100m",    "m/s"),
    "Solar":       ("fcst_shortwave_radiation", "actual_shortwave_radiation", "W/m²"),
    "Temperature": ("fcst_temperature_2m",      "actual_temperature_2m",      "°C"),
}


def plot_weather_error_distributions(
    error_dist_csv: str | Path = DATA_DIR / "weather_error_distributions.csv",
) -> None:
    import matplotlib.pyplot as plt

    err       = pd.read_csv(error_dist_csv)
    variables = err["forecast_variable"].unique()
    n         = len(variables)
    ncols     = 4
    nrows     = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    axes      = axes.flatten()

    for ax, var in zip(axes, variables):
        sub = err[err["forecast_variable"] == var].sort_values("horizon_hours")
        h   = sub["horizon_hours"].values
        ax.fill_between(h, sub["p05_error"], sub["p95_error"], alpha=0.15, color="steelblue", label="p05-p95")
        ax.fill_between(h, sub["p25_error"], sub["p75_error"], alpha=0.35, color="steelblue", label="p25-p75")
        ax.plot(h, sub["mean_error"], color="steelblue", linewidth=2, marker="o", markersize=4, label="Mean error")
        ax.plot(h, sub["p50_error"], color="steelblue", linewidth=1.5, linestyle="--", marker="o", markersize=3, alpha=0.7, label="Median error")
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
    plt.show()


def plot_weather_forecast(
    dashboard_parquet: str | Path = DATA_DIR / "weather_dk1_dashboard.parquet",
    ctx_days: int = 7,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df         = pd.read_parquet(dashboard_parquet)
    issue_ts   = pd.to_datetime(df["issue_time"].max())
    hist_start = issue_ts - pd.Timedelta(days=ctx_days)
    fcst_end   = issue_ts + pd.Timedelta(days=5)

    hist = (
        df[(df["target_time"] >= hist_start) & (df["target_time"] < issue_ts)]
        .groupby("target_time", as_index=False).first()
        .sort_values("target_time")
    )
    fcst = (
        df[(df["issue_time"] == issue_ts) & (df["target_time"] > issue_ts) & (df["target_time"] <= fcst_end)]
        .sort_values("target_time")
    )

    colors = ["#38bdf8", "#fbbf24", "#f87171"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, (var_name, (fcst_col, actual_col, unit)), color in zip(axes, WEATHER_PLOT_VARS.items(), colors):
        ax.plot(hist["target_time"], hist[actual_col], color=color, linewidth=1.5, label="Actual")
        if not fcst.empty:
            ax.plot(fcst["target_time"], fcst[fcst_col], color=color, linewidth=1.5, linestyle="--", label="Forecast")
            if fcst[actual_col].notna().any():
                ax.plot(fcst["target_time"], fcst[actual_col], color="grey", linewidth=1, linestyle=":", label="Actual (fcst window)")
            ax.axvspan(fcst["target_time"].iloc[0], fcst["target_time"].iloc[-1], alpha=0.07, color=color)
        ax.axvline(issue_ts, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_ylabel(f"{var_name}\n({unit})", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    fig.suptitle(
        f"Weather · DK1 West · {ctx_days}d history + 5-day forecast\n"
        f"issued at {issue_ts.strftime('%Y-%m-%d %H:%M')}",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


# Dashboard-style weather plot


def plot_weather_forecast_dashboard_style(
    dashboard_parquet: str | Path = DATA_DIR / "weather_dk1_dashboard.parquet",
    issue_time: str | pd.Timestamp | None = None,
    ctx_days: int = 7,
    show_actuals: bool = True,
):
    """Plot DK1 weather actuals followed by the 5-day forecast window."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df = pd.read_parquet(dashboard_parquet).copy()
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])

    available_issues = pd.to_datetime(df["issue_time"].dropna().unique())
    if len(available_issues) == 0:
        raise ValueError("No issue_time values found in weather dashboard data.")

    if issue_time is None:
        selected_issue = pd.Timestamp(max(available_issues))
    else:
        requested = pd.Timestamp(issue_time)
        selected_issue = min(available_issues, key=lambda x: abs(x - requested))

    hist_start = selected_issue - pd.Timedelta(days=ctx_days)
    fcst_end = selected_issue + pd.Timedelta(days=5)

    hist = (
        df[(df["target_time"] >= hist_start) & (df["target_time"] < selected_issue)]
        .groupby("target_time", as_index=False)
        .first()
        .sort_values("target_time")
    )
    fcst = (
        df[
            (df["issue_time"] == selected_issue)
            & (df["target_time"] > selected_issue)
            & (df["target_time"] <= fcst_end)
        ]
        .sort_values("target_time")
    )

    variables = {
        "Wind 100m": ("fcst_wind_speed_100m", "actual_wind_speed_100m", "m/s", "#38bdf8"),
        "Solar": ("fcst_shortwave_radiation", "actual_shortwave_radiation", "W/m2", "#fbbf24"),
        "Temperature": ("fcst_temperature_2m", "actual_temperature_2m", "C", "#f87171"),
    }

    fig = make_subplots(
        rows=len(variables),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=list(variables.keys()),
    )

    for row, (name, (fcst_col, actual_col, unit, color)) in enumerate(variables.items(), start=1):
        fig.add_trace(
            go.Scatter(
                x=hist["target_time"],
                y=hist[actual_col],
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{name} actual",
                legendgroup=name,
            ),
            row=row,
            col=1,
        )

        if not fcst.empty:
            fig.add_trace(
                go.Scatter(
                    x=fcst["target_time"],
                    y=fcst[fcst_col],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    name=f"{name} forecast",
                    legendgroup=name,
                ),
                row=row,
                col=1,
            )

            if show_actuals and fcst[actual_col].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=fcst["target_time"],
                        y=fcst[actual_col],
                        mode="lines",
                        line=dict(color="#4ade80", width=1.5, dash="dash"),
                        name=f"{name} actual in forecast window",
                        legendgroup=name,
                    ),
                    row=row,
                    col=1,
                )

        fig.update_yaxes(title_text=unit, row=row, col=1)

    if not fcst.empty:
        for row in range(1, len(variables) + 1):
            fig.add_vrect(
                x0=fcst["target_time"].iloc[0],
                x1=fcst["target_time"].iloc[-1],
                fillcolor="rgba(167,139,250,0.10)",
                line_width=0,
                layer="below",
                annotation_text="5-day forecast" if row == 1 else "",
                annotation_position="top right",
                row=row,
                col=1,
            )

    fig.update_layout(
        template="plotly_white",
        height=760,
        title=(
            f"Weather - DK1 West - {ctx_days}d history + 5-day forecast "
            f"(issued {selected_issue:%Y-%m-%d %H:%M})"
        ),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=70, b=90),
    )
    fig.show()
    return fig


def load_hourly_prices(
    price_csv: str | Path = DATA_DIR / "day_ahead_prices_dk1_raw.csv",
    price_area: str = "DK1",
    target_col: str = TARGET_COL,
) -> pd.Series:
    """Load hourly day-ahead prices used for targets and issue-time lags."""
    raw = pd.read_csv(price_csv, parse_dates=["TimeDK"])
    if "PriceArea" in raw.columns:
        raw = raw[raw["PriceArea"].astype(str).eq(price_area)].copy()
    raw[target_col] = pd.to_numeric(raw[target_col], errors="coerce")
    return raw.set_index("TimeDK").sort_index()[target_col].resample("h").mean()


def add_price_target_and_lags(
    df: pd.DataFrame,
    prices: pd.Series,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Add target prices and lag features known at each issue_time."""
    out = df.copy()
    out["issue_time"] = pd.to_datetime(out["issue_time"])
    out["target_time"] = pd.to_datetime(out["target_time"])

    price_df = (
        prices.rename(target_col)
        .reset_index()
        .rename(columns={"TimeDK": "target_time"})
    )
    out = out.merge(price_df, on="target_time", how="inner")

    price_map = prices.to_dict()
    rows = {}
    for issue_time in pd.to_datetime(out["issue_time"].unique()):
        past_24 = [price_map.get(issue_time - pd.Timedelta(hours=h), np.nan) for h in range(1, 25)]
        rows[issue_time] = {
            "price_lag_24h": price_map.get(issue_time - pd.Timedelta(hours=24), np.nan),
            "price_lag_48h": price_map.get(issue_time - pd.Timedelta(hours=48), np.nan),
            "price_lag_168h": price_map.get(issue_time - pd.Timedelta(hours=168), np.nan),
            "price_rolling_24h_mean": np.nanmean(past_24),
        }

    lag_df = pd.DataFrame.from_dict(rows, orient="index")
    lag_df.index.name = "issue_time"
    out = out.merge(lag_df.reset_index(), on="issue_time", how="left")
    return out.dropna(subset=[target_col] + PRICE_LAG_FEATURES).reset_index(drop=True)


VARIABLE_MAP = {
    "temperature_2m":      "temperature_2m",
    "pressure_msl":        "pressure_msl",
    "cloud_cover":         "cloud_cover",
    "shortwave_radiation": "shortwave_radiation",
    "wind_speed_10m":      "wind_speed_10m",
    "wind_direction_10m":  "wind_direction_10m",
    "wind_speed_120m":     "wind_speed_100m",
    "wind_direction_120m": "wind_direction_100m",
}

ERR_HORIZONS = {24: "previous_day1", 48: "previous_day2", 72: "previous_day3",
                96: "previous_day4", 120: "previous_day5"}


def _circular_diff(forecast: pd.Series, actual: pd.Series) -> pd.Series:
    return (forecast - actual + 180) % 360 - 180


def build_error_distributions(
    actuals_csv:  str | Path = DATA_DIR / "weather_actuals_raw.csv",
    forecast_csv: str | Path = DATA_DIR / "weather_forecasts_raw.csv",
    summary_csv:  str | Path = DATA_DIR / "weather_error_distributions.csv",
    errors_csv:   str | Path = DATA_DIR / "weather_errors_raw.csv",
) -> None:
    """Estimate weather forecast error distributions from real NWP previous-run data."""
    print("[build_error_distributions] Loading actuals and forecasts...")
    actual_df   = pd.read_csv(actuals_csv,  parse_dates=["TimeDK"])
    forecast_df = pd.read_csv(forecast_csv, parse_dates=["TimeDK"])

    merged = forecast_df.merge(actual_df, on=["TimeDK", "region"], how="inner",
                               suffixes=("_forecastfile", "_actualfile"))
    print(f"  Merged rows: {len(merged):,}")

    rows = []
    for fcst_base, act_base in VARIABLE_MAP.items():
        # When names collide the merge adds suffixes; otherwise columns are unique
        if fcst_base == act_base:
            act_col = f"{act_base}_actualfile"
        else:
            act_col = act_base
        if act_col not in merged.columns:
            continue

        for horizon_h, suffix in ERR_HORIZONS.items():
            fcst_col = f"{fcst_base}_{suffix}"
            if fcst_col not in merged.columns:
                continue

            sub = (
                merged[["TimeDK", "region", fcst_col, act_col]]
                .rename(columns={fcst_col: "forecast_value", act_col: "actual_value"})
                .dropna(subset=["forecast_value", "actual_value"])
                .copy()
            )
            if sub.empty:
                continue

            is_dir = "direction" in fcst_base
            sub["error"] = (_circular_diff(sub["forecast_value"], sub["actual_value"])
                            if is_dir else sub["forecast_value"] - sub["actual_value"])
            sub["forecast_variable"] = fcst_base
            sub["actual_variable"]   = act_base
            sub["horizon_hours"]     = horizon_h
            rows.append(sub[[
                "TimeDK", "region",
                "forecast_variable", "actual_variable", "horizon_hours",
                "forecast_value", "actual_value", "error",
            ]])

    error_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[
        "TimeDK", "region", "forecast_variable", "actual_variable",
        "horizon_hours", "forecast_value", "actual_value", "error",
    ])
    print(f"  Error rows: {len(error_df):,}")

    if not error_df.empty:
        grp = error_df.groupby(["forecast_variable", "actual_variable", "horizon_hours"])["error"]
        summary_df = grp.agg(
            n            = "count",
            mean_error   = "mean",
            std_error    = "std",
            mae          = lambda x: np.mean(np.abs(x)),
            rmse         = lambda x: np.sqrt(np.mean(np.square(x))),
            p05_error    = lambda x: np.quantile(x, 0.05),
            p25_error    = lambda x: np.quantile(x, 0.25),
            p50_error    = lambda x: np.quantile(x, 0.50),
            p75_error    = lambda x: np.quantile(x, 0.75),
            p95_error    = lambda x: np.quantile(x, 0.95),
        ).reset_index().sort_values(["forecast_variable", "horizon_hours"]).reset_index(drop=True)
    else:
        summary_df = pd.DataFrame(columns=[
            "forecast_variable", "actual_variable", "horizon_hours",
            "n", "mean_error", "std_error", "mae", "rmse",
            "p05_error", "p25_error", "p50_error", "p75_error", "p95_error",
        ])

    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    print(f"  {len(summary_df)} summary rows saved -> {summary_csv}")

    if not summary_df.empty:
        print("\nSummary preview:")
        print(summary_df.head(20).to_string(index=False))

    Path(errors_csv).parent.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(errors_csv, index=False)
    print(f"  {len(error_df):,} raw error rows saved -> {errors_csv}")


def main() -> None:
    """Run the full processing step with a short notebook-friendly summary."""
    with contextlib.redirect_stdout(io.StringIO()):
        build_error_distributions()
        build_forecast_dataset()
        model_df = build_model_dataset()

    forecast_raw = pd.read_csv(DATA_DIR / "weather_forecasts_raw.csv", parse_dates=["TimeDK"])
    period_start = pd.to_datetime(model_df["target_time"]).min().strftime("%Y-%m-%d")
    period_end = pd.to_datetime(model_df["target_time"]).max().strftime("%Y-%m-%d")
    forecast_start = forecast_raw["TimeDK"].min().strftime("%Y-%m-%d")
    forecast_end = forecast_raw["TimeDK"].max().strftime("%Y-%m-%d")

    print(f"Data processing from {period_start} to {period_end} is done.")
    print(
        f"Actual weather forecasts from {forecast_start} to {forecast_end} "
        "were used to estimate forecast errors and generate forecast-like weather inputs."
    )


if __name__ == "__main__":
    main()
