from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths & constants ─────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
PRICE_FILE  = DATA_DIR / "day_ahead_prices_dk1_raw.csv"
PRED_FILE    = Path("outputs/model/predictions.parquet")
METRICS_FILE = Path("outputs/model/metrics.csv")
MODEL_FILE        = Path("outputs/model/final_day_models.joblib")
SHAP_VALUES_FILE   = Path("outputs/model/shap_day1_values.parquet")
SHAP_FEATURES_FILE = Path("outputs/model/shap_day1_features.parquet")
WEATHER_FILE       = Path("data/weather_dk1_dashboard.parquet")
PRICE_AREA  = "DK1"
EUR_DKK_FB  = 7.46   # fallback exchange rate

WEATHER_VARS = {
    "Wind 100m":   ("fcst_wind_speed_100m",    "actual_wind_speed_100m",    "m/s",  "#38bdf8"),
    "Solar":       ("fcst_shortwave_radiation", "actual_shortwave_radiation", "W/m²", "#fbbf24"),
    "Temperature": ("fcst_temperature_2m",      "actual_temperature_2m",      "°C",   "#f87171"),
}

C = {
    "bg":          "#0f172a",
    "card":        "#1e293b",
    "border":      "#334155",
    "text":        "#f1f5f9",
    "muted":       "#94a3b8",
    "accent":      "#38bdf8",
    "price":       "#fb923c",
    "price_fill":  "rgba(251,146,60,0.12)",
    "fcst":        "#a78bfa",
    "fcst_fill":   "rgba(167,139,250,0.12)",
    "good":        "#4ade80",
    "warn":        "#f87171",
    "grid":        "rgba(148,163,184,0.07)",
}


# ── Page config & CSS ─────────────────────────────────────────────────────────
def setup_page() -> None:
    st.set_page_config(
        page_title="DK1 Energy Planner",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(f"""
    <style>
    .stApp {{background:{C['bg']}; color:{C['text']};}}
    .block-container {{padding:1.5rem 2rem 3rem 2rem; max-width:1440px;}}

    /* KPI cards */
    .kpi {{background:{C['card']}; border:1px solid {C['border']}; border-radius:16px;
           padding:1.2rem 1.5rem; min-height:108px;}}
    .kpi-label {{color:{C['muted']}; font-size:.72rem; font-weight:700;
                 text-transform:uppercase; letter-spacing:.07em;}}
    .kpi-val   {{font-size:1.9rem; font-weight:900; line-height:1.1; margin:.2rem 0 0 0;}}
    .kpi-unit  {{color:{C['muted']}; font-size:.76rem; margin-top:.15rem;}}

    /* Section headers */
    .section {{color:{C['muted']}; font-size:.72rem; font-weight:700;
               text-transform:uppercase; letter-spacing:.08em; margin:1.5rem 0 .65rem 0;}}

    /* Forecast day cards */
    .fcard {{background:{C['card']}; border:1px solid {C['border']}; border-radius:14px;
             padding:1rem 1.1rem; text-align:center;}}

    /* Delta badges */
    .tag-up  {{background:rgba(248,113,113,.15); color:{C['warn']};   border-radius:20px;
               padding:.18rem .65rem; font-size:.76rem; font-weight:700; white-space:nowrap;}}
    .tag-dn  {{background:rgba(74,222,128,.15);  color:{C['good']};   border-radius:20px;
               padding:.18rem .65rem; font-size:.76rem; font-weight:700; white-space:nowrap;}}
    .tag-eq  {{background:rgba(148,163,184,.15); color:{C['muted']};  border-radius:20px;
               padding:.18rem .65rem; font-size:.76rem; font-weight:700; white-space:nowrap;}}

    /* Hide Streamlit chrome */
    #MainMenu, footer, .stDeployButton {{display:none !important;}}
    div[data-testid="stSidebar"] {{background:{C['card']};}}
    </style>
    """, unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(PRICE_FILE)

    ts_col = next((c for c in ("TimeDK", "HourDK", "TimeUTC", "HourUTC") if c in df.columns), None)
    if ts_col is None:
        return pd.DataFrame()
    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")

    if "PriceArea" in df.columns:
        df = df[df["PriceArea"].astype(str).str.strip() == PRICE_AREA].copy()

    if "DayAheadPriceDKK" not in df.columns:
        return pd.DataFrame()

    df["PriceDKK"] = pd.to_numeric(df["DayAheadPriceDKK"], errors="coerce")
    if "DayAheadPriceEUR" in df.columns:
        df["PriceEUR"] = pd.to_numeric(df["DayAheadPriceEUR"], errors="coerce")

    df = df.dropna(subset=["_ts", "PriceDKK"]).sort_values("_ts").reset_index(drop=True)
    df["Date"] = df["_ts"].dt.normalize()
    df["Hour"] = df["_ts"].dt.hour
    return df


@st.cache_data(show_spinner=False)
def load_raw_predictions() -> pd.DataFrame:
    if not PRED_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(PRED_FILE)


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    if not METRICS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_FILE)


@st.cache_data(show_spinner=False)
def compute_forecast_bands(_raw: pd.DataFrame, eur_dkk: float) -> dict[int, float]:
    """Return {horizon_h: sigma_dkk} from historical walk-forward residuals."""
    if _raw.empty:
        return {}
    valid = _raw.dropna(subset=["predicted", "DayAheadPriceEUR"])
    valid = valid[valid["DayAheadPriceEUR"] > 0]
    if valid.empty:
        return {}
    valid = valid.copy()
    valid["residual_eur"] = valid["predicted"] - valid["DayAheadPriceEUR"]
    sigma = valid.groupby("horizon_h")["residual_eur"].std() * eur_dkk
    return sigma.to_dict()


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    """Top-N feature importances from the Day 1 XGBoost model (final_day_models.joblib)."""
    if not MODEL_FILE.exists():
        return pd.DataFrame()
    try:
        import joblib
        obj   = joblib.load(MODEL_FILE)
        # final_day_models is {1: model, 2: model, ..., 5: model}
        model = obj[1] if isinstance(obj, dict) and 1 in obj else obj
        names = [str(f) for f in model.feature_names_in_]
        df    = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_shap_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-computed SHAP values and features for the Day 1 model."""
    if not SHAP_VALUES_FILE.exists() or not SHAP_FEATURES_FILE.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        return pd.read_parquet(SHAP_VALUES_FILE), pd.read_parquet(SHAP_FEATURES_FILE)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_weather_data() -> pd.DataFrame:
    """Load pre-aggregated DK1_west weather forecasts and actuals."""
    if not WEATHER_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(WEATHER_FILE)
        df["target_time"] = pd.to_datetime(df["target_time"])
        df["issue_time"]  = pd.to_datetime(df["issue_time"])
        return df.sort_values("target_time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── Data processing ───────────────────────────────────────────────────────────
def compute_eur_dkk(prices: pd.DataFrame) -> float:
    if prices.empty or "PriceEUR" not in prices.columns:
        return EUR_DKK_FB
    valid = prices.dropna(subset=["PriceDKK", "PriceEUR"])
    valid = valid[valid["PriceEUR"] > 1]
    if len(valid) < 24:
        return EUR_DKK_FB
    return float((valid["PriceDKK"] / valid["PriceEUR"]).tail(24 * 90).median())


def _resolve_issue_time(raw: pd.DataFrame, selected: pd.Timestamp | None) -> pd.Timestamp:
    """Return the closest available issue_time to `selected`, or the latest if None."""
    if selected is None:
        return pd.Timestamp(raw["issue_time"].max())
    available = pd.to_datetime(raw["issue_time"].unique())
    return min(available, key=lambda x: abs(x - selected))


def process_predictions(
    raw: pd.DataFrame, eur_dkk: float, selected: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Aggregate selected issue_time's hourly predictions to 5 daily forecasts."""
    if raw.empty:
        return pd.DataFrame()

    issue_ts   = _resolve_issue_time(raw, selected)
    issue_date = issue_ts.normalize()
    sub        = raw[raw["issue_time"] == issue_ts].copy()
    sub["target_date"] = sub["target_time"].dt.normalize()

    future = sub[sub["target_date"] > issue_date]
    daily  = (
        future.groupby("target_date")
        .agg(pred_eur=("predicted", "mean"), actual_eur=("DayAheadPriceEUR", "mean"))
        .reset_index()
        .rename(columns={"target_date": "Date"})
        .head(5)
    )
    daily["pred_dkk"]   = daily["pred_eur"]   * eur_dkk
    daily["actual_dkk"] = daily["actual_eur"] * eur_dkk
    daily["issue_date"] = issue_date
    return daily


def process_predictions_hourly(
    raw: pd.DataFrame,
    eur_dkk: float,
    selected: pd.Timestamp | None = None,
    bands: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Return all 120 hourly predictions for the selected issue_time, with ±1σ bands."""
    if raw.empty:
        return pd.DataFrame()

    issue_ts = _resolve_issue_time(raw, selected)
    sub      = raw[raw["issue_time"] == issue_ts].copy()

    future = sub[sub["target_time"] > issue_ts].sort_values("target_time").head(120)
    future["pred_dkk"]   = future["predicted"]        * eur_dkk
    future["actual_dkk"] = future["DayAheadPriceEUR"] * eur_dkk
    future["issue_date"] = issue_ts.normalize()
    future["sigma_dkk"]  = future["horizon_h"].map(bands or {}).fillna(0.0)
    return future[["target_time", "horizon_h", "pred_dkk", "actual_dkk", "issue_date", "sigma_dkk"]].reset_index(drop=True)


def daily_history(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    return (
        prices.groupby("Date", as_index=False)["PriceDKK"]
        .mean()
        .rename(columns={"PriceDKK": "AvgDKK"})
        .sort_values("Date")
    )


# ── Groq AI ───────────────────────────────────────────────────────────────────
def groq_call(prompt: str) -> str:
    try:
        from groq import Groq
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            return ""
        if not api_key:
            return ""
        client = Groq(api_key=api_key)
        resp   = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": prompt}],
            max_tokens  = 320,
            temperature = 0.4,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _build_hourly_day_summary(hourly_fcst: pd.DataFrame) -> str:
    """Build a per-day intraday breakdown string for LLM prompts."""
    if hourly_fcst.empty:
        return ""
    lines = []
    hourly_fcst = hourly_fcst.copy()
    hourly_fcst["_date"] = hourly_fcst["target_time"].dt.normalize()
    for date, grp in hourly_fcst.groupby("_date"):
        grp = grp.sort_values("target_time")
        peak_row   = grp.loc[grp["pred_dkk"].idxmax()]
        cheap_row  = grp.loc[grp["pred_dkk"].idxmin()]
        avg        = grp["pred_dkk"].mean()
        lines.append(
            f"  {pd.Timestamp(date).strftime('%a %d %b')}:  "
            f"avg {avg:.0f}  |  peak {peak_row['pred_dkk']:.0f} @ {peak_row['target_time'].strftime('%H:%M')}  "
            f"|  cheapest {cheap_row['pred_dkk']:.0f} @ {cheap_row['target_time'].strftime('%H:%M')}  "
            f"  (all DKK/MWh)"
        )
    return "\n".join(lines)


def _build_weather_summary(weather: pd.DataFrame, issue_ts: pd.Timestamp) -> str:
    """Per-day avg wind speed, solar radiation, and temperature for the 5-day forecast window."""
    if weather.empty:
        return "  (weather data unavailable)"
    available = pd.to_datetime(weather["issue_time"].unique())
    w_issue = min(available, key=lambda x: abs(x - issue_ts))
    fcst = (
        weather[
            (weather["issue_time"] == w_issue)
            & (weather["target_time"] > w_issue)
            & (weather["target_time"] <= w_issue + pd.Timedelta(days=5))
        ]
        .copy()
    )
    if fcst.empty:
        return "  (weather data unavailable)"
    fcst["_date"] = fcst["target_time"].dt.normalize()
    lines = []
    for date, grp in fcst.groupby("_date"):
        lines.append(
            f"  {pd.Timestamp(date).strftime('%a %d %b')}:  "
            f"wind {grp['fcst_wind_speed_100m'].mean():.1f} m/s  |  "
            f"solar {grp['fcst_shortwave_radiation'].mean():.0f} W/m²  |  "
            f"temp {grp['fcst_temperature_2m'].mean():.1f} °C"
        )
    return "\n".join(lines)


def prompt_reasoning(
    fcst: pd.DataFrame,
    today_avg: float,
    issue_date: pd.Timestamp,
    hourly_fcst: pd.DataFrame,
    weather: pd.DataFrame,
) -> str:
    daily_rows = "\n".join(
        f"  {r.Date.strftime('%a %d %b')}: {r.pred_dkk:.0f} DKK/MWh  ({r.pred_dkk - today_avg:+.0f} vs today)"
        for _, r in fcst.iterrows()
    )
    intraday    = _build_hourly_day_summary(hourly_fcst)
    weather_str = _build_weather_summary(weather, issue_date)

    return f"""You are a Nordic electricity market analyst. Be direct and concise.

DK1 forecast issued {issue_date.strftime('%d %b %Y')} — today avg: {today_avg:.0f} DKK/MWh

PRICES (daily avg, peak, cheapest):
{intraday}

WEATHER FORECAST (avg per day):
{weather_str}

Write EXACTLY 4 bullet points. Format: "• [day]: [weather fact] → [price impact]."
Rules:
- ONE sentence per bullet. Hard limit: 20 words per bullet.
- Must include a number from the weather table and a number from the price table.
- No hedging words. No preamble. No closing. Only the 4 bullets."""


def prompt_household(
    fcst: pd.DataFrame,
    today_avg: float,
    hourly_fcst: pd.DataFrame,
) -> str:
    daily_rows = "\n".join(
        f"  {r.Date.strftime('%A %d %b')}: avg {r.pred_dkk:.0f} DKK/MWh  "
        f"({'↑ EXPENSIVE' if r.pred_dkk - today_avg > 50 else '↓ CHEAP' if r.pred_dkk - today_avg < -50 else '→ similar'})"
        for _, r in fcst.iterrows()
    )
    intraday = _build_hourly_day_summary(hourly_fcst)

    return f"""You are an energy advisor helping a Danish household cut their electricity bill.

DK1 spot price forecast:
Today's average: {today_avg:.0f} DKK/MWh

DAILY AVERAGES:
{daily_rows}

INTRADAY PROFILE per day (cheapest hour is the best time to run appliances):
{intraday}

RULES:
- Write 5 bullet points, each starting with •.
- Every bullet must name a specific day AND a specific hour window from the data above.
- Do NOT use "may", "could", "might", "perhaps", or "consider".
- Give direct instructions, not suggestions: "Charge your EV on [day] between [H1] and [H2] — price is [X] DKK/MWh."
- Cover: EV charging, dishwasher/washing machine, heat pump/electric heating, hot water boiler, and one other flexible load.
- No preamble. No closing sentence."""


# ── Chart helpers ─────────────────────────────────────────────────────────────
_BASE = dict(
    template      = "plotly_dark",
    paper_bgcolor = C["card"],
    plot_bgcolor  = C["card"],
    font          = dict(color=C["text"], size=12),
    margin        = dict(l=10, r=10, t=36, b=10),
    xaxis         = dict(showgrid=True, gridcolor=C["grid"], zeroline=False, linecolor=C["border"]),
    yaxis         = dict(showgrid=True, gridcolor=C["grid"], zeroline=False, linecolor=C["border"]),
    hoverlabel    = dict(bgcolor=C["bg"], bordercolor=C["border"]),
)


def fig_today(hourly: pd.DataFrame, date: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly["Hour"], y=hourly["PriceDKK"],
        mode="lines+markers",
        line=dict(color=C["price"], width=3, shape="spline", smoothing=0.7),
        marker=dict(size=6, color=C["price"]),
        fill="tozeroy", fillcolor=C["price_fill"],
        hovertemplate="<b>%{x}:00</b><br>%{y:.0f} DKK/MWh<extra></extra>",
    ))
    if not hourly.empty:
        peak  = hourly.loc[hourly["PriceDKK"].idxmax()]
        cheap = hourly.loc[hourly["PriceDKK"].idxmin()]
        for h, lbl, col in [
            (int(peak["Hour"]),  "Peak",     C["warn"]),
            (int(cheap["Hour"]), "Cheapest", C["good"]),
        ]:
            fig.add_vline(x=h, line_dash="dot", line_color=col, opacity=0.55,
                          annotation_text=lbl, annotation_font_color=col, annotation_font_size=11)
    fig.update_layout(**{
        **_BASE,
        "height": 320, "showlegend": False,
        "title": dict(text=f"Hourly spot price · {date.strftime('%d %b %Y')}", font_size=13),
        "xaxis": dict(**_BASE["xaxis"], dtick=3, ticksuffix=":00", range=[-0.5, 23.5]),
        "yaxis_title": "DKK/MWh",
    })
    return fig


def fig_history(hist: pd.DataFrame, days_back: int) -> go.Figure:
    df  = hist.tail(days_back).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["AvgDKK"],
        mode="lines", line=dict(color=C["price"], width=2.5),
        fill="tozeroy", fillcolor=C["price_fill"],
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.0f} DKK/MWh</b><extra></extra>",
        name="Daily avg",
    ))
    if len(df) >= 7:
        roll = df["AvgDKK"].rolling(30, min_periods=7).mean()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=roll,
            mode="lines", line=dict(color=C["accent"], width=2, dash="dash"),
            hovertemplate="%{x|%d %b %Y}<br>30d avg: <b>%{y:.0f}</b><extra></extra>",
            name="30-day avg",
        ))
    fig.update_layout(
        **_BASE, height=320,
        title=dict(text=f"Daily average price · last {days_back} days", font_size=13),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        yaxis_title="DKK/MWh",
    )
    return fig


def fig_context(
    prices: pd.DataFrame,
    hourly_fcst: pd.DataFrame,
    ctx_days: int,
    cutoff: pd.Timestamp | None = None,
    show_actuals: bool = True,
) -> go.Figure:
    """Last ctx_days of hourly actuals + 120-hour XGBoost forecast at hourly resolution."""
    # hist_end = first forecast target hour (exclusive), so there is no overlap
    if not hourly_fcst.empty:
        hist_end = hourly_fcst["target_time"].iloc[0]
    elif cutoff is not None:
        hist_end = pd.Timestamp(cutoff) + pd.Timedelta(hours=1)
    else:
        hist_end = prices["_ts"].max() + pd.Timedelta(hours=1)

    hist_start = hist_end - pd.Timedelta(days=ctx_days)

    # Floor raw timestamps to the hour so granularity matches the forecast
    df = prices[(prices["_ts"] >= hist_start) & (prices["_ts"] < hist_end)].copy()
    df["_ts_h"] = df["_ts"].dt.floor("h")
    hourly_hist = df.groupby("_ts_h", as_index=False)["PriceDKK"].mean().sort_values("_ts_h")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hourly_hist["_ts_h"], y=hourly_hist["PriceDKK"],
        mode="lines", line=dict(color=C["price"], width=2),
        hovertemplate="%{x|%d %b %H:%M}<br>Actual: <b>%{y:.0f} DKK/MWh</b><extra></extra>",
        name="Historical actual",
    ))

    # No explicit connector needed — hist ends at the hour before the first forecast point
    # so the two lines meet naturally with no gap and no overlap.

    if not hourly_fcst.empty:
        has_bands = "sigma_dkk" in hourly_fcst.columns and hourly_fcst["sigma_dkk"].gt(0).any()

        if has_bands:
            upper = hourly_fcst["pred_dkk"] + hourly_fcst["sigma_dkk"]
            lower = hourly_fcst["pred_dkk"] - hourly_fcst["sigma_dkk"]
            # Upper boundary (invisible line, sets ceiling for fill)
            fig.add_trace(go.Scatter(
                x=hourly_fcst["target_time"], y=upper,
                mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ))
            # Lower boundary filled back to upper → ±1σ band
            fig.add_trace(go.Scatter(
                x=hourly_fcst["target_time"], y=lower,
                mode="lines", line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(167,139,250,0.18)",
                hovertemplate="%{x|%d %b %H:%M}<br>±1σ: <b>%{y:.0f}</b><extra></extra>",
                name="±1σ confidence",
            ))

        # XGBoost hourly predictions
        fig.add_trace(go.Scatter(
            x=hourly_fcst["target_time"], y=hourly_fcst["pred_dkk"],
            mode="lines",
            line=dict(color=C["fcst"], width=2.5),
            fill="tozeroy", fillcolor=C["fcst_fill"],
            hovertemplate="%{x|%d %b %H:%M}<br>XGBoost: <b>%{y:.0f} DKK/MWh</b><extra></extra>",
            name="XGBoost forecast",
        ))

        # Actual prices for the forecast window (optional overlay)
        if show_actuals and hourly_fcst["actual_dkk"].notna().any():
            fig.add_trace(go.Scatter(
                x=hourly_fcst["target_time"], y=hourly_fcst["actual_dkk"],
                mode="lines",
                line=dict(color=C["good"], width=2, dash="dot"),
                hovertemplate="%{x|%d %b %H:%M}<br>Actual: <b>%{y:.0f} DKK/MWh</b><extra></extra>",
                name="Actual (for comparison)",
            ))

        # Shaded forecast zone
        fig.add_vrect(
            x0=hourly_fcst["target_time"].iloc[0],
            x1=hourly_fcst["target_time"].iloc[-1],
            fillcolor=C["fcst_fill"], layer="below", line_width=0,
            annotation_text="5-day forecast (hourly)", annotation_position="top left",
            annotation_font_color=C["fcst"], annotation_font_size=11,
        )

    fig.update_layout(
        **_BASE, height=400,
        title=dict(text="Historical context + XGBoost 5-day forecast · hourly resolution", font_size=13),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        yaxis_title="DKK/MWh",
    )
    return fig


def fig_feature_importance(imp_df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    df = imp_df.head(top_n).copy()
    # Shorten labels: drop region prefix, replace underscores, title-case
    def clean(name: str) -> str:
        name = name.replace("DK1_west_", "").replace("DK2_east_", "")
        name = name.replace("DE_north_", "DE ").replace("NO_south_", "NO ")
        name = name.replace("SE_south_", "SE ").replace("_", " ")
        return name.title()
    df["label"] = df["feature"].apply(clean)
    df = df.sort_values("importance", ascending=True)   # ascending so highest is at top in bar chart

    # Colour bars by category
    def bar_color(name: str) -> str:
        if "wind" in name:    return C["accent"]
        if "price" in name:   return C["price"]
        if "temp" in name or "radiation" in name or "cloud" in name: return C["fcst"]
        return C["muted"]

    colors = [bar_color(f) for f in df["feature"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["importance"],
        y=df["label"],
        orientation="h",
        marker=dict(color=colors, opacity=0.9),
        hovertemplate="%{y}<br>Importance: <b>%{x:.4f}</b><extra></extra>",
    ))
    fig.update_layout(**{
        **_BASE,
        "height": 280,
        "showlegend": False,
        "title": dict(text="Top 5 feature importances · XGBoost Day 1 model (h=1–24)", font_size=13),
        "xaxis": dict(**_BASE["xaxis"], title="Gain importance"),
        "yaxis": dict(**{**_BASE["yaxis"], "showgrid": False}),
        "margin": dict(l=10, r=10, t=36, b=10),
    })
    return fig


def fig_shap(shap_vals: pd.DataFrame, shap_feat: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Beeswarm-style SHAP scatter: x=SHAP value, y=feature, colour=feature magnitude."""
    def clean(name: str) -> str:
        return name.replace("fcst_", "").replace("_", " ").title()

    mean_abs  = shap_vals.abs().mean().sort_values(ascending=False)
    top_feats = mean_abs.head(top_n).index.tolist()

    rng = np.random.default_rng(42)
    fig = go.Figure()

    # Iterate bottom → top so highest-importance feature renders at top of y-axis
    for i, feat in enumerate(reversed(top_feats)):
        sv    = shap_vals[feat].values
        fv    = shap_feat[feat].values
        fv_n  = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
        jit   = rng.uniform(-0.35, 0.35, size=len(sv))

        show_cb = (i == top_n - 1)  # only one colorbar on the topmost trace
        fig.add_trace(go.Scatter(
            x=sv,
            y=np.full(len(sv), i) + jit,
            mode="markers",
            marker=dict(
                color=fv_n,
                colorscale="RdBu_r",
                cmin=0, cmax=1,
                size=4,
                opacity=0.65,
                showscale=show_cb,
                colorbar=dict(
                    title=dict(text="Feature value", font=dict(size=10)),
                    tickvals=[0, 0.5, 1],
                    ticktext=["Low", "Mid", "High"],
                    tickfont=dict(size=9),
                    thickness=10,
                    len=0.55,
                    x=1.02,
                ) if show_cb else None,
            ),
            hovertemplate=f"{clean(feat)}<br>SHAP: <b>%{{x:.3f}}</b><extra></extra>",
            showlegend=False,
        ))

    feat_labels = [clean(f) for f in reversed(top_feats)]
    fig.update_layout(**{
        **_BASE,
        "height": 380,
        "showlegend": False,
        "title": dict(text="SHAP values · XGBoost Day 1 model  (red = high feature value, blue = low)", font_size=13),
        "xaxis": dict(**{**_BASE["xaxis"], "zeroline": True, "zerolinecolor": C["border"]}, title="SHAP value (EUR/MWh impact)"),
        "yaxis": dict(**{
            **_BASE["yaxis"],
            "showgrid": False,
            "tickmode": "array",
            "tickvals": list(range(top_n)),
            "ticktext": feat_labels,
        }),
        "margin": dict(l=10, r=20, t=42, b=10),
    })
    return fig


def fig_weather(
    weather: pd.DataFrame,
    selected_vars: list[str],
    ctx_days: int,
    issue_ts: pd.Timestamp,
    show_actuals: bool = True,
) -> go.Figure:
    """Actuals up to issue_ts, then 5-day fcst_* values — mirrors the XGBoost context chart."""
    from plotly.subplots import make_subplots

    # Resolve closest available issue_time in the weather data
    available_issues = pd.to_datetime(weather["issue_time"].unique())
    w_issue = min(available_issues, key=lambda x: abs(x - issue_ts))

    hist_start  = w_issue - pd.Timedelta(days=ctx_days)
    fcst_end    = w_issue + pd.Timedelta(days=5)

    # Historical: deduplicate target_time, take actual_* columns
    hist = (
        weather[(weather["target_time"] >= hist_start) & (weather["target_time"] < w_issue)]
        .groupby("target_time", as_index=False)
        .first()
        .sort_values("target_time")
    )

    # Forecast: rows from the resolved issue_time, target > issue
    fcst = (
        weather[
            (weather["issue_time"] == w_issue)
            & (weather["target_time"] > w_issue)
            & (weather["target_time"] <= fcst_end)
        ]
        .sort_values("target_time")
    )

    n   = len(selected_vars)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=selected_vars,
    )

    for i, var_name in enumerate(selected_vars, start=1):
        fcst_col, actual_col, unit, color = WEATHER_VARS[var_name]

        # Historical actuals
        fig.add_trace(go.Scatter(
            x=hist["target_time"], y=hist[actual_col],
            mode="lines", line=dict(color=color, width=2),
            name=f"{var_name} · actual",
            legendgroup=var_name,
            hovertemplate=f"%{{x|%d %b %H:%M}}<br>Actual: <b>%{{y:.1f}} {unit}</b><extra></extra>",
        ), row=i, col=1)

        # 5-day weather forecast
        if not fcst.empty:
            fig.add_trace(go.Scatter(
                x=fcst["target_time"], y=fcst[fcst_col],
                mode="lines", line=dict(color=color, width=2, dash="dot"),
                name=f"{var_name} · forecast",
                legendgroup=var_name,
                hovertemplate=f"%{{x|%d %b %H:%M}}<br>Forecast: <b>%{{y:.1f}} {unit}</b><extra></extra>",
            ), row=i, col=1)

            if show_actuals and fcst[actual_col].notna().any():
                fig.add_trace(go.Scatter(
                    x=fcst["target_time"], y=fcst[actual_col],
                    mode="lines", line=dict(color=C["good"], width=1.5, dash="dash"),
                    name=f"{var_name} · actual (fcst window)",
                    legendgroup=var_name,
                    hovertemplate=f"%{{x|%d %b %H:%M}}<br>Actual: <b>%{{y:.1f}} {unit}</b><extra></extra>",
                ), row=i, col=1)

        fig.update_yaxes(
            title_text=unit, row=i, col=1,
            showgrid=True, gridcolor=C["grid"],
            zeroline=False, linecolor=C["border"],
            color=C["text"], tickfont=dict(size=10),
        )
        fig.update_xaxes(
            showgrid=True, gridcolor=C["grid"],
            zeroline=False, linecolor=C["border"],
            color=C["text"], row=i, col=1,
        )

    # Shade the 5-day forecast window on every subplot row
    if not fcst.empty:
        x0 = fcst["target_time"].iloc[0]
        x1 = fcst["target_time"].iloc[-1]
        for row_i in range(1, n + 1):
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor="rgba(167,139,250,0.08)", layer="below", line_width=0,
                # Only annotate on the first row, positioned inside so it doesn't clip
                annotation_text="5-day forecast" if row_i == 1 else "",
                annotation_position="top right",
                annotation_font_color=C["fcst"], annotation_font_size=10,
                row=row_i, col=1,
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["card"],
        plot_bgcolor=C["card"],
        font=dict(color=C["text"], size=12),
        margin=dict(l=10, r=10, t=48, b=90),
        height=200 * n + 90,
        hoverlabel=dict(bgcolor=C["bg"], bordercolor=C["border"]),
        legend=dict(
            orientation="h", y=-0.08, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            groupclick="toggleitem",
            tracegroupgap=4,
        ),
        title=dict(
            text=f"Weather · DK1 West · {ctx_days}d history + 5-day forecast  "
                 f"<span style='font-size:12px;color:{C['muted']};'>"
                 f"solid = actual · dotted = forecast</span>",
            font_size=13,
        ),
    )
    for ann in fig.layout.annotations:
        ann.font.color = C["muted"]
        ann.font.size  = 11
    return fig


# ── Forecast cards ────────────────────────────────────────────────────────────
def render_forecast_cards(fcst: pd.DataFrame, today_avg: float) -> None:
    cols = st.columns(len(fcst))
    for col, (_, row) in zip(cols, fcst.iterrows()):
        delta = row["pred_dkk"] - today_avg
        if delta > 25:
            tag = f"<span class='tag-up'>↑ +{delta:.0f}</span>"
        elif delta < -25:
            tag = f"<span class='tag-dn'>↓ {delta:.0f}</span>"
        else:
            tag = f"<span class='tag-eq'>→ {delta:+.0f}</span>"

        col.markdown(f"""
        <div class="fcard">
            <div class="kpi-label" style="font-size:.7rem;">{row['Date'].strftime('%a')}</div>
            <div class="kpi-label">{row['Date'].strftime('%d %b')}</div>
            <div style="font-size:1.75rem; font-weight:900; color:{C['fcst']};
                        margin:.25rem 0 .1rem 0;">{row['pred_dkk']:.0f}</div>
            <div class="kpi-unit">DKK/MWh · daily avg</div>
            <div style="margin-top:.55rem;">{tag} vs today avg</div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(
    prices: pd.DataFrame,
    fcst: pd.DataFrame,
    metrics: pd.DataFrame,
    raw_preds: pd.DataFrame,
) -> tuple[int, int, bool, pd.Timestamp, bool]:
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:.3rem 0 1rem 0;">
            <div style="font-size:1.4rem; font-weight:900; color:{C['accent']};">⚡ DK1 Energy</div>
            <div style="color:{C['muted']}; font-size:.78rem; margin-top:.2rem;">
                XGBoost · Groq AI · Streamlit
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Data status**")
        try:
            groq_ok = bool(st.secrets.get("GROQ_API_KEY", ""))
        except Exception:
            groq_ok = False

        for label, ok, note in [
            ("Price data",      not prices.empty,      f"{len(prices):,} rows" if not prices.empty else "missing"),
            ("Model forecasts", not raw_preds.empty,   "predictions.parquet"),
            ("Groq AI",         groq_ok,               "API key configured" if groq_ok else "add key to secrets"),
        ]:
            dot = "🟢" if ok else "🔴"
            st.markdown(
                f"{dot} **{label}**  \n"
                f"<span style='color:{C['muted']};font-size:.76rem;'>{note}</span>",
                unsafe_allow_html=True,
            )

        st.divider()

        days_back = st.select_slider(
            "History window",
            options=[30, 60, 90, 180, 365],
            value=90,
            format_func=lambda x: f"{x} days",
        )
        ctx_days = st.select_slider(
            "Context chart — days of history",
            options=[3, 5, 7, 14, 21, 30],
            value=7,
            format_func=lambda x: f"{x} days",
        )

        # Forecast window date picker
        st.divider()
        st.markdown("**Forecast window**")
        if not raw_preds.empty:
            issue_dates = sorted(raw_preds["issue_time"].dt.normalize().unique())
            min_d = issue_dates[0].date()
            max_d = issue_dates[-1].date()
            picked = st.date_input(
                "Issue date",
                value=max_d,
                min_value=min_d,
                max_value=max_d,
                help="Select any date to view the 5-day forecast issued on that day.",
            )
            selected_issue = pd.Timestamp(picked)
        else:
            selected_issue = None
        show_actuals = st.checkbox("Show actual prices in forecast period", value=True)

        show_raw = st.checkbox("Show raw data tables", value=False)

        if not metrics.empty:
            st.divider()
            st.markdown("**XGBoost accuracy (3 recent folds)**")
            recent = metrics.tail(3)
            for day in range(1, 6):
                col_name = f"day{day}_mae"
                if col_name in recent.columns:
                    mae = recent[col_name].mean()
                    st.markdown(
                        f"<span style='color:{C['muted']};font-size:.76rem;'>Day {day}</span> "
                        f"**{mae:.1f}** "
                        f"<span style='color:{C['muted']};font-size:.76rem;'>EUR/MWh MAE</span>",
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.caption("Energi Data Service · DK1 West Denmark\nPrices in DKK/MWh")

    return days_back, ctx_days, show_raw, selected_issue, show_actuals


# ── Main dashboard ────────────────────────────────────────────────────────────
def main() -> None:
    setup_page()

    with st.spinner("Loading data…"):
        prices  = load_prices()
        raw_preds = load_raw_predictions()
        metrics = load_metrics()

    eur_dkk = compute_eur_dkk(prices)

    # Sidebar first so selected_issue is known before building forecasts
    days_back, ctx_days, show_raw, selected_issue, show_actuals = render_sidebar(
        prices, pd.DataFrame(), metrics, raw_preds
    )

    bands       = compute_forecast_bands(raw_preds, eur_dkk)
    fcst        = process_predictions(raw_preds, eur_dkk, selected_issue)
    hourly_fcst = process_predictions_hourly(raw_preds, eur_dkk, selected_issue, bands)
    hist        = daily_history(prices)

    # Anchor "today" to the model's latest issue date for a consistent time story
    if not fcst.empty:
        today_date = fcst["issue_date"].iloc[0]
    elif not prices.empty:
        today_date = prices["Date"].max()
    else:
        today_date = None

    today_prices = (
        prices[prices["Date"] == today_date]
        if today_date is not None and not prices.empty
        else pd.DataFrame()
    )
    # Aggregate to hourly so KPIs match the chart (raw data has 15-min intervals)
    today_hourly = (
        today_prices.groupby("Hour", as_index=False)["PriceDKK"].mean()
        if not today_prices.empty else pd.DataFrame()
    )
    today_avg = today_hourly["PriceDKK"].mean() if not today_hourly.empty else np.nan
    today_min = today_hourly["PriceDKK"].min()  if not today_hourly.empty else np.nan
    today_max = today_hourly["PriceDKK"].max()  if not today_hourly.empty else np.nan
    fcst_avg  = fcst["pred_dkk"].mean()          if not fcst.empty         else np.nan

    # ── Header ────────────────────────────────────────────────────────────────
    date_str  = today_date.strftime("%A, %d %B %Y") if today_date else "n/a"
    issue_str = fcst["issue_date"].iloc[0].strftime("%d %b %Y") if not fcst.empty else "n/a"

    st.markdown(f"""
    <h1 style="margin:0; font-size:2.1rem; font-weight:900; color:{C['text']};">
        ⚡ DK1 Energy Planner
    </h1>
    <p style="color:{C['muted']}; margin:.3rem 0 1.2rem 0; font-size:.88rem;">
        Reference date:&nbsp;<strong style="color:{C['accent']};">{date_str}</strong>
        &nbsp;·&nbsp;XGBoost model run:&nbsp;<strong>{issue_str}</strong>
        &nbsp;·&nbsp;Price area:&nbsp;<strong>DK1 West Denmark</strong>
    </p>
    """, unsafe_allow_html=True)

    if prices.empty:
        st.error("Price data not found. Add `data/day_ahead_prices_dk1_raw.csv`.")
        return

    # ── KPI cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, unit, color in [
        (k1, "Today's average price",     today_avg, "DKK/MWh",               C["price"]),
        (k2, "Today's minimum",           today_min, "DKK/MWh",               C["good"]),
        (k3, "Today's maximum",           today_max, "DKK/MWh",               C["warn"]),
        (k4, "5-day forecast avg · XGBoost", fcst_avg, "DKK/MWh predicted",   C["fcst"]),
    ]:
        val_str = f"{value:.0f}" if pd.notna(value) else "–"
        col.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-val" style="color:{color};">{val_str}</div>
            <div class="kpi-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

    # ── Today's hourly + historic daily ──────────────────────────────────────
    st.markdown("<div class='section'>Current day &amp; historical prices</div>", unsafe_allow_html=True)
    c_tod, c_hist_col = st.columns(2)

    with c_tod:
        if today_prices.empty:
            st.info("No hourly price data available for the reference date.")
        else:
            st.plotly_chart(
                fig_today(today_hourly, today_date),
                use_container_width=True, config={"displayModeBar": False},
            )

    with c_hist_col:
        hist_filtered = hist[hist["Date"] <= today_date] if today_date is not None else hist
        if hist_filtered.empty:
            st.info("No historical price data available.")
        else:
            st.plotly_chart(
                fig_history(hist_filtered, days_back),
                use_container_width=True, config={"displayModeBar": False},
            )

    # ── XGBoost 5-day forecast ────────────────────────────────────────────────
    st.markdown("<div class='section'>XGBoost 5-day spot price forecast</div>", unsafe_allow_html=True)

    if fcst.empty:
        st.info("No forecast data found. Add `outputs/model/predictions.parquet`.")
    else:
        render_forecast_cards(fcst, today_avg)

        st.markdown("<div style='height:.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section'>Historical context + XGBoost forecast</div>", unsafe_allow_html=True)

        st.plotly_chart(
            fig_context(prices, hourly_fcst, ctx_days, cutoff=today_date, show_actuals=show_actuals),
            use_container_width=True, config={"displayModeBar": False},
        )

    # ── Feature importance + SHAP ─────────────────────────────────────────────
    imp_df              = load_feature_importance()
    shap_vals, shap_feat = load_shap_data()

    has_imp  = not imp_df.empty
    has_shap = not shap_vals.empty and not shap_feat.empty

    if has_imp or has_shap:
        st.markdown("<div class='section'>XGBoost model insights · Day 1 model</div>", unsafe_allow_html=True)
        fi_col, shap_col = st.columns(2)

        with fi_col:
            if has_imp:
                st.plotly_chart(
                    fig_feature_importance(imp_df),
                    use_container_width=True, config={"displayModeBar": False},
                )
            else:
                st.info("Feature importance unavailable — run `XG_Boost_full_Res.py` to generate `final_day_models.joblib`.")

        with shap_col:
            if has_shap:
                st.plotly_chart(
                    fig_shap(shap_vals, shap_feat),
                    use_container_width=True, config={"displayModeBar": False},
                )
            else:
                st.info("SHAP data unavailable — run `XG_Boost_full_Res.py` to generate `shap_day1_*.parquet`.")

    # ── Weather explorer ──────────────────────────────────────────────────────
    weather = load_weather_data()
    if not weather.empty:
        st.markdown("<div class='section'>Weather explorer · DK1 West · actual vs forecast</div>", unsafe_allow_html=True)

        # Variable toggles
        tog_cols = st.columns(4)
        show_wind      = tog_cols[0].toggle("Wind 100m",   value=True, key="w_wind")
        show_solar     = tog_cols[1].toggle("Solar",        value=True, key="w_solar")
        show_temp      = tog_cols[2].toggle("Temperature",  value=True, key="w_temp")
        show_w_actuals = tog_cols[3].toggle("Show actuals in forecast window", value=True, key="w_actuals")

        selected_weather = [
            v for v, on in [
                ("Wind 100m",   show_wind),
                ("Solar",       show_solar),
                ("Temperature", show_temp),
            ] if on
        ]

        # History window — reuses the same issue_ts as the XGBoost forecast
        w_days = st.select_slider(
            "Weather history window",
            options=[3, 5, 7, 14, 21, 30],
            value=7,
            format_func=lambda x: f"{x} days",
            key="w_days",
        )

        # Resolve the issue timestamp (same logic as the price forecast)
        w_issue_ts = _resolve_issue_time(raw_preds, selected_issue) if not raw_preds.empty else pd.Timestamp(weather["issue_time"].max())

        if selected_weather:
            st.plotly_chart(
                fig_weather(weather, selected_weather, w_days, w_issue_ts, show_w_actuals),
                use_container_width=True, config={"displayModeBar": False},
            )
        else:
            st.info("Select at least one weather variable above.")

    # ── Groq AI section ───────────────────────────────────────────────────────
    st.markdown("<div class='section'>AI insights · Groq Llama 3.3 70B</div>", unsafe_allow_html=True)

    ai_l, ai_r = st.columns(2)

    with ai_l:
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:.95rem; font-weight:700; color:{C['accent']};"
                f" margin-bottom:.5rem;'>🔮 Forecast reasoning summary</div>",
                unsafe_allow_html=True,
            )
            if fcst.empty or pd.isna(today_avg):
                st.info("Forecast data required.")
            else:
                if st.button("Generate forecast reasoning", key="btn_r", type="primary"):
                    with st.spinner("Asking Groq…"):
                        text = groq_call(prompt_reasoning(fcst, today_avg, fcst["issue_date"].iloc[0], hourly_fcst, weather))
                        st.session_state["_r"] = text

                if "_r" in st.session_state:
                    if st.session_state["_r"]:
                        st.markdown(st.session_state["_r"])
                    else:
                        st.warning("Groq API unreachable — add `GROQ_API_KEY` to Streamlit secrets.")
                else:
                    st.caption(
                        "Generates a 4-bullet AI analysis of what market dynamics are driving "
                        "the 5-day price pattern (wind, demand, seasonal effects, etc.)."
                    )

    with ai_r:
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:.95rem; font-weight:700; color:{C['good']};"
                f" margin-bottom:.5rem;'>🏠 Household guidance</div>",
                unsafe_allow_html=True,
            )
            if fcst.empty or pd.isna(today_avg):
                st.info("Forecast data required.")
            else:
                if st.button("Generate household tips", key="btn_h", type="primary"):
                    with st.spinner("Asking Groq…"):
                        text = groq_call(prompt_household(fcst, today_avg, hourly_fcst))
                        st.session_state["_h"] = text

                if "_h" in st.session_state:
                    if st.session_state["_h"]:
                        st.markdown(st.session_state["_h"])
                    else:
                        st.warning("Groq API unreachable — add `GROQ_API_KEY` to Streamlit secrets.")
                else:
                    st.caption(
                        "Generates actionable tips on when to run your EV charger, dishwasher, "
                        "heat pump, and other flexible loads based on the 5-day price forecast."
                    )

    # ── Raw data tables ───────────────────────────────────────────────────────
    if show_raw:
        st.markdown("<div class='section'>Raw data</div>", unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["Price history (last 100)", "XGBoost forecast", "Model metrics"])
        with t1:
            st.dataframe(prices.tail(100), use_container_width=True)
        with t2:
            st.dataframe(fcst, use_container_width=True)
        with t3:
            st.dataframe(metrics, use_container_width=True)


if __name__ == "__main__":
    main()
