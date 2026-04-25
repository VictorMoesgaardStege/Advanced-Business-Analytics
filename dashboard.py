from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths & constants ─────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
PRICE_FILE  = DATA_DIR / "day_ahead_prices_dk1_raw.csv"
PRED_FILE   = Path("outputs/model/predictions.parquet")
METRICS_FILE = Path("outputs/model/metrics.csv")
PRICE_AREA  = "DK1"
EUR_DKK_FB  = 7.46   # fallback exchange rate

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
    raw: pd.DataFrame, eur_dkk: float, selected: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Return all 120 hourly predictions for the selected issue_time."""
    if raw.empty:
        return pd.DataFrame()

    issue_ts = _resolve_issue_time(raw, selected)
    sub      = raw[raw["issue_time"] == issue_ts].copy()

    future = sub[sub["target_time"] > issue_ts].sort_values("target_time").head(120)
    future["pred_dkk"]   = future["predicted"]        * eur_dkk
    future["actual_dkk"] = future["DayAheadPriceEUR"] * eur_dkk
    future["issue_date"] = issue_ts.normalize()
    return future[["target_time", "horizon_h", "pred_dkk", "actual_dkk", "issue_date"]].reset_index(drop=True)


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
            max_tokens  = 700,
            temperature = 0.4,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def prompt_reasoning(fcst: pd.DataFrame, today_avg: float, issue_date: pd.Timestamp) -> str:
    rows = "\n".join(
        f"  • {r.Date.strftime('%a %d %b')}: {r.pred_dkk:.0f} DKK/MWh  "
        f"({r.pred_dkk - today_avg:+.0f} vs today)"
        for _, r in fcst.iterrows()
    )
    return f"""You are an electricity market analyst specialising in Nordic energy markets.

DK1 (West Denmark) XGBoost forecast, issued {issue_date.strftime('%d %b %Y')}:
Today's average: {today_avg:.0f} DKK/MWh

5-day outlook:
{rows}

The model's top features: wind speed (10 m & 100 m), shortwave radiation, cloud cover, \
temperature, pressure, hour-of-day, day-of-week, month, spot-price lags at 24 h, 48 h, 168 h.

Write exactly 4 bullet points (each starting with •) explaining what energy-market dynamics \
are most likely driving this specific 5-day price pattern. Be concise, factual, and specific \
to the DK1 market. No preamble or closing sentence."""


def prompt_household(fcst: pd.DataFrame, today_avg: float) -> str:
    rows = "\n".join(
        f"  • {r.Date.strftime('%A %d %b')}: {r.pred_dkk:.0f} DKK/MWh  "
        f"({'↑ higher' if r.pred_dkk - today_avg > 20 else '↓ lower' if r.pred_dkk - today_avg < -20 else '→ similar'})"
        for _, r in fcst.iterrows()
    )
    return f"""You are an energy advisor helping a Danish household cut their electricity bill.

DK1 spot price forecast for the next 5 days:
Today: {today_avg:.0f} DKK/MWh
{rows}

Write 4-5 bullet points (each starting with •) of specific, actionable advice for:
- EV charging (best day(s) and time of day)
- Dishwasher and washing machine
- Heat pump or electric heating
- Any other flexible high-consumption appliance

Name the specific days from the forecast. Be direct and practical. No preamble."""


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
            <div class="kpi-unit">DKK/MWh</div>
            <div style="margin-top:.55rem;">{tag} vs today</div>
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
            ("Price data",      not prices.empty, f"{len(prices):,} rows" if not prices.empty else "missing"),
            ("Model forecasts", not fcst.empty,   "predictions.parquet"),
            ("Groq AI",         groq_ok,          "API key configured" if groq_ok else "add key to secrets"),
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

    fcst        = process_predictions(raw_preds, eur_dkk, selected_issue)
    hourly_fcst = process_predictions_hourly(raw_preds, eur_dkk, selected_issue)
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
    today_avg = today_prices["PriceDKK"].mean() if not today_prices.empty else np.nan
    today_min = today_prices["PriceDKK"].min()  if not today_prices.empty else np.nan
    today_max = today_prices["PriceDKK"].max()  if not today_prices.empty else np.nan
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
            hourly = today_prices.groupby("Hour", as_index=False)["PriceDKK"].mean()
            st.plotly_chart(
                fig_today(hourly, today_date),
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
                        text = groq_call(prompt_reasoning(fcst, today_avg, fcst["issue_date"].iloc[0]))
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
                        text = groq_call(prompt_household(fcst, today_avg))
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
