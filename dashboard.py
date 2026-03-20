from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Streamlit dashboard skeleton for internal DK1 household users
# Uses existing raw CSV files from the repository.
#
# Expected files in repo root /data:
#   - day_ahead_prices_dk1_raw.csv
#   - consumption_dk1_raw.csv
#   - supply_forecasts_dk1_raw.csv
#
# Run with:
#   streamlit run streamlit_energy_dashboard.py
# ============================================================

DATA_DIR = Path("data")
PRICE_FILE = DATA_DIR / "day_ahead_prices_dk1_raw.csv"
CONSUMPTION_FILE = DATA_DIR / "consumption_dk1_raw.csv"
SUPPLY_FILE = DATA_DIR / "supply_forecasts_dk1_raw.csv"

REFRESH_FREQUENCY = "Daily"
PRICE_AREA = "DK1"

COLORS = {
    "bg": "#f3f4f6",
    "card": "#ffffff",
    "text": "#1f2937",
    "muted": "#6b7280",
    "price": "#dc2626",
    "price_soft": "#fca5a5",
    "wind_onshore": "#0ea5a4",
    "wind_offshore": "#2563eb",
    "solar": "#f59e0b",
    "consumption": "#7c3aed",
    "good": "#059669",
    "warn": "#dc2626",
    "neutral": "#475569",
}


def apply_page_style() -> None:
    st.set_page_config(
        page_title="DK1 Energy Planner",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {COLORS['bg']};
                color: {COLORS['text']};
            }}
            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 1.2rem;
            }}
            .metric-card {{
                background: {COLORS['card']};
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 3px 14px rgba(15, 23, 42, 0.08);
                min-height: 120px;
            }}
            .section-card {{
                background: {COLORS['card']};
                border-radius: 18px;
                padding: 1rem 1.2rem 0.7rem 1.2rem;
                box-shadow: 0 3px 14px rgba(15, 23, 42, 0.08);
                margin-bottom: 1rem;
            }}
            .small-muted {{
                color: {COLORS['muted']};
                font-size: 0.88rem;
            }}
            .recommend-good {{
                color: {COLORS['good']};
                font-weight: 600;
            }}
            .recommend-warn {{
                color: {COLORS['warn']};
                font-weight: 600;
            }}
            .recommend-neutral {{
                color: {COLORS['neutral']};
                font-weight: 600;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "TimeDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
    elif "TimeUTC" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeUTC"], errors="coerce")

    numeric_cols = ["DayAheadPriceEUR", "DayAheadPriceDKK"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "PriceArea" in df.columns:
        df = df[df["PriceArea"] == PRICE_AREA].copy()

    df = df.dropna(subset=["TimeDK"]).sort_values("TimeDK").reset_index(drop=True)
    df["Date"] = df["TimeDK"].dt.date
    return df


@st.cache_data(show_spinner=False)
def load_consumption(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "TimeDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
    elif "Date" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["Date"], errors="coerce")

    if "ConsumptionkWh" in df.columns:
        df["ConsumptionkWh"] = pd.to_numeric(df["ConsumptionkWh"], errors="coerce")

    if "PriceArea" in df.columns:
        df = df[df["PriceArea"] == PRICE_AREA].copy()

    df = df.dropna(subset=["TimeDK"]).sort_values("TimeDK").reset_index(drop=True)
    df["Date"] = df["TimeDK"].dt.date
    return df


@st.cache_data(show_spinner=False)
def load_supply(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "HourDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["HourDK"], errors="coerce")
    elif "HourUTC" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["HourUTC"], errors="coerce")
    else:
        df["TimeDK"] = pd.NaT

    numeric_cols = [
        "ForecastDayAhead",
        "ForecastIntraday",
        "Forecast5Hour",
        "Forecast1Hour",
        "ForecastCurrent",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "PriceArea" in df.columns:
        df = df[df["PriceArea"] == PRICE_AREA].copy()

    df = df.dropna(subset=["TimeDK"]).sort_values("TimeDK").reset_index(drop=True)
    df["Date"] = df["TimeDK"].dt.date
    return df


def compute_daily_price_history(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame(columns=["Date", "AvgPriceDKK"])

    out = (
        price_df.groupby("Date", as_index=False)["DayAheadPriceDKK"]
        .mean()
        .rename(columns={"DayAheadPriceDKK": "AvgPriceDKK"})
    )
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def get_today_hourly_prices(price_df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    if price_df.empty:
        return pd.DataFrame(), None

    latest_ts = price_df["TimeDK"].max()
    today = latest_ts.date()
    today_df = price_df[price_df["Date"] == today].copy()
    return today_df, latest_ts


def choose_supply_value_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "ForecastCurrent",
        "Forecast1Hour",
        "Forecast5Hour",
        "ForecastIntraday",
        "ForecastDayAhead",
    ]
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def build_supply_daily_features(supply_df: pd.DataFrame) -> pd.DataFrame:
    if supply_df.empty or "ForecastType" not in supply_df.columns:
        return pd.DataFrame(columns=["Date", "Solar", "Onshore Wind", "Offshore Wind", "TotalRenewables"])

    value_col = choose_supply_value_column(supply_df)
    if value_col is None:
        return pd.DataFrame(columns=["Date", "Solar", "Onshore Wind", "Offshore Wind", "TotalRenewables"])

    grouped = (
        supply_df.groupby(["Date", "ForecastType"], as_index=False)[value_col]
        .mean()
        .pivot(index="Date", columns="ForecastType", values=value_col)
        .reset_index()
    )

    for col in ["Solar", "Onshore Wind", "Offshore Wind"]:
        if col not in grouped.columns:
            grouped[col] = np.nan

    grouped["TotalRenewables"] = grouped[["Solar", "Onshore Wind", "Offshore Wind"]].sum(axis=1, skipna=True)
    grouped["Date"] = pd.to_datetime(grouped["Date"])
    return grouped.sort_values("Date")


def build_consumption_daily_features(cons_df: pd.DataFrame) -> pd.DataFrame:
    if cons_df.empty or "ConsumptionkWh" not in cons_df.columns:
        return pd.DataFrame(columns=["Date", "AvgConsumptionkWh"])

    out = (
        cons_df.groupby("Date", as_index=False)["ConsumptionkWh"]
        .mean()
        .rename(columns={"ConsumptionkWh": "AvgConsumptionkWh"})
    )
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date")


def build_placeholder_forecast(
    daily_prices: pd.DataFrame,
    daily_supply: pd.DataFrame,
    daily_consumption: pd.DataFrame,
    horizon_days: int = 5,
) -> pd.DataFrame:
    if daily_prices.empty:
        return pd.DataFrame(columns=["Date", "PredictedAvgDKK", "DeltaVsToday", "Reason"])

    history = daily_prices.sort_values("Date").copy()
    latest_date = history["Date"].max()
    today_avg = history.loc[history["Date"] == latest_date, "AvgPriceDKK"].iloc[0]

    recent = history.tail(7)
    base_avg = recent["AvgPriceDKK"].mean()

    supply_recent = daily_supply.sort_values("Date").tail(7) if not daily_supply.empty else pd.DataFrame()
    cons_recent = daily_consumption.sort_values("Date").tail(7) if not daily_consumption.empty else pd.DataFrame()

    renewable_trend = 0.0
    consumption_trend = 0.0

    if not supply_recent.empty and supply_recent["TotalRenewables"].notna().any():
        renewable_trend = supply_recent["TotalRenewables"].tail(3).mean() - supply_recent["TotalRenewables"].head(3).mean()

    if not cons_recent.empty and cons_recent["AvgConsumptionkWh"].notna().any():
        consumption_trend = cons_recent["AvgConsumptionkWh"].tail(3).mean() - cons_recent["AvgConsumptionkWh"].head(3).mean()

    renewable_scale = 0.0 if math.isnan(renewable_trend) else renewable_trend * (-0.015)
    consumption_scale = 0.0 if math.isnan(consumption_trend) else consumption_trend * 0.0008

    rows = []
    for step in range(1, horizon_days + 1):
        drift = (today_avg - base_avg) * 0.15 * step
        placeholder_adjustment = renewable_scale + consumption_scale
        pred = max(0.0, base_avg + drift + placeholder_adjustment)
        date = latest_date + pd.Timedelta(days=step)
        delta = pred - today_avg

        if delta <= -25:
            reason = "Renewable supply looks supportive and prices may soften."
        elif delta >= 25:
            reason = "Recent price momentum and demand pressure suggest higher prices."
        else:
            reason = "Prices look broadly stable with only modest short-term change."

        rows.append(
            {
                "Date": date,
                "PredictedAvgDKK": pred,
                "DeltaVsToday": delta,
                "Reason": reason,
            }
        )

    return pd.DataFrame(rows)


def generate_recommendation_text(forecast_df: pd.DataFrame) -> tuple[str, str, str]:
    if forecast_df.empty:
        return (
            "No forecast available yet.",
            "recommend-neutral",
            "Add more data to activate the household recommendation engine.",
        )

    avg_delta = forecast_df["DeltaVsToday"].mean()
    day1_delta = forecast_df.iloc[0]["DeltaVsToday"]

    if day1_delta > 20 or avg_delta > 15:
        headline = "Frontload flexible electricity use"
        style = "recommend-warn"
        body = (
            "Average prices are expected to increase over the next few days. Consider charging the EV tonight, "
            "running the dishwasher and laundry sooner rather than later, and avoiding a wait-and-see strategy."
        )
    elif day1_delta < -20 or avg_delta < -15:
        headline = "Wait before using flexible loads"
        style = "recommend-good"
        body = (
            "Prices look softer ahead. Delay EV charging, tumble drying, dishwashing, and other flexible household "
            "loads if possible, especially if you can shift them into tomorrow or the next low-price window."
        )
    else:
        headline = "Use smart hourly shifting"
        style = "recommend-neutral"
        body = (
            "The next few days look relatively stable on average. Focus on avoiding the most expensive evening hours "
            "rather than shifting all consumption across days. Overnight and midday use may still offer better value."
        )

    return headline, style, body


def make_price_line_chart(today_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=today_df["TimeDK"],
            y=today_df["DayAheadPriceDKK"],
            mode="lines+markers",
            line=dict(color=COLORS["price"], width=3),
            marker=dict(size=7),
            name="Spot price",
            hovertemplate="%{x|%H:%M}<br>%{y:.1f} DKK/MWh<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Hour",
        yaxis_title="DKK/MWh",
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def make_daily_history_chart(daily_df: pd.DataFrame, days_back: int) -> go.Figure:
    plot_df = daily_df.tail(days_back).copy() if len(daily_df) > days_back else daily_df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["AvgPriceDKK"],
            mode="lines",
            line=dict(color=COLORS["price"], width=3),
            fill="tozeroy",
            fillcolor="rgba(220,38,38,0.12)",
            name="Daily average",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} DKK/MWh<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date",
        yaxis_title="Average DKK/MWh",
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def make_forecast_card_row(forecast_df: pd.DataFrame) -> None:
    cols = st.columns(len(forecast_df))
    for col, (_, row) in zip(cols, forecast_df.iterrows()):
        delta = row["DeltaVsToday"]
        if delta > 0:
            delta_txt = f"↑ {delta:.1f}"
            color = COLORS["warn"]
            emoji = "📈"
        elif delta < 0:
            delta_txt = f"↓ {abs(delta):.1f}"
            color = COLORS["good"]
            emoji = "📉"
        else:
            delta_txt = "→ 0.0"
            color = COLORS["neutral"]
            emoji = "➖"

        col.markdown(
            f"""
            <div class="metric-card">
                <div class="small-muted">{row['Date'].strftime('%a %d %b')}</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: {COLORS['text']}; margin-top: 0.2rem;">
                    {row['PredictedAvgDKK']:.0f}
                </div>
                <div class="small-muted">DKK/MWh</div>
                <div style="margin-top: 0.6rem; color: {color}; font-weight: 700;">
                    {emoji} {delta_txt} vs today
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def make_supply_reasoning_chart(daily_supply: pd.DataFrame) -> go.Figure:
    plot_df = daily_supply.tail(21).copy()
    fig = go.Figure()

    if "Solar" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["Solar"],
                mode="lines",
                name="☀️ Solar",
                line=dict(color=COLORS["solar"], width=3),
            )
        )
    if "Onshore Wind" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["Onshore Wind"],
                mode="lines",
                name="🌿 Onshore wind",
                line=dict(color=COLORS["wind_onshore"], width=3),
            )
        )
    if "Offshore Wind" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["Offshore Wind"],
                mode="lines",
                name="🌊 Offshore wind",
                line=dict(color=COLORS["wind_offshore"], width=3),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date",
        yaxis_title="MWh/h",
        legend_title="Supply source",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def make_consumption_reasoning_chart(daily_consumption: pd.DataFrame) -> go.Figure:
    plot_df = daily_consumption.tail(21).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["Date"],
            y=plot_df["AvgConsumptionkWh"],
            name="🏠 Consumption",
            marker_color=COLORS["consumption"],
            opacity=0.85,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date",
        yaxis_title="Average kWh",
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def render_header(latest_ts: Optional[pd.Timestamp]) -> None:
    st.title("⚡ DK1 Home Energy Planner")
    st.caption(
        "Internal prototype for household-oriented energy planning. Uses raw data already present in the repository and refreshes daily."
    )

    col1, col2, col3 = st.columns([2.2, 1.2, 1.2])
    with col1:
        st.markdown(
            f"""
            <div class="section-card">
                <div style="font-size: 1.15rem; font-weight: 700;">Today’s spot market overview</div>
                <div class="small-muted" style="margin-top: 0.3rem;">
                    Focus area: {PRICE_AREA} · Refresh cadence: {REFRESH_FREQUENCY}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        last_str = latest_ts.strftime("%Y-%m-%d %H:%M") if latest_ts is not None else "n/a"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-muted">Latest data point</div>
                <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.3rem;">{last_str}</div>
                <div class="small-muted">Danish local time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-muted">Audience</div>
                <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.3rem;">Private households</div>
                <div class="small-muted">Simple guidance for flexible home loads</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_data_status(prices: pd.DataFrame, consumption: pd.DataFrame, supply: pd.DataFrame) -> None:
    with st.sidebar:
        st.header("Data status")
        st.write(f"**Prices:** {'Loaded' if not prices.empty else 'Missing'}")
        st.write(f"**Consumption:** {'Loaded' if not consumption.empty else 'Missing'}")
        st.write(f"**Supply forecasts:** {'Loaded' if not supply.empty else 'Missing'}")
        st.caption("This first version reads the raw CSV files already present in the repository.")
        days_back = st.selectbox("History window", options=[30, 90, 180, 365], index=1)
        show_raw_preview = st.checkbox("Show raw data preview", value=False)
    return days_back, show_raw_preview


def render_main_dashboard() -> None:
    apply_page_style()

    prices = load_prices(PRICE_FILE)
    consumption = load_consumption(CONSUMPTION_FILE)
    supply = load_supply(SUPPLY_FILE)

    daily_prices = compute_daily_price_history(prices)
    today_prices, latest_ts = get_today_hourly_prices(prices)
    daily_supply = build_supply_daily_features(supply)
    daily_consumption = build_consumption_daily_features(consumption)
    forecast_df = build_placeholder_forecast(daily_prices, daily_supply, daily_consumption, horizon_days=5)

    days_back, show_raw_preview = render_data_status(prices, consumption, supply)
    render_header(latest_ts)

    if prices.empty:
        st.error("No day-ahead price file was found or could be parsed. Add data/day_ahead_prices_dk1_raw.csv to continue.")
        return

    today_avg = today_prices["DayAheadPriceDKK"].mean() if not today_prices.empty else np.nan
    today_min = today_prices["DayAheadPriceDKK"].min() if not today_prices.empty else np.nan
    today_max = today_prices["DayAheadPriceDKK"].max() if not today_prices.empty else np.nan

    c1, c2, c3 = st.columns(3)
    for c, title, value in [
        (c1, "Today average", today_avg),
        (c2, "Today minimum", today_min),
        (c3, "Today maximum", today_max),
    ]:
        c.markdown(
            f"""
            <div class="metric-card">
                <div class="small-muted">{title}</div>
                <div style="font-size: 1.9rem; font-weight: 800; color: {COLORS['price']}; margin-top: 0.35rem;">
                    {(f"{value:.0f}" if pd.notna(value) else "n/a")}
                </div>
                <div class="small-muted">DKK/MWh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.05, 1.15])

    with left:
        st.markdown('<div class="section-card"><h4>📅 Current day hourly spot price</h4></div>', unsafe_allow_html=True)
        if today_prices.empty:
            st.info("No hourly prices available for the latest day in the file.")
        else:
            st.plotly_chart(make_price_line_chart(today_prices), use_container_width=True)

    with right:
        st.markdown('<div class="section-card"><h4>📚 Historic daily average spot price</h4></div>', unsafe_allow_html=True)
        st.plotly_chart(make_daily_history_chart(daily_prices, days_back=days_back), use_container_width=True)

    st.markdown('<div class="section-card"><h4>🔮 Placeholder prediction: next 1–5 days average price</h4></div>', unsafe_allow_html=True)
    make_forecast_card_row(forecast_df)

    headline, style, body = generate_recommendation_text(forecast_df)
    st.markdown(
        f"""
        <div class="section-card">
            <h4>🏠 What should a household do?</h4>
            <p class="{style}" style="font-size: 1.1rem; margin-bottom: 0.2rem;">{headline}</p>
            <p style="margin-top: 0.2rem;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reasoning_left, reasoning_right = st.columns(2)
    with reasoning_left:
        st.markdown('<div class="section-card"><h4>🌤️ Why the forecast looks like this: supply outlook</h4></div>', unsafe_allow_html=True)
        if daily_supply.empty:
            st.info("No supply forecast file available yet.")
        else:
            st.plotly_chart(make_supply_reasoning_chart(daily_supply), use_container_width=True)
            st.caption("Solar is shown in amber, onshore wind in teal, and offshore wind in blue.")

    with reasoning_right:
        st.markdown('<div class="section-card"><h4>👪 Demand pressure: consumption background</h4></div>', unsafe_allow_html=True)
        if daily_consumption.empty:
            st.info("No consumption file available yet.")
        else:
            st.plotly_chart(make_consumption_reasoning_chart(daily_consumption), use_container_width=True)
            st.caption("This is currently a simple daily-average demand background signal used for intuition, not a full predictive feature view.")

    if not forecast_df.empty:
        st.markdown('<div class="section-card"><h4>🧠 Forecast reasoning summary</h4></div>', unsafe_allow_html=True)
        bullets = "\n".join(
            [f"- **{row.Date.strftime('%a %d %b')}**: {row.Reason}" for _, row in forecast_df.iterrows()]
        )
        st.markdown(bullets)

    if show_raw_preview:
        st.markdown('<div class="section-card"><h4>🔎 Raw data preview</h4></div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Prices", "Consumption", "Supply forecasts"])
        with tab1:
            st.dataframe(prices.tail(20), use_container_width=True)
        with tab2:
            st.dataframe(consumption.tail(20), use_container_width=True)
        with tab3:
            st.dataframe(supply.tail(20), use_container_width=True)


if __name__ == "__main__":
    render_main_dashboard()