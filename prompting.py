import ollama
import pandas as pd

from typing import List, Dict, Any


client = ollama.Client()


def build_weather_comparison(
    weather_df: pd.DataFrame,
    horizon_days: int = 5,
    today_date: pd.Timestamp | None = None,
) -> List[Dict[str, Any]]:
    """
    Compare the next horizon_days of weather against the current displayed day.

    Expected columns in weather_df:
        - Date
        - TempC
        - WindSpeed
        - CloudCover

    Returns a list of dictionaries, one per future day.
    """
    if weather_df.empty:
        return []

    df = weather_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if df.empty:
        return []

    if today_date is None:
        today_date = df["Date"].min()
    else:
        today_date = pd.to_datetime(today_date).normalize()

    today_row = df[df["Date"] == today_date]
    if today_row.empty:
        return []

    today_row = today_row.iloc[0]
    future_df = df[df["Date"] > today_date].head(horizon_days)

    rows: List[Dict[str, Any]] = []

    for _, row in future_df.iterrows():
        item: Dict[str, Any] = {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "temp_vs_today": None,
            "wind_vs_today": None,
            "cloud_vs_today": None,
        }

        if "TempC" in df.columns and pd.notna(row.get("TempC")) and pd.notna(today_row.get("TempC")):
            diff = float(row["TempC"] - today_row["TempC"])
            if diff > 1:
                item["temp_vs_today"] = f"warmer by {diff:.1f} °C"
            elif diff < -1:
                item["temp_vs_today"] = f"colder by {abs(diff):.1f} °C"
            else:
                item["temp_vs_today"] = "similar temperature"

        if "WindSpeed" in df.columns and pd.notna(row.get("WindSpeed")) and pd.notna(today_row.get("WindSpeed")):
            diff = float(row["WindSpeed"] - today_row["WindSpeed"])
            if diff > 1:
                item["wind_vs_today"] = f"windier by {diff:.1f}"
            elif diff < -1:
                item["wind_vs_today"] = f"less windy by {abs(diff):.1f}"
            else:
                item["wind_vs_today"] = "similar wind conditions"

        if "CloudCover" in df.columns and pd.notna(row.get("CloudCover")) and pd.notna(today_row.get("CloudCover")):
            diff = float(row["CloudCover"] - today_row["CloudCover"])
            if diff > 5:
                item["cloud_vs_today"] = f"more cloudy by {diff:.0f} percentage points"
            elif diff < -5:
                item["cloud_vs_today"] = f"less cloudy by {abs(diff):.0f} percentage points"
            else:
                item["cloud_vs_today"] = "similar cloud cover"

        rows.append(item)

    return rows


def build_price_reasoning_prompt(
    forecast_df: pd.DataFrame,
    today_avg: float,
    weather_comparison: List[Dict[str, Any]],
    price_area: str = "DK1",
) -> str:
    """
    Build a prompt for the LLM using the 5-day forecast and weather comparison.
    """
    if forecast_df.empty:
        return "No forecast data available."

    forecast_lines: List[str] = []
    for _, row in forecast_df.iterrows():
        forecast_lines.append(
            f"- {pd.to_datetime(row['Date']).strftime('%Y-%m-%d')}: "
            f"forecast average price = {row['PredictedAvgDKK']:.1f} DKK/MWh "
            f"(delta vs today: {row['DeltaVsToday']:+.1f})"
        )

    if weather_comparison:
        weather_lines: List[str] = []
        for item in weather_comparison:
            parts = [f"- {item['date']}"]
            if item.get("temp_vs_today"):
                parts.append(item["temp_vs_today"])
            if item.get("wind_vs_today"):
                parts.append(item["wind_vs_today"])
            if item.get("cloud_vs_today"):
                parts.append(item["cloud_vs_today"])
            weather_lines.append(": ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0])
        weather_block = "\n".join(weather_lines)
    else:
        weather_block = "- No weather comparison data available."

    prompt = f"""
You are explaining a prototype 5-day electricity price forecast for Danish households in {price_area}.

Today's displayed average electricity price is:
- {today_avg:.1f} DKK/MWh

Forecast for the next 5 days:
{chr(10).join(forecast_lines)}

Weather for the next 5 days compared with today:
{weather_block}

Task:
Explain why the next 5 days of electricity prices might look the way they do.

Important instructions:
- Base the explanation only on the forecast values and the weather comparison given above.
- Use cautious wording such as "may", "might", "could", or "is consistent with".
- Mention weather-related effects such as wind possibly affecting wind power generation, and cloudiness possibly affecting solar generation, when relevant.
- Do not invent market facts not provided in the input.
- Keep the explanation suitable for a household user, but still reasonably analytical.

Return exactly:
1. A short headline
2. A short paragraph explanation
3. A short household recommendation
"""
    return prompt.strip()


def generate_llm_reasoning(prompt: str, model: str = "llama2") -> Dict[str, str]:
    """
    Send the prompt to Ollama and return the raw model text.
    """
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response["message"]["content"]
        return {"raw_text": text}

    except Exception as e:
        return {"raw_text": f"LLM call failed: {e}"}


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Example test data
    # ---------------------------------------------------------
    forecast_df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-03-14", periods=5, freq="D"),
            "PredictedAvgDKK": [720.0, 745.0, 760.0, 735.0, 710.0],
            "DeltaVsToday": [15.0, 40.0, 55.0, 30.0, 5.0],
        }
    )

    weather_df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-03-13", periods=6, freq="D"),
            "TempC": [6.0, 5.5, 4.0, 7.0, 8.0, 6.5],
            "WindSpeed": [7.0, 4.0, 3.5, 6.0, 8.5, 9.0],
            "CloudCover": [50, 75, 80, 60, 35, 40],
        }
    )

    today_avg = 705.0

    weather_comparison = build_weather_comparison(
        weather_df=weather_df,
        horizon_days=5,
        today_date="2026-03-13",
    )

    prompt = build_price_reasoning_prompt(
        forecast_df=forecast_df,
        today_avg=today_avg,
        weather_comparison=weather_comparison,
        price_area="DK1",
    )

    print("\n--- PROMPT SENT TO MODEL ---\n")
    print(prompt)

    result = generate_llm_reasoning(prompt, model="llama2")

    print("\n--- MODEL RESPONSE ---\n")
    print(result["raw_text"])