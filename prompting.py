from typing import Dict
import streamlit as st
from google import genai
from google.genai import types

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "style": {
            "type": "string",
            "enum": ["recommend-good", "recommend-warn", "recommend-neutral"],
        },
        "explanation": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
        "summary_bullets": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
    },
    "required": ["headline", "style", "explanation", "actions", "summary_bullets"],
}

def generate_llm_reasoning(
    prompt: str,
    model: str = "gemini-2.5-flash",
) -> Dict[str, str]:
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RECOMMENDATION_SCHEMA,
            ),
        )
        text = (response.text or "").strip()
        print("GEMINI RAW RESPONSE:", text)
        return {"raw_text": text}
    except Exception as e:
        print(f"GEMINI ERROR: {e}")
        return {
            "raw_text": (
                '{"headline":"Fallback",'
                '"style":"recommend-neutral",'
                '"explanation":"The AI reasoning service is temporarily unavailable.",'
                '"actions":["Use the dashboard trend as a guide for now."],'
                '"summary_bullets":["Gemini call failed"]}'
            )
        }
        