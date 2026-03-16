from typing import Dict

import streamlit as st
from google import genai


client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def generate_llm_reasoning(
    prompt: str,
    model: str = "gemini-2.5-flash",
) -> Dict[str, str]:

    try:
        print("PROMPT LENGTH:", len(prompt))
        print("PROMPT PREVIEW:", prompt[:500])

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        text = (response.text or "").strip()
        print("GEMINI RAW RESPONSE:", text)

        return {"raw_text": text}

    except Exception as e:
        print(f"GEMINI ERROR FULL: {repr(e)}")

        return {
            "raw_text": (
                '{"headline":"Fallback",'
                '"style":"recommend-neutral",'
                '"explanation":"The AI reasoning service is temporarily unavailable.",'
                '"actions":["Use the dashboard trend as a guide for now."],'
                '"summary_bullets":["Gemini call failed"]}'
            )
        }