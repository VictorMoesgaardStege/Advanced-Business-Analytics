from typing import Dict

import streamlit as st
from groq import Groq


def generate_llm_reasoning(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
) -> Dict[str, str]:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    try:
        print("PROMPT LENGTH:", len(prompt))
        print("PROMPT PREVIEW:", prompt[:500])

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        text = (response.choices[0].message.content or "").strip()
        print("GROQ RAW RESPONSE:", text)

        return {"raw_text": text}

    except Exception as e:
        print(f"GROQ ERROR FULL: {repr(e)}")

        return {
            "raw_text": (
                '{"headline":"Fallback",'
                '"style":"recommend-neutral",'
                '"explanation":"The AI reasoning service is temporarily unavailable.",'
                '"actions":["Use the dashboard trend as a guide for now."],'
                '"summary_bullets":["Groq call failed"]}'
            )
        }
