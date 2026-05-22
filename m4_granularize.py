# granularize.py
# -------------------------------------------------
# Adds a "topics" array (20 sub‑topics) to every
# theme object inside themes.json.
# -------------------------------------------------

import os
import json
import time
from typing import List

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
CONFIG = {
    # Choose inference engine: "API" (Gemini) or "LOCAL" (Ollama)
    "ENGINE": "LOCAL",

    # How many sub‑topics to generate per theme
    "SUBTOPICS_PER_THEME": 20,

    # ------------------------------
    # Gemini API settings (used when ENGINE == "API")
    # ------------------------------
    "GEMINI_MODEL": "gemini-2.5-flash",   # paid Gemini model
    # .env file must contain GEMINI_API_KEY = your key

    # ------------------------------
    # Ollama settings (used when ENGINE == "LOCAL")
    # ------------------------------
    "LOCAL_MODEL": "gemma4:e4b",           # local model name
    "OLLAMA_URL": "http://localhost:11434/api/generate",

    # Rate‑limit safety (seconds between calls)
    "SLEEP_SECONDS": 1,
}
# -------------------------------------------------

# ----------------------------------------------------------------------
# Helper: load .env (only needed for Gemini API)
# ----------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------------------------------------------------------------
# Gemini API generation (requires google‑genai package)
# ----------------------------------------------------------------------
if CONFIG["ENGINE"] == "API":
    try:
        from google import genai
        from google.genai import types
        from pydantic import BaseModel

        client = genai.Client(api_key=GEMINI_API_KEY)

        class SubTopicResponse(BaseModel):
            topics: List[str]

        def generate_subtopics_api(theme_name: str, category: str) -> List[str]:
            """Ask Gemini for 20 distinct sub‑topics for a given theme."""
            prompt = (
                f"You are a curriculum designer for language learners.\n"
                f"Generate exactly {CONFIG['SUBTOPICS_PER_THEME']} short, distinct sub‑topics (2‑5 words each) "
                f"for the theme \"{theme_name}\" (category: {category}).\n"
                f"Return only a JSON object with a single key \"topics\" containing the list.\n"
            )
            try:
                response = client.models.generate_content(
                    model=CONFIG["GEMINI_MODEL"],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SubTopicResponse,
                        temperature=0.8,
                    ),
                )
                data = json.loads(response.text)
                return data.get("topics", [])[: CONFIG["SUBTOPICS_PER_THEME"]]
            except Exception as e:
                print(f"  Gemini error for '{theme_name}': {e}")
                return []
    except Exception as exc:
        raise RuntimeError(
            "Gemini SDK not available. Install with `pip install google-genai` "
            "or switch ENGINE to \"LOCAL\"."
        ) from exc
# ----------------------------------------------------------------------
# Ollama generation (local model)
# ----------------------------------------------------------------------
else:
    import requests

    def generate_subtopics_local(theme_name: str, category: str) -> List[str]:
        """Ask Ollama for 20 distinct sub‑topics."""
        prompt = (
            f"You are a curriculum designer for language learners.\n"
            f"Generate exactly {CONFIG['SUBTOPICS_PER_THEME']} short, distinct sub‑topics (2‑5 words each) "
            f"for the theme \"{theme_name}\" (category: {category}).\n"
            f"Return ONLY a valid JSON object with a single key \"topics\" containing the list."
        )
        payload = {
            "model": CONFIG["LOCAL_MODEL"],
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.8},
        }
        try:
            resp = requests.post(CONFIG["OLLAMA_URL"], json=payload, timeout=None)
            if resp.status_code != 200:
                print(f"Ollama HTTP {resp.status_code}: {resp.text}")
                return []
            result = resp.json().get("response", "")
            data = json.loads(result)
            return data.get("topics", [])[: CONFIG["SUBTOPICS_PER_THEME"]]
        except Exception as e:
            print(f"  Ollama error for '{theme_name}': {e}")
            return []


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
THEMES_FILE = "themes.json"

def main():
    # Load existing themes
    try:
        with open(THEMES_FILE, "r", encoding="utf-8") as f:
            themes = json.load(f)
    except FileNotFoundError:
        print(f"Error: {THEMES_FILE} not found.")
        return

    total = len(themes)
    updated = 0

    for idx, theme in enumerate(themes, start=1):
        # Skip if already has a full topics list
        if theme.get("topics") and len(theme["topics"]) >= CONFIG["SUBTOPICS_PER_THEME"]:
            print(f"[{idx}/{total}] Already granularized: {theme['name']}")
            continue

        print(f"[{idx}/{total}] Generating sub‑topics for: {theme['name']} ...")
        if CONFIG["ENGINE"] == "API":
            topics = generate_subtopics_api(theme.get("name", ""), theme.get("category", ""))
        else:
            topics = generate_subtopics_local(theme.get("name", ""), theme.get("category", ""))

        if not topics:
            print("  -> No topics generated (will retry on next run).")
            continue

        theme["topics"] = topics
        updated += 1

        # Save progressively after each successful theme
        with open(THEMES_FILE, "w", encoding="utf-8") as f:
            json.dump(themes, f, indent=2, ensure_ascii=False)

        print(f"  -> Added {len(topics)} topics.")
        time.sleep(CONFIG["SLEEP_SECONDS"])

    print(f"\nFinished. {updated} theme(s) now contain a 'topics' array.")
    print("Run `gen_dataset.py` afterwards – it will use these sub‑topics for batching.")


if __name__ == "__main__":
    main()