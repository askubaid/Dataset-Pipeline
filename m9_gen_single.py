import os
import json
import time
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Inference engine: 'API' (Gemini) or 'LOCAL' (Ollama) , 'VERTEX' is only for API mode
    "ENGINE": "LOCAL",

    # Range of themes to process (1-indexed, inclusive)
    "STARTING_FROM": 47,
    "UPTILL_TO": 130,

    # Language
    "SOURCE_LANG": "English",

    # Total sentences to generate per theme
    "SENTENCES_PER_THEME": 160,

    # Batch sizes (sentences requested per LLM call)
    "API_BATCH_SIZE": 60,
    "LOCAL_BATCH_SIZE": 40,

    # Max retries per failed batch
    "MAX_RETRIES": 3,

    # Local Ollama config
    "LOCAL_MODEL": "gemma4:e4b",
    "OLLAMA_URL": "http://localhost:11434/api/generate",

    # Output file
    "FILE_OUTPUT": "learner_single.en",
}
# ==========================================


# Pydantic schema for strict Gemini API output
class Sentence(BaseModel):
    text: str = Field(description=f"Sentence in the source language")

class DatasetResponse(BaseModel):
    sentences: List[Sentence]


def get_api_client(engine):
    """Initialize the correct GenAI client based on the chosen engine."""
    if engine == "VERTEX":
        return genai.Client(vertexai=True)
    
    elif engine == "API":
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            print("ERROR: Please set your GEMINI_API_KEY in the .env file to use API engine.")
            exit(1)
        return genai.Client(api_key=api_key)
    
    return None


def generate_via_api(client, prompt, engine):
    """Call Gemini (via either AI Studio or Vertex AI) and return a list of dicts."""
    model_name = "gemini-3.5-flash" if engine == "VERTEX" else "gemini-3.1-flash-lite"
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DatasetResponse,
                temperature=0.7,
            ),
        )
        data = json.loads(response.text)
        return data.get("sentences", [])
    except Exception as e:
        print(f"    {engine} Error: {e}")
        return []


def generate_via_local(prompt):
    """Call local Ollama model and return a list of sentence dicts."""
    payload = {
        "model": CONFIG["LOCAL_MODEL"],
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
    try:
        response = requests.post(CONFIG["OLLAMA_URL"], json=payload, timeout=None)
        if response.status_code != 200:
            print(f"    Ollama HTTP Error {response.status_code}: {response.text}")
            return []

        result_text = response.json().get("response", "")
        data = json.loads(result_text)

        if isinstance(data, dict) and "sentences" in data:
            return data["sentences"]
        elif isinstance(data, list):
            return data
        else:
            print(f"    Unexpected JSON format from local model: {data}")
            return []

    except json.JSONDecodeError as e:
        print(f"    JSON decode error: {e}")
        return []
    except Exception as e:
        print(f"    Ollama Connection Error: {e}")
        return []


def build_prompt(theme_name, subtopic, batch_size, nc, source_lang, sample_1, sample_2):
    """Build the prompt for a single batch, focused on a specific sub-topic."""

    if nc == "low":
        length_instruction = "Each sentence should be around 10 to 15 tokens (words) long."
    elif nc == "medium":
        length_instruction = "Each text should be around 30 tokens (words) long and can be multi-sentence."
    else:
        length_instruction = "Each text should be around 60 tokens (words) long. It can be a short story or coherent paragraph."

    examples_text = ""
    if sample_1 or sample_2:
        examples_text = f"\nHere are examples of the style and length expected:\n"
        if sample_1:
            examples_text += f"- {sample_1}\n"
        if sample_2:
            examples_text += f"- {sample_2}\n"

    prompt = (
        f"You are a professional dataset generator for language learners.\n"
        f"Theme: {theme_name}\n"
        f"Sub-topic: {subtopic}\n"
        f"Task: Generate exactly {batch_size} unique, coherent, and meaningful sentences.\n"
        f"Each sentence must be in {source_lang}.\n"
        f"Focus exclusively on the sub-topic '{subtopic}' within the theme '{theme_name}'.\n"
        f"Length constraint: {length_instruction}\n"
        f"All sentences must be distinct — no duplicates.{examples_text}\n"
    )

    return prompt


def append_sentences_to_file(sentences, file_output):
    """Append validated sentences to the dataset file. Returns count of written sentences."""
    written = 0
    with open(file_output, "a", encoding="utf-8") as f_out:
        for item in sentences:
            # Handle both string arrays and object arrays with "text" key just in case
            if isinstance(item, dict):
                text = item.get("text", "").replace('\\n', ' ').replace('\n', ' ').strip()
            elif isinstance(item, str):
                text = item.replace('\\n', ' ').replace('\n', ' ').strip()
            else:
                continue

            if text:
                f_out.write(f"{text}\n")
                written += 1
    return written


def main():
    engine = CONFIG["ENGINE"]
    batch_size = CONFIG["API_BATCH_SIZE"] if engine == "API" or engine == "VERTEX" else CONFIG["LOCAL_BATCH_SIZE"]
    total_per_theme = CONFIG["SENTENCES_PER_THEME"]

    print(f"=== Single Dataset Generation ===")
    print(f"Engine: {engine} | Batch size: {batch_size} | Target per theme: {total_per_theme}")
    print()

    # 1. Load Themes
    themes_file = "themes.json"
    try:
        with open(themes_file, "r", encoding="utf-8") as f:
            themes = json.load(f)
    except FileNotFoundError:
        print(f"Error: {themes_file} not found.")
        return

    start_idx = max(0, CONFIG["STARTING_FROM"] - 1)
    end_idx = CONFIG["UPTILL_TO"]
    themes_to_process = themes[start_idx:end_idx]

    # 2. Setup Client if needed
    client = None
    if engine in ["API", "VERTEX"]:
        client = get_api_client(engine)

    # 3. Process each theme
    for t_idx, theme in enumerate(themes_to_process):
        theme_name = theme.get("name", "Unknown")
        nc = theme.get("nc", "").lower()
        sample_1 = theme.get("sample_1", "")
        sample_2 = theme.get("sample_2", "")
        subtopics = theme.get("topics", [])

        if not subtopics:
            print(f"[Theme {t_idx+1}] '{theme_name}' has no sub-topics! Run granularize.py first. Skipping.")
            continue

        sentences_per_subtopic = total_per_theme // len(subtopics)
        remainder = total_per_theme % len(subtopics)

        print(f"{'='*60}")
        print(f"[Theme {t_idx+1}/{len(themes_to_process)}] '{theme_name}' | NC: {nc}")
        print(f"  Sub-topics: {len(subtopics)} | ~{sentences_per_subtopic} sentences each")
        print(f"{'='*60}")

        theme_total_written = 0

        for s_idx, subtopic in enumerate(subtopics):
            target_for_subtopic = sentences_per_subtopic + (remainder if s_idx == len(subtopics) - 1 else 0)
            generated_for_subtopic = 0

            print(f"\n  [{s_idx+1}/{len(subtopics)}] Sub-topic: '{subtopic}' (target: {target_for_subtopic})")

            while generated_for_subtopic < target_for_subtopic:
                remaining = target_for_subtopic - generated_for_subtopic
                current_batch = min(batch_size, remaining)

                prompt = build_prompt(
                    theme_name, subtopic, current_batch, nc,
                    CONFIG["SOURCE_LANG"], sample_1, sample_2
                )

                if engine == "LOCAL":
                    prompt += (
                        f"\nYou MUST return ONLY a valid JSON object containing a 'sentences' array.\n"
                        f"Each item must have a 'text' key.\n"
                        f'Example: {{"sentences": [{{"text": "Hello world"}}]}}'
                    )

                sentences = []
                for attempt in range(1, CONFIG["MAX_RETRIES"] + 1):
                    if engine in ["API", "VERTEX"]:
                        sentences = generate_via_api(client, prompt, engine)
                        time.sleep(1)
                    else:
                        sentences = generate_via_local(prompt)

                    if sentences:
                        break
                    else:
                        print(f"    Attempt {attempt}/{CONFIG['MAX_RETRIES']} failed. Retrying...")
                        time.sleep(2)

                if not sentences:
                    print(f"    All {CONFIG['MAX_RETRIES']} attempts failed for this batch. Moving on.")
                    continue

                written = append_sentences_to_file(
                    sentences,
                    CONFIG["FILE_OUTPUT"]
                )

                generated_for_subtopic += written
                theme_total_written += written
                print(f"    Batch done: +{written} sentences | Sub-topic progress: {generated_for_subtopic}/{target_for_subtopic} | Theme total: {theme_total_written}")

        print(f"\n  >>> Theme '{theme_name}' complete: {theme_total_written} total sentences written.\n")

    print("=== Dataset generation complete! ===")

if __name__ == "__main__":
    main()
