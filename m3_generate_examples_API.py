import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_api_key_here":
    print("ERROR: Please set your GEMINI_API_KEY in the .env file.")
    exit(1)

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

# Define the expected JSON output structure using Pydantic
class Samples(BaseModel):
    sample_1: str
    sample_2: str

def generate_samples_for_theme(theme):
    nc = theme.get("nc", "").lower()
    name = theme.get("name", "")
    
    # Determine the prompt logic based on NC level
    if nc == "low":
        instructions = f"Create two sample English sentences, sample_1 and sample_2. These sentences should be regarding the theme name '{name}' and should be around 15 tokens (words) long each."
    elif nc == "medium":
        instructions = f"Create two samples, sample_1 and sample_2. These samples should be multi-sentence, regarding the theme name '{name}', and should be up to 30 tokens (words) long each."
    else: # high or default
        instructions = f"Create two samples, sample_1 and sample_2. These samples should be short stories or simple text giving ample vocabulary to the learner, regarding the theme name '{name}', and should be around 60 tokens (words) long each."
    
    prompt = f"Theme Name: {name}\nNC Level: {nc}\nInstructions: {instructions}\n\nPlease generate the two samples."
    
    try:
        # Call the Gemini Flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Samples,
                temperature=0.7,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating for '{name}': {e}")
        return None

def main():
    file_path = "themes.json"
    
    # Load the existing themes
    with open(file_path, "r", encoding="utf-8") as f:
        themes = json.load(f)
    
    updated = False
    
    for i, theme in enumerate(themes):
        # Skip if already processed
        if "sample_1" in theme and "sample_2" in theme:
            continue
            
        print(f"Processing ({i+1}/{len(themes)}): {theme.get('name')} [NC: {theme.get('nc')}]")
        
        result = generate_samples_for_theme(theme)
        
        if result:
            theme["sample_1"] = result.get("sample_1", "")
            theme["sample_2"] = result.get("sample_2", "")
            updated = True
            
            # Save progressively so we don't lose data if it crashes
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(themes, f, indent=2, ensure_ascii=False)
                
        # Sleep for a moment to respect rate limits
        time.sleep(2)
        
    if updated:
        print("Successfully processed and updated themes.json!")
    else:
        print("No themes needed updating (they already have samples).")

if __name__ == "__main__":
    main()
