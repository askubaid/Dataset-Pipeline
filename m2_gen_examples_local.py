import json
import time
import requests

# Default Ollama API endpoint for generating text
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e4b"

def generate_samples_for_theme(theme):
    nc = theme.get("nc", "").lower()
    name = theme.get("name", "")
    
    # Determine the prompt logic based on NC level
    if nc == "low":
        instructions = f"Create two sample English sentences, sample_1 and sample_2. These sentences should be regarding the theme name '{name}' and should be around 15 words long each."
    elif nc == "medium":
        instructions = f"Create two samples, sample_1 and sample_2. These samples should be multi-sentence, regarding the theme name '{name}', and should be up to 30 words long each."
    else: # high or default
        instructions = f"Create two samples, sample_1 and sample_2. These samples should be short stories or simple text giving ample vocabulary to the learner, regarding the theme name '{name}', and should be around 60 words long each."
    
    prompt = (
        f"Theme Name: {name}\n"
        f"NC Level: {nc}\n"
        f"Instructions: {instructions}\n\n"
        f"You must respond ONLY with a valid JSON object containing exactly two keys: 'sample_1' and 'sample_2', both containing your generated strings."
    )
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",  # Forces Ollama to output valid JSON
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
    
    try:
        # Call the local Ollama API
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        
        # Check if Ollama returned an error (like 400 Bad Request)
        if response.status_code != 200:
            print(f"API Error {response.status_code} for '{name}': {response.text}")
            return None
            
        # Extract the text response and parse the JSON
        result_text = response.json().get("response", "")
        return json.loads(result_text)
        
    except requests.exceptions.RequestException as e:
        print(f"Connection Error for '{name}': Make sure Ollama is running! Details: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for '{name}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for '{name}': {e}")
        return None

def main():
    file_path = "themes.json"
    
    # Load the existing themes
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            themes = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return
    
    updated = False
    
    for i, theme in enumerate(themes):
        # Skip if already processed
        if "sample_1" in theme and "sample_2" in theme:
            continue
            
        print(f"Processing ({i+1}/{len(themes)}): {theme.get('name')} [NC: {theme.get('nc')}]")
        
        start_time = time.time()
        result = generate_samples_for_theme(theme)
        elapsed = time.time() - start_time
        
        # Validate that we got a dictionary with the keys we need
        if result and isinstance(result, dict) and "sample_1" in result and "sample_2" in result:
            theme["sample_1"] = result.get("sample_1", "")
            theme["sample_2"] = result.get("sample_2", "")
            updated = True
            
            # Save progressively
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(themes, f, indent=2, ensure_ascii=False)
                
            print(f"  -> Success! (took {elapsed:.1f}s)")
        else:
            print(f"  -> Failed to generate valid data. Model output: {result}")
            
    if updated:
        print("Successfully processed and updated themes.json using local Ollama model!")
    else:
        print("No themes needed updating (they already have samples).")

if __name__ == "__main__":
    main()
