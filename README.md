# Dataset Pipeline

This is our pipeline for generating multilingual synthetic datasets, AI-powered, which is part of a semester project in the Institute of Space technology Islamabad supervised by Doctor Madihar Tahir. The team comprized of four members. Ubaid ur Rehman, Hafsa Ayub, Iqra Younas and Hajra Shahid. 

## Overview

This part of the project automates the creation of language datasets by:
1. **Parsing themes** from an Excel source file
2. **Generating sample examples** for each theme using AI models
3. **Enriching themes** with granular sub-topics
4. **Creating multilingual datasets** with sentence pairs for Synthetic Dataset Generation.

The pipeline supports both **local inference** (using Ollama) and **cloud API** (using Google Gemini) for maximum flexibility.

## Project Structure

### Core Modules

| File | Purpose |
|------|---------|
| **m1_parse_themes.py** | Parses themes from `RAWThemes.xlsx` and structures them into JSON format |
| **m2_gen_examples_local.py** | Generates sample examples for each theme using a local Ollama model |
| **m3_generate_examples_API.py** | Generates sample examples using Google Gemini API (alternative to local) |
| **m4_granularize.py** | Adds 20 sub-topics to each theme for more granular categorization |
| **m5_gen_dataset.py** | Generates multilingual sentence pairs (English ↔ Portuguese) for Synthetic Dataset Engineering |

### Data Files

| File | Description |
|------|-------------|
| **RAWThemes.xlsx** | Source Excel file containing raw theme definitions organized by categories |
| **themes.json** | Processed themes with samples and sub-topics |
| **learner.en** | Generated English language learning dataset (sentences) |
| **learner.pt** | Generated Portuguese language learning dataset (translations) |

## Features

### Theme Categories
Themes are organized into categories such as:
- Personal & Daily Life
- Travel & Transportation
- Food & Dining
- Work & Professional
- Education & Learning
- Healthcare & Wellness
- Entertainment & Leisure
- And more...

### Complexity Levels
Each theme is assigned a **Narrative Capacity (NC)**:
- **Low**: Simple sentences (~15 words)
- **Medium**: Multi-sentence paragraphs (~30 words)
- **High**: Short stories with rich vocabulary (~60 words)

### AI Inference Engines
Choose one of two inference engines for example generation:

#### 1. Local Inference (Ollama)
- Free and privacy-focused
- Runs on local hardware
- Model: `gemma4:e4b` (configurable)

#### 2. Cloud API (Google Gemini)
- More powerful models
- Better quality outputs
- Requires API key

## Prerequisites

### System Requirements
- Python 3.8+
- pip or conda package manager

### Dependencies
Install required packages:
```bash
pip install pandas openpyxl requests python-dotenv google-genai pydantic
```

## How to Run

### Step 1: Parse Themes
Extract and structure themes from the Excel file:
```bash
python m1_parse_themes.py
```
**Output**: `themes.json` (basic theme structure)

### Step 2: Generate Sample Examples (Choose One)

#### Option A: Using Local Ollama
```bash
python m2_gen_examples_local.py
```

#### Option B: Using Google Gemini API
```bash
python m3_generate_examples_API.py
```
**Output**: `themes.json` (updated with `sample_1` and `sample_2`)

### Step 3: Granularize Themes
Add 20 sub-topics to each theme:
```bash
python m4_granularize.py
```
**Output**: `themes.json` (updated with `topics` array)

**Configuration**:
- Edit `CONFIG` in the script to select inference engine:
  ```python
  "ENGINE": "LOCAL"  # or "API"
  ```

### Step 4: Generate Multilingual Dataset
Create sentence pairs for synthetic dataset:
```bash
python m5_gen_dataset.py
```
**Outputs**:
- `learner.en` - English sentences
- `learner.pt` - Portuguese translations

**Configuration**:
- Edit `CONFIG` in the script:
  ```python
  "STARTING_FROM": 1,      # Start from theme index
  "UPTILL_TO": 130,          # End at theme index
  "SENTENCES_PER_THEME": 1160,  # Adjust based on needs
  "ENGINE": "LOCAL"        # or "API"
  ```

## Complete Pipeline Example

Run the full pipeline sequentially:
```bash
# Step 1: Parse themes
python m1_parse_themes.py

# Step 2: Generate examples (choose one)
python m2_gen_examples_local.py
# OR
# python m3_generate_examples_API.py

# Step 3: Add sub-topics
python m4_granularize.py

# Step 4: Create dataset
python m5_gen_dataset.py
```

## Configuration Guide

Most modules have a `CONFIG` dictionary at the top. Key settings:

### Inference Engine Selection
```python
CONFIG = {
    "ENGINE": "LOCAL",  # "LOCAL" (Ollama) or "API" (Gemini)
}
```

### Ollama Settings
```python
CONFIG = {
    "LOCAL_MODEL": "gemma4:e4b",
    "OLLAMA_URL": "http://localhost:11434/api/generate",
    "SLEEP_SECONDS": 1,  # Rate limiting
}
```

### Gemini API Settings
```python
CONFIG = {
    "GEMINI_MODEL": "gemini-2.5-flash",
    # Requires GEMINI_API_KEY in .env file
}
```

### Dataset Generation
```python
CONFIG = {
    "STARTING_FROM": 1,
    "UPTILL_TO": 130,
    "SENTENCES_PER_THEME": 1160,
    "API_BATCH_SIZE": 30,
    "LOCAL_BATCH_SIZE": 20,
}
```

## Output Examples

### themes.json Structure
```json
{
  "id": 1,
  "category": "Personal & Daily Life",
  "name": "Basic Greetings & Introductions",
  "nc": "Low",
  "tokens": 17700,
  "sp": 1250,
  "sample_1": "Hello, my name is John and it is very nice to finally meet you today.",
  "sample_2": "Good morning, how are you doing today? Let me introduce myself to the whole team.",
  "topics": [
    "Saying hello politely",
    "Asking how someone is",
    "Responding to 'goodbye'",
    ...
  ]
}
```

### learner.en / learner.pt Structure
```
One sentence or paragraph per line
English sentences paired with Portuguese translations
Used as a synthetic Dataset for model training.
```

## Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check URL is correct: `http://localhost:11434`

### Gemini API Errors
- Verify API key is set in `.env`
- Check API key has Generative AI permissions
- Ensure internet connection is active

### Memory Issues
- Reduce `SENTENCES_PER_THEME` in config
- Use smaller batch sizes
- Process themes in smaller ranges

### Model Not Found (Ollama)
```bash
ollama pull gemma4:e4b
```


## License

This project is open-source. A gift from Institute of space Technology Islamabad. Feel free to use, modify, and distribute.

## Contact & Support

For issues, questions, or contributions, please open an issue in the repository.

---

**Created**: May 2026  
**Status**: Active Development
