import os
import sys
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

def main():
    input_file = "learner_single___.en"
    output_file = "learner_single___.pt"
    model_dir = "./models/marian-mt"
    batch_size = 32

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Loading model and tokenizer from '{model_dir}'...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_dir)
        model = MarianMTModel.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model is downloaded at the specified path.")
        sys.exit(1)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    print(f"Reading sentences from '{input_file}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    total_sentences = len(sentences)
    print(f"Total sentences to translate: {total_sentences}")

    print(f"Starting translation (Batch Size: {batch_size})...")
    translated_sentences = []

    # Process in batches
    for i in tqdm(range(0, total_sentences, batch_size), desc="Translating", unit="batch"):
        batch = sentences[i : i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move inputs to device (GPU/CPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translations without computing gradients
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
            
        # Decode the tokens back into strings
        decoded_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translated_sentences.extend(decoded_batch)

    print(f"Saving {len(translated_sentences)} translated sentences to '{output_file}'...")
    with open(output_file, "w", encoding="utf-8") as f:
        for pt_sentence in translated_sentences:
            f.write(pt_sentence + "\n")

    print("Translation complete!")

if __name__ == "__main__":
    main()
