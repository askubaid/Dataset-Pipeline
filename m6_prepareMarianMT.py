import os
from transformers import MarianMTModel, MarianTokenizer

def main():
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
    save_directory = "./models/marian-mt"

    print(f"Downloading MarianMT model and tokenizer for '{model_name}'...")
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Download tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    print(f"Saving model and tokenizer to '{save_directory}'...")
    
    # Save them locally
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    print("Download and save complete!")
    print(f"You can now load this model in other scripts using: MarianMTModel.from_pretrained('{save_directory}')")

if __name__ == "__main__":
    main()
