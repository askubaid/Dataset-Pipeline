import sys
from transformers import MarianMTModel, MarianTokenizer

def main():
    model_dir = "./models/marian-mt"
    
    print(f"Loading model and tokenizer from '{model_dir}'...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_dir)
        model = MarianMTModel.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have run m6_prepareMarianMT.py first to download the model.")
        sys.exit(1)

    print("\nModel loaded successfully! Starting inference loop.")
    print("-" * 50)
    
    while True:
        try:
            english_sentence = input("\nEnter English sentence (or type 'quit' to exit): ").strip()
            
            if english_sentence.lower() in ['quit', 'q', 'exit']:
                print("Exiting inference loop. Goodbye!")
                break
                
            if not english_sentence:
                continue

            # Tokenize the input text
            inputs = tokenizer(english_sentence, return_tensors="pt", padding=True)
            
            # Generate translation
            translated_tokens = model.generate(**inputs)
            
            # Decode the generated tokens
            portuguese_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            
            print(f"Portuguese: {portuguese_sentence}")
            
        except KeyboardInterrupt:
            print("\nExiting inference loop. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred during translation: {e}")

if __name__ == "__main__":
    main()
 