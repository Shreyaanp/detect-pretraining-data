import json
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load the dataset (train split for demo; change to 'validation' or 'test' if needed)
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

# Initialize tokenizer for chunking (use GPT2 for consistency)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Output file
output_file = "wikitext_chunks.jsonl"

with open(output_file, "w") as f:
    for article in dataset["text"]:
        if not article.strip():  # Skip empty lines
            continue
        # Tokenize and chunk into ~64 token pieces
        tokens = tokenizer.encode(article)
        for i in range(0, len(tokens), 64):
            chunk_tokens = tokens[i:i+64]
            if len(chunk_tokens) < 10:  # Skip very short chunks
                continue
            chunk_text = tokenizer.decode(chunk_tokens)
            # Add dummy label (0 = assume unseen; mix with 1 for some if you know they are seen)
            data = {"input": chunk_text, "label": 0}
            f.write(json.dumps(data) + "\n")

print(f"Saved chunks to {output_file}")
