# ▶️ Run this in the same Colab to shuffle the records in-place.

import json, random

SRC = "wikitext_chunks.jsonl"          # change if your file is named differently
DEST = "answers_shuffled.jsonl"  # output file

# 1. Load all JSONL records
with open(SRC, "r") as f:
    records = [json.loads(line) for line in f]

# 2. Shuffle their order (labels/inputs stay intact)
random.shuffle(records)        # add random.seed(42) above for reproducibility

# 3. Save to a new JSONL
with open(DEST, "w") as f:
    for obj in records:
        json.dump(obj, f)
        f.write("\n")

print(f"{len(records)} records shuffled → {DEST}")

