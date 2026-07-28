import json
import random
from collections import Counter

random.seed(42)

# Load everything we have
original = [json.loads(l) for l in open("dataset.jsonl", encoding="utf-8")]
diverse_complex = [json.loads(l) for l in open("dataset_complex_diverse.jsonl", encoding="utf-8")]

# Separate original by class
simple = [d for d in original if d["label"] == "simple"]
medium = [d for d in original if d["label"] == "medium"]
orig_complex = [d for d in original if d["label"] == "complex"]

# Combine original complex + diverse complex, then sample down to match simple/medium count
all_complex = orig_complex + diverse_complex
random.shuffle(all_complex)

target_size = max(len(simple), len(medium))  # balance to the larger of simple/medium
complex_balanced = all_complex[:target_size]

merged = simple + medium + complex_balanced
random.shuffle(merged)

counts = Counter(d["label"] for d in merged)
print("Balanced dataset class counts:", dict(counts))
print("Total examples:", len(merged))

with open("dataset_v3.jsonl", "w", encoding="utf-8") as f:
    for d in merged:
        f.write(json.dumps(d) + "\n")

print("Saved to dataset_v3.jsonl")
