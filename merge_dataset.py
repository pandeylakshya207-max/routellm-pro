import json
import random

random.seed(42)

# Load original dataset
original = [json.loads(l) for l in open("dataset.jsonl", encoding="utf-8")]

# Load new diverse complex examples, cap at 80
diverse_complex = [json.loads(l) for l in open("dataset_complex_diverse.jsonl", encoding="utf-8")]
random.shuffle(diverse_complex)
diverse_complex = diverse_complex[:80]

merged = original + diverse_complex

# Report new class balance
from collections import Counter
counts = Counter(d["label"] for d in merged)
print("Merged dataset class balance:", dict(counts))
print("Total examples:", len(merged))

with open("dataset_v2.jsonl", "w", encoding="utf-8") as f:
    for d in merged:
        f.write(json.dumps(d) + "\n")

print("Saved to dataset_v2.jsonl")
