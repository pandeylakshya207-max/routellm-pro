import json
import random
from sklearn.model_selection import train_test_split

random.seed(42)
data = [json.loads(l) for l in open("dataset.jsonl", encoding="utf-8")]

queries = [d["query"] for d in data]
labels = [d["label"] for d in data]

train_q, val_q, train_l, val_l = train_test_split(
    queries, labels, test_size=0.2, random_state=42, stratify=labels
)

with open("train.jsonl", "w", encoding="utf-8") as f:
    for q, l in zip(train_q, train_l):
        f.write(json.dumps({"query": q, "label": l}) + "\n")

with open("val.jsonl", "w", encoding="utf-8") as f:
    for q, l in zip(val_q, val_l):
        f.write(json.dumps({"query": q, "label": l}) + "\n")

print("Train size:", len(train_q))
print("Val size:", len(val_q))
