from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained("./classifier_model/best")
model = DistilBertForSequenceClassification.from_pretrained("./classifier_model/best")
model.eval()

ID2LABEL = {0: "simple", 1: "medium", 2: "complex"}

test_queries = [
    "What is the capital of Japan?",
    "Write a Python implementation of merge sort with time complexity analysis",
    "Explain the mathematical proof behind why P vs NP is unsolved",
    "How many days are in a week?",
    "Describe how a hash table handles collisions",
    "What are the philosophical implications of Godel's incompleteness theorems for the limits of formal systems?",
    "What time is it in Tokyo right now?",
    "Implement a binary search tree with insert and delete operations",
    "Derive the Navier-Stokes equations from first principles and discuss their limitations",
]

for q in test_queries:
    inputs = tokenizer(q, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    print(f"{ID2LABEL[pred]:10s} -> {q}")
