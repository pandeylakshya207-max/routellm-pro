import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from router.classifier import classify_query as prompt_classify

TEST_SET = [
    ("What is the capital of France?", "simple"),
    ("How many continents are there?", "simple"),
    ("What year did World War 2 end?", "simple"),
    ("Say hello to me", "simple"),
    ("What is the boiling point of water in Celsius?", "simple"),
    ("Convert 10 miles to kilometers", "simple"),
    ("Who wrote Romeo and Juliet?", "simple"),
    ("What is the chemical symbol for gold?", "simple"),
    ("Explain the difference between TCP and UDP", "medium"),
    ("How does a hash table handle collisions?", "medium"),
    ("Write a function to check if a string is a palindrome", "medium"),
    ("What causes seasons to change on Earth?", "medium"),
    ("Explain how photosynthesis works", "medium"),
    ("Compare REST and GraphQL APIs", "medium"),
    ("How does garbage collection work in Java?", "medium"),
    ("Write a Python function to reverse a linked list", "medium"),
    ("Explain the mathematical proof behind why P vs NP is unsolved", "complex"),
    ("What are the philosophical implications of the simulation hypothesis?", "complex"),
    ("Derive the equations of general relativity from the equivalence principle", "complex"),
    ("Design a distributed consensus algorithm resilient to Byzantine faults", "complex"),
    ("What are the fundamental limits of computability according to the halting problem?", "complex"),
    ("Analyze the economic tradeoffs of universal basic income at a macro scale", "complex"),
    ("Prove that there are infinitely many prime numbers", "complex"),
    ("What ethical framework should guide autonomous weapons decision-making?", "complex"),
]

queries = [q for q, _ in TEST_SET]
true_labels = [l for _, l in TEST_SET]
ID2LABEL = {0: "simple", 1: "medium", 2: "complex"}

def load_model(path):
    tok = DistilBertTokenizerFast.from_pretrained(path)
    mdl = DistilBertForSequenceClassification.from_pretrained(path)
    mdl.eval()
    return tok, mdl

def classify_with_model(tok, mdl, query):
    inputs = tok(query, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    return ID2LABEL[torch.argmax(logits, dim=1).item()]

tok_v1, mdl_v1 = load_model("./classifier_model/best")
tok_v2, mdl_v2 = load_model("./classifier_model_v2/best")

print("Running prompt-based classifier...")
prompt_preds = [prompt_classify(q) for q in queries]

print("Running trained v1...")
v1_preds = [classify_with_model(tok_v1, mdl_v1, q) for q in queries]

print("Running trained v2...")
v2_preds = [classify_with_model(tok_v2, mdl_v2, q) for q in queries]

print("\n" + "=" * 110)
print(f"{'QUERY':<50} {'TRUE':<9} {'PROMPT':<9} {'V1':<9} {'V2':<9}")
print("=" * 110)
for q, t, p, v1, v2 in zip(queries, true_labels, prompt_preds, v1_preds, v2_preds):
    print(f"{q[:48]:<50} {t:<9} {p:<9} {v1:<9} {v2:<9}")

print("\n" + "=" * 110)
print("SUMMARY")
print("=" * 110)
for name, preds in [("Prompt-based", prompt_preds), ("Trained v1", v1_preds), ("Trained v2", v2_preds)]:
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="macro")
    print(f"{name:<15} accuracy={acc:.3f}  f1_macro={f1:.3f}")

print("\nTrained v2 classification report:")
print(classification_report(true_labels, v2_preds, zero_division=0))

results = {
    "n_test_queries": len(queries),
    "prompt_based": {"accuracy": accuracy_score(true_labels, prompt_preds), "f1_macro": f1_score(true_labels, prompt_preds, average="macro")},
    "trained_v1": {"accuracy": accuracy_score(true_labels, v1_preds), "f1_macro": f1_score(true_labels, v1_preds, average="macro")},
    "trained_v2": {"accuracy": accuracy_score(true_labels, v2_preds), "f1_macro": f1_score(true_labels, v2_preds, average="macro")},
}
with open("benchmark_results_v2.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to benchmark_results_v2.json")
