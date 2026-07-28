"""
Generates a labeled dataset for training the RouteLLM-Pro complexity classifier.
Uses Groq to synthesize diverse example queries for each complexity label,
then saves them as JSONL for downstream training.
"""

import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LABELS = ["simple", "medium", "complex"]

PROMPTS = {
    "simple": (
        "Generate 25 diverse example user queries that are SIMPLE: basic facts, "
        "greetings, short definitions, unit conversions, trivia. "
        "One per line, no numbering, no extra text, just the raw queries."
    ),
    "medium": (
        "Generate 25 diverse example user queries that require MEDIUM reasoning: "
        "explanations of how something works, bounded coding tasks (e.g. implement "
        "a standard well-known algorithm), comparisons, multi-step but well-defined "
        "problems. One per line, no numbering, no extra text, just the raw queries."
    ),
    "complex": (
        "Generate 25 diverse example user queries that require COMPLEX, open-ended "
        "reasoning: mathematical proofs, deep theoretical questions, novel or ambiguous "
        "problems, multi-part analysis with no fixed procedure, research-level questions. "
        "One per line, no numbering, no extra text, just the raw queries."
    ),
}

def generate_examples(label):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": PROMPTS[label]}],
        temperature=0.9,
    )
    text = response.choices[0].message.content
    lines = [line.strip("-* \t") for line in text.split("\n") if line.strip()]
    return lines

def main():
    dataset = []
    for label in LABELS:
        print(f"Generating examples for: {label}")
        for i in range(3):
            examples = generate_examples(label)
            for ex in examples:
                if len(ex) > 5:
                    dataset.append({"query": ex, "label": label})
            print(f"  batch {i+1}/3 done, total so far: {len(dataset)}")
            time.sleep(1)

    with open("dataset.jsonl", "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")

    print(f"\nDone. Total examples: {len(dataset)}")
    print("Saved to dataset.jsonl")

if __name__ == "__main__":
    main()
