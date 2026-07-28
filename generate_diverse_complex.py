import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Explicitly target the failure mode: complex queries phrased as direct
# requests/commands, not just declarative "Derive/Develop" research phrasing
PROMPT_COMPLEX_DIVERSE = (
    "Generate 40 diverse example user queries requiring COMPLEX, open-ended reasoning "
    "(math proofs, deep theoretical questions, novel/ambiguous problems, research-level "
    "analysis, no fixed procedure to solve them). "
    "IMPORTANT: vary the phrasing style significantly. Include a mix of: "
    "- Direct commands: 'Prove that...', 'Design a...', 'Build a...', 'Solve...'\n"
    "- Questions: 'What is...', 'How does...', 'Why is...'\n"
    "- Declarative requests: 'Derive...', 'Develop...', 'Analyze...'\n"
    "- Casual phrasing: 'Can you figure out...', 'I need help understanding...'\n"
    "Examples of the direct-command style to include more of: "
    "'Prove that there are infinitely many prime numbers', "
    "'Design a distributed consensus algorithm resilient to Byzantine faults'. "
    "One query per line, no numbering, no extra text."
)

def generate_batch():
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": PROMPT_COMPLEX_DIVERSE}],
        temperature=1.0,
    )
    text = response.choices[0].message.content
    lines = [line.strip("-* \t") for line in text.split("\n") if line.strip()]
    return lines

def main():
    dataset = []
    print("Generating phrasing-diverse complex examples...")
    for i in range(3):
        examples = generate_batch()
        for ex in examples:
            if len(ex) > 5:
                dataset.append({"query": ex, "label": "complex"})
        print(f"  batch {i+1}/3 done, total so far: {len(dataset)}")
        time.sleep(1)

    with open("dataset_complex_diverse.jsonl", "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")

    print(f"\nDone. Total new complex examples: {len(dataset)}")

if __name__ == "__main__":
    main()
