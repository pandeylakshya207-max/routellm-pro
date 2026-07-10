from router.providers import groq_client

def classify_query(query: str) -> str:
    """
    Classifies a query as simple, medium, or complex.
    Uses a fast, cheap model call for the classification itself.
    Returns one of: "simple", "medium", "complex"
    """
    prompt = (
        "Classify the following query as exactly one word: "
        "simple, medium, or complex. "
        "simple = basic facts, greetings, short definitions. "
        "medium = requires some reasoning, explanation, or multi-step thought. "
        "complex = requires deep reasoning, math, code, or multi-part analysis.\n\n"
        f"Query: {query}\n\n"
        "Answer with exactly one word: simple, medium, or complex."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        label = response.choices[0].message.content.strip().lower()

        if "simple" in label:
            return "simple"
        elif "complex" in label:
            return "complex"
        else:
            return "medium"

    except Exception:
        return "medium"
