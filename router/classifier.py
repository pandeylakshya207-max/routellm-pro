from router.providers import groq_client

def classify_query_a(query: str) -> str:
    """
    Policy A (control): current production classifier.
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
    return _call_classifier(prompt)


def classify_query_b(query: str) -> str:
    """
    Policy B (variant): more cost-aggressive classifier.
    Relaxes the bar for "simple" to route more borderline queries to
    the cheap tier, testing whether cost can drop further without a
    meaningful quality hit.
    Returns one of: "simple", "medium", "complex"
    """
    prompt = (
        "Classify the following query as exactly one word: "
        "simple, medium, or complex. "
        "simple = basic facts, greetings, short definitions, AND any question "
        "with a well-known, commonly-taught answer even if it requires one step "
        "of explanation (e.g. how photosynthesis works, why seasons change). "
        "medium = requires genuine multi-step reasoning or non-trivial coding. "
        "complex = requires deep reasoning, math proofs, or open-ended research-level analysis.\n\n"
        f"Query: {query}\n\n"
        "Answer with exactly one word: simple, medium, or complex."
    )
    return _call_classifier(prompt)


def _call_classifier(prompt: str) -> str:
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


# Keep the original name as an alias to Policy A, so nothing else breaks
def classify_query(query: str) -> str:
    return classify_query_a(query)
