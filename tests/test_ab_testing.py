import pytest
from unittest.mock import patch, MagicMock

def make_result(text, tier, model, provider):
    return {"text": text, "tier": tier, "model": model, "provider": provider}

@patch("router.policy.redis_client")
@patch("router.classifier.classify_query_a")
@patch("router.providers.call_mid")
def test_explicit_variant_a_used(mock_mid, mock_classify_a, mock_redis):
    mock_classify_a.return_value = "medium"
    mock_mid.return_value = make_result("answer", "mid", "llama-3.3-70b-versatile", "groq")
    from router.policy import route_query
    result = route_query("some query", ab_variant="A")
    assert result["ab_variant"] == "A"
    mock_classify_a.assert_called_once()

@patch("router.policy.redis_client")
@patch("router.classifier.classify_query_b")
@patch("router.providers.call_small")
def test_explicit_variant_b_used(mock_small, mock_classify_b, mock_redis):
    mock_classify_b.return_value = "simple"
    mock_small.return_value = make_result("answer", "small", "llama-3.1-8b-instant", "groq")
    from router.policy import route_query
    result = route_query("some query", ab_variant="B")
    assert result["ab_variant"] == "B"
    mock_classify_b.assert_called_once()

@patch("router.policy.redis_client")
@patch("router.classifier.classify_query_a")
@patch("router.classifier.classify_query_b")
@patch("router.providers.call_small")
def test_random_assignment_picks_a_or_b(mock_small, mock_classify_b, mock_classify_a, mock_redis):
    mock_classify_a.return_value = "simple"
    mock_classify_b.return_value = "simple"
    mock_small.return_value = make_result("answer", "small", "llama-3.1-8b-instant", "groq")
    from router.policy import route_query
    variants_seen = set()
    for _ in range(20):
        result = route_query("some query")
        variants_seen.add(result["ab_variant"])
    assert variants_seen == {"A", "B"}
