import pytest
from unittest.mock import patch, MagicMock

def make_result(text, tier, model, provider):
    return {"text": text, "tier": tier, "model": model, "provider": provider}

@patch("router.policy.redis_client")
@patch("router.policy.classify_query")
@patch("router.providers.call_small")
def test_simple_query_routes_to_small(mock_small, mock_classify, mock_redis):
    mock_classify.return_value = "simple"
    mock_small.return_value = make_result("Paris", "small", "llama-3.1-8b-instant", "groq")
    from router.policy import route_query
    result = route_query("What is the capital of France?")
    assert result["tier"] == "small"
    assert result["fallback_occurred"] == False
    assert result["attempted_tiers"] == ["small"]

@patch("router.policy.redis_client")
@patch("router.policy.classify_query")
@patch("router.providers.call_frontier")
@patch("router.providers.call_mid")
def test_frontier_failure_falls_back_to_mid(mock_mid, mock_frontier, mock_classify, mock_redis):
    mock_classify.return_value = "complex"
    mock_frontier.side_effect = Exception("Simulated outage")
    mock_mid.return_value = make_result("Proof text", "mid", "llama-3.3-70b-versatile", "groq")
    from router.policy import route_query
    result = route_query("Prove something complex")
    assert result["tier"] == "mid"
    assert result["fallback_occurred"] == True
    assert result["attempted_tiers"] == ["frontier", "mid"]

@patch("router.policy.redis_client")
@patch("router.policy.classify_query")
@patch("router.providers.call_frontier")
@patch("router.providers.call_mid")
@patch("router.providers.call_small")
def test_all_tiers_fail_returns_error_result(mock_small, mock_mid, mock_frontier, mock_classify, mock_redis):
    mock_classify.return_value = "complex"
    mock_frontier.side_effect = Exception("Frontier down")
    mock_mid.side_effect = Exception("Mid down")
    mock_small.side_effect = Exception("Small down")
    from router.policy import route_query
    result = route_query("Some complex query")
    assert result["tier"] == "none"
    assert result["error"] == "all_tiers_failed"
    assert result["attempted_tiers"] == ["frontier", "mid", "small"]

@patch("router.policy.redis_client")
@patch("router.policy.classify_query")
@patch("router.providers.call_mid")
def test_medium_query_cost_weight(mock_mid, mock_classify, mock_redis):
    mock_classify.return_value = "medium"
    mock_mid.return_value = make_result("answer", "mid", "llama-3.3-70b-versatile", "groq")
    from router.policy import route_query
    result = route_query("Explain something")
    assert result["cost_weight"] == 3

@patch("router.policy.redis_client")
@patch("router.policy.classify_query")
@patch("router.providers.call_small")
def test_logs_to_redis(mock_small, mock_classify, mock_redis):
    mock_classify.return_value = "simple"
    mock_small.return_value = make_result("answer", "small", "llama-3.1-8b-instant", "groq")
    from router.policy import route_query
    route_query("test query")
    mock_redis.lpush.assert_called_once()
