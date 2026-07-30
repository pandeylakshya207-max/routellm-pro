import pytest
from unittest.mock import patch, MagicMock

def make_result(text, tier, model, provider):
    return {"text": text, "tier": tier, "model": model, "provider": provider}

@patch("router.policy.redis_client")
@patch("router.classifier.classify_query_a")
@patch("router.providers.call_mid")
@patch("router.policy.is_tier_degraded")
def test_skips_degraded_tier(mock_degraded, mock_mid, mock_classify, mock_redis):
    mock_classify.return_value = "medium"
    # Mark "mid" as degraded so it should be skipped, falling through to "small"
    mock_degraded.side_effect = lambda tier: tier == "mid"
    with patch("router.providers.call_small") as mock_small:
        mock_small.return_value = make_result("answer", "small", "llama-3.1-8b-instant", "groq")
        from router.policy import route_query
        result = route_query("some query", ab_variant="A")
        assert result["tier"] == "small"
        assert "mid" in result["skipped_degraded_tiers"]
        assert result["fallback_occurred"] == True
        mock_mid.assert_not_called()

@patch("router.policy.redis_client")
@patch("router.classifier.classify_query_a")
@patch("router.providers.call_mid")
@patch("router.policy.is_tier_degraded")
def test_does_not_skip_healthy_tier(mock_degraded, mock_mid, mock_classify, mock_redis):
    mock_classify.return_value = "medium"
    mock_degraded.return_value = False  # nothing is degraded
    mock_mid.return_value = make_result("answer", "mid", "llama-3.3-70b-versatile", "groq")
    from router.policy import route_query
    result = route_query("some query", ab_variant="A")
    assert result["tier"] == "mid"
    assert result["skipped_degraded_tiers"] == []
    assert result["fallback_occurred"] == False

@patch("router.policy.redis_client")
def test_get_rolling_avg_latency_returns_baseline_when_no_data(mock_redis):
    mock_redis.lrange.return_value = []
    from router.policy import get_rolling_avg_latency, BASELINE_LATENCY
    avg = get_rolling_avg_latency("small")
    assert avg == BASELINE_LATENCY["small"]

@patch("router.policy.redis_client")
def test_get_rolling_avg_latency_computes_correctly(mock_redis):
    mock_redis.lrange.return_value = ["1.0", "2.0", "3.0"]
    from router.policy import get_rolling_avg_latency
    avg = get_rolling_avg_latency("small")
    assert avg == 2.0

@patch("router.policy.redis_client")
def test_is_tier_degraded_detects_slow_tier(mock_redis):
    mock_redis.lrange.return_value = ["20.0"] * 5  # way above any baseline
    from router.policy import is_tier_degraded
    assert is_tier_degraded("small") == True

@patch("router.policy.redis_client")
def test_is_tier_degraded_healthy_tier(mock_redis):
    mock_redis.lrange.return_value = ["0.9", "1.1", "1.0"]  # near baseline for small
    from router.policy import is_tier_degraded
    assert is_tier_degraded("small") == False
