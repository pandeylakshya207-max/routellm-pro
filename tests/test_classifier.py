import pytest
from unittest.mock import patch, MagicMock

def make_mock_response(content):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response

@patch("router.classifier.groq_client")
def test_classify_simple(mock_client):
    mock_client.chat.completions.create.return_value = make_mock_response("simple")
    from router.classifier import classify_query
    result = classify_query("What is the capital of France?")
    assert result == "simple"

@patch("router.classifier.groq_client")
def test_classify_medium(mock_client):
    mock_client.chat.completions.create.return_value = make_mock_response("medium")
    from router.classifier import classify_query
    result = classify_query("Explain how hash tables work")
    assert result == "medium"

@patch("router.classifier.groq_client")
def test_classify_complex(mock_client):
    mock_client.chat.completions.create.return_value = make_mock_response("complex")
    from router.classifier import classify_query
    result = classify_query("Prove the Riemann hypothesis")
    assert result == "complex"

@patch("router.classifier.groq_client")
def test_classify_handles_malformed_response(mock_client):
    """If the model returns something unexpected, should default to medium."""
    mock_client.chat.completions.create.return_value = make_mock_response("banana")
    from router.classifier import classify_query
    result = classify_query("some query")
    assert result == "medium"

@patch("router.classifier.groq_client")
def test_classify_handles_api_exception(mock_client):
    """If the API call itself fails, should default to medium, not crash."""
    mock_client.chat.completions.create.side_effect = Exception("API down")
    from router.classifier import classify_query
    result = classify_query("some query")
    assert result == "medium"
