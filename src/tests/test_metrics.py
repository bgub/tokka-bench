"""
Tests for metrics calculation functions.

This demonstrates how to test the word metrics and global tracking functionality.
"""

import pytest
from unittest.mock import Mock
from tokka_bench.metrics import (
    calculate_word_metrics,
    GlobalMetricsTracker,
)


def test_calculate_word_metrics():
    """Test word metrics calculation with a mock tokenizer."""
    # Create a mock tokenizer
    mock_tokenizer = Mock()

    # Mock the encode method to return predictable results
    def mock_encode(text):
        if text == "hello world test":
            return [1, 2, 3]  # 3 tokens for full text
        elif text == "hello":
            return [1]  # 1 token (not split)
        elif text == "world":
            return [2]  # 1 token (not split)
        elif text == "test":
            return [3]  # 1 token (not split)
        return []

    mock_tokenizer.encode = mock_encode

    # Mock the decode method
    def mock_decode(token_ids, skip_special_tokens=True):
        token_map = {1: "hello", 2: "world", 3: "test"}
        return token_map.get(token_ids[0], "")

    mock_tokenizer.decode = mock_decode

    # Test with simple text
    text = "hello world test"
    result = calculate_word_metrics(mock_tokenizer, text)

    # Verify results
    assert result["subword_fertility"] == 1.0  # 3 tokens / 3 words
    assert result["continued_word_rate"] == 0.0  # No words split
    assert result["debug_info"]["total_words"] == 3
    assert result["debug_info"]["words_split"] == 0


def test_calculate_word_metrics_with_splits():
    """Test word metrics calculation with words that get split."""
    # Create a mock tokenizer that splits some words
    mock_tokenizer = Mock()

    def mock_encode(text):
        if text == "hello world":
            return [1, 2, 3]  # 3 tokens for full text
        elif text == "hello":
            return [1, 2]  # 2 tokens (split)
        elif text == "world":
            return [3]  # 1 token (not split)
        return []

    mock_tokenizer.encode = mock_encode

    def mock_decode(token_ids, skip_special_tokens=True):
        token_map = {1: "hel", 2: "lo", 3: "world"}
        return token_map.get(token_ids[0], "")

    mock_tokenizer.decode = mock_decode

    # Test with text that has splits
    text = "hello world"
    result = calculate_word_metrics(mock_tokenizer, text)

    # Verify results
    assert result["subword_fertility"] == 1.5  # 3 tokens / 2 words
    assert result["continued_word_rate"] == 50.0  # 1 out of 2 words split
    assert result["debug_info"]["total_words"] == 2
    assert result["debug_info"]["words_split"] == 1


def test_calculate_word_metrics_empty_text():
    """Test word metrics calculation with empty text."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(return_value=[])

    result = calculate_word_metrics(mock_tokenizer, "")

    assert result["subword_fertility"] == 0.0
    assert result["continued_word_rate"] == 0.0
    assert result["debug_info"]["total_words"] == 0


def test_global_metrics_tracker():
    """Test global metrics tracking functionality."""
    tracker = GlobalMetricsTracker()

    # Test initial state
    assert tracker.total_token_count == 0
    assert tracker.get_global_metrics() == {}

    # Add some token information
    tokens_info = [
        {
            "id": 1,
            "text": " hello",
            "starts_with_space": True,
            "has_whitespace_in_middle": False,
            "scripts": ["Latin"],
            "script_overlap": False,
        },
        {
            "id": 2,
            "text": "world",
            "starts_with_space": False,
            "has_whitespace_in_middle": False,
            "scripts": ["Latin"],
            "script_overlap": False,
        },
        {
            "id": 3,
            "text": "!",
            "starts_with_space": False,
            "has_whitespace_in_middle": False,
            "scripts": ["Punctuation"],
            "script_overlap": False,
        },
    ]

    tracker.add_tokens(tokens_info)

    # Check metrics
    metrics = tracker.get_global_metrics()

    assert metrics["total_tokens_analyzed"] == 3
    assert metrics["tokens_starting_with_space_pct"] == pytest.approx(
        100.0 / 3
    )  # 1 out of 3
    assert metrics["tokens_with_whitespace_in_middle_pct"] == 0.0  # None
    assert metrics["tokens_with_script_overlap_pct"] == 0.0  # None
    assert metrics["tokens_with_latin_unicode_pct"] == pytest.approx(
        200.0 / 3
    )  # 2 out of 3
    assert metrics["tokens_with_punctuation_unicode_pct"] == pytest.approx(
        100.0 / 3
    )  # 1 out of 3


def test_global_metrics_tracker_with_overlap():
    """Test global metrics tracking with script overlap."""
    tracker = GlobalMetricsTracker()

    # Add token with script overlap
    tokens_info = [
        {
            "id": 1,
            "text": "hello世界",
            "starts_with_space": False,
            "has_whitespace_in_middle": False,
            "scripts": ["Latin", "Chinese"],
            "script_overlap": True,
        },
    ]

    tracker.add_tokens(tokens_info)
    metrics = tracker.get_global_metrics()

    assert metrics["tokens_with_script_overlap_pct"] == 100.0  # 1 out of 1
    assert metrics["tokens_with_latin_unicode_pct"] == 100.0
    assert metrics["tokens_with_chinese_unicode_pct"] == 100.0
