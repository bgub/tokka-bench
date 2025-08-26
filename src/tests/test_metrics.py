"""
Tests for metrics calculation functions.
"""

import pytest
from unittest.mock import Mock
from tokka_bench.metrics import (
    calculate_word_metrics,
    GlobalMetricsTracker,
    is_character_based_language,
)


# ===== FIXTURES =====


@pytest.fixture
def simple_tokenizer():
    """Mock tokenizer that doesn't split any words."""
    mock = Mock()

    # Simple word-to-token mapping (no splits)
    word_tokens = {"hello": [1], "world": [2], "test": [3], "foo": [4], "bar": [5]}

    def encode(text):
        words = text.split()
        tokens = []
        for word in words:
            tokens.extend(word_tokens.get(word, [999]))  # Unknown word = 999
        return tokens

    def decode(token_ids, skip_special_tokens=True):
        token_map = {1: "hello", 2: "world", 3: "test", 4: "foo", 5: "bar", 999: "UNK"}
        return token_map.get(token_ids[0], "UNK")

    mock.encode = encode
    mock.decode = decode
    return mock


@pytest.fixture
def splitting_tokenizer():
    """Mock tokenizer that splits some words."""
    mock = Mock()

    def encode(text):
        # Simulate subword tokenization
        if text == "hello world":
            return [1, 2, 3]  # "hello" split, "world" not split
        elif text == "hello":
            return [1, 2]  # Split into 2 tokens
        elif text == "world":
            return [3]  # Not split
        elif text == "testing":
            return [4, 5, 6]  # Split into 3 tokens
        elif not text.strip():  # Handle empty/whitespace text
            return []
        else:
            # For unknown text, return one token per word (no splits)
            words = text.split()
            return list(range(100, 100 + len(words)))

    def decode(token_ids, skip_special_tokens=True):
        token_map = {1: "hel", 2: "lo", 3: "world", 4: "test", 5: "ing", 6: ""}
        if token_ids and len(token_ids) == 1:
            return token_map.get(token_ids[0], "UNK")
        return "UNK"

    mock.encode = encode
    mock.decode = decode
    return mock


@pytest.fixture
def chinese_tokenizer():
    """Mock tokenizer for Chinese text."""
    mock = Mock()

    def encode(text):
        # For Chinese, each character might be 1-2 tokens
        tokens = []
        for char in text:
            if char == "你":
                tokens.extend([1001, 1002])  # Split character
            elif char == "好":
                tokens.append(1003)  # Single token
            elif char == "世":
                tokens.extend([1004, 1005])  # Split character
            elif char == "界":
                tokens.append(1006)  # Single token
            elif char.isspace():
                continue  # Skip whitespace
            else:
                tokens.append(9999)  # Unknown character
        return tokens

    def decode(token_ids, skip_special_tokens=True):
        token_map = {
            1001: "你1",
            1002: "你2",
            1003: "好",
            1004: "世1",
            1005: "世2",
            1006: "界",
            9999: "UNK",
        }
        if token_ids and len(token_ids) == 1:
            return token_map.get(token_ids[0], "UNK")
        return "UNK"

    mock.encode = encode
    mock.decode = decode
    return mock


@pytest.fixture
def sample_tokens():
    """Sample token data for GlobalMetricsTracker tests."""
    return [
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


@pytest.fixture
def mixed_script_tokens():
    """Tokens with multiple scripts for testing script overlap."""
    return [
        {
            "id": 1,
            "text": "hello世界",
            "starts_with_space": False,
            "has_whitespace_in_middle": False,
            "scripts": ["Latin", "Chinese"],
            "script_overlap": True,
        },
        {
            "id": 2,
            "text": " пример",
            "starts_with_space": True,
            "has_whitespace_in_middle": False,
            "scripts": ["Cyrillic"],
            "script_overlap": False,
        },
        {
            "id": 3,
            "text": "test\tmulti",
            "starts_with_space": False,
            "has_whitespace_in_middle": True,
            "scripts": ["Latin"],
            "script_overlap": False,
        },
    ]


# ===== LANGUAGE DETECTION TESTS =====


@pytest.mark.parametrize(
    "language_info,expected",
    [
        # Chinese variants
        ({"iso_code": "cmn", "script": "Hani"}, True),
        ({"iso_code": "zho", "script": "Hans"}, True),
        ({"iso_code": "yue", "script": "Hant"}, True),
        # Japanese
        ({"iso_code": "jpn", "script": "Jpan"}, True),
        # Thai
        ({"iso_code": "tha", "script": "Thai"}, True),
        # English (space-based)
        ({"iso_code": "eng", "script": "Latn"}, False),
        # Spanish (space-based)
        ({"iso_code": "spa", "script": "Latn"}, False),
        # Russian (space-based)
        ({"iso_code": "rus", "script": "Cyrl"}, False),
        # None/empty
        (None, False),
        ({}, False),
        ({"iso_code": "unknown"}, False),
    ],
)
def test_is_character_based_language(language_info, expected):
    """Test language detection logic."""
    assert is_character_based_language(language_info) == expected


# ===== WORD METRICS TESTS =====


@pytest.mark.parametrize(
    "text,expected_fertility,expected_split_rate,expected_words",
    [
        ("hello world test", 1.0, 0.0, 3),  # No splits
        ("foo bar", 1.0, 0.0, 2),  # No splits, different words
        ("", 0.0, 0.0, 0),  # Empty text
        ("hello", 1.0, 0.0, 1),  # Single word
    ],
)
def test_word_metrics_no_splits(
    simple_tokenizer, text, expected_fertility, expected_split_rate, expected_words
):
    """Test word metrics with tokenizer that doesn't split words."""
    result = calculate_word_metrics(simple_tokenizer, text)

    assert result["subword_fertility"] == expected_fertility
    assert result["continued_word_rate"] == expected_split_rate
    assert result["continuation_token_pct"] == expected_split_rate
    assert result["debug_info"]["total_words"] == expected_words
    assert (
        result["debug_info"]["words_split"]
        == int(
            expected_split_rate * expected_words / 100
        )  # Convert to int to match implementation
    )
    assert result["debug_info"]["segmentation_method"] == "whitespace"


@pytest.mark.parametrize(
    "text,expected_fertility,expected_split_rate",
    [
        ("hello world", 1.5, 33.33),  # 1 continuation out of 3 tokens
        ("testing", 3.0, 66.67),  # 2 continuations out of 3 tokens
    ],
)
def test_word_metrics_with_splits(
    splitting_tokenizer, text, expected_fertility, expected_split_rate
):
    """Test word metrics with tokenizer that splits words."""
    result = calculate_word_metrics(splitting_tokenizer, text)

    assert result["subword_fertility"] == expected_fertility
    assert result["continued_word_rate"] == pytest.approx(expected_split_rate, rel=1e-2)
    assert result["continuation_token_pct"] == pytest.approx(
        expected_split_rate, rel=1e-2
    )
    assert result["debug_info"]["segmentation_method"] == "whitespace"


def test_word_metrics_edge_cases():
    """Test edge cases for word metrics."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(return_value=[])

    # Test with whitespace-only text
    result = calculate_word_metrics(mock_tokenizer, "   \t\n  ")
    assert result["subword_fertility"] == 0.0
    assert result["continued_word_rate"] == 0.0
    assert result["continuation_token_pct"] == 0.0
    assert result["debug_info"]["total_words"] == 0

    # Test with punctuation-only text
    mock_tokenizer.encode = Mock(return_value=[999])  # Single token for punctuation
    result = calculate_word_metrics(mock_tokenizer, "!@#$%")
    assert result["subword_fertility"] == 1.0  # 1 token / 1 word
    assert result["continued_word_rate"] == 0.0  # No splits
    assert result["continuation_token_pct"] == 0.0
    assert result["debug_info"]["total_words"] == 1

    # Test with mixed whitespace and words
    mock_tokenizer.encode = Mock(return_value=[1, 2])
    result = calculate_word_metrics(mock_tokenizer, "  hello   world  ")
    assert result["subword_fertility"] == 1.0  # 2 tokens / 2 words
    assert result["debug_info"]["total_words"] == 2


# ===== CHARACTER-BASED LANGUAGE TESTS =====


def test_chinese_character_segmentation(chinese_tokenizer):
    """Test character-based segmentation for Chinese text."""
    chinese_lang_info = {"iso_code": "cmn", "script": "Hani"}
    chinese_text = "你好世界"  # "Hello World" in Chinese

    result = calculate_word_metrics(chinese_tokenizer, chinese_text, chinese_lang_info)

    # Should segment by character, not by space
    assert result["debug_info"]["segmentation_method"] == "character"
    assert result["debug_info"]["total_words"] == 4  # 4 characters

    # Calculate expected metrics
    # "你" -> [1001, 1002] (split), "好" -> [1003] (not split)
    # "世" -> [1004, 1005] (split), "界" -> [1006] (not split)
    # Total tokens: 6, Total characters: 4
    assert result["subword_fertility"] == 1.5  # 6 tokens / 4 characters
    assert result["continued_word_rate"] == pytest.approx(
        33.33, rel=1e-2
    )  # 2 continuations / 6 tokens
    assert result["continuation_token_pct"] == pytest.approx(33.33, rel=1e-2)


def test_japanese_character_segmentation(chinese_tokenizer):
    """Test character-based segmentation for Japanese text."""
    japanese_lang_info = {"iso_code": "jpn", "script": "Jpan"}
    japanese_text = "你好"  # Using same tokenizer for simplicity

    result = calculate_word_metrics(
        chinese_tokenizer, japanese_text, japanese_lang_info
    )

    assert result["debug_info"]["segmentation_method"] == "character"
    assert result["debug_info"]["total_words"] == 2  # 2 characters
    assert result["subword_fertility"] == 1.5  # 3 tokens / 2 characters
    assert result["continued_word_rate"] == pytest.approx(
        33.33, rel=1e-2
    )  # 1 continuation / 3 tokens
    assert result["continuation_token_pct"] == pytest.approx(33.33, rel=1e-2)


def test_thai_character_segmentation(chinese_tokenizer):
    """Test character-based segmentation for Thai text."""
    thai_lang_info = {"iso_code": "tha", "script": "Thai"}
    thai_text = "你好"  # Using same tokenizer for simplicity

    result = calculate_word_metrics(chinese_tokenizer, thai_text, thai_lang_info)

    assert result["debug_info"]["segmentation_method"] == "character"
    assert result["debug_info"]["total_words"] == 2


def test_english_word_segmentation(simple_tokenizer):
    """Test word-based segmentation for English text."""
    english_lang_info = {"iso_code": "eng", "script": "Latn"}
    english_text = "hello world"

    result = calculate_word_metrics(simple_tokenizer, english_text, english_lang_info)

    assert result["debug_info"]["segmentation_method"] == "whitespace"
    assert result["debug_info"]["total_words"] == 2  # 2 words
    assert result["subword_fertility"] == 1.0  # 2 tokens / 2 words
    assert result["continued_word_rate"] == 0.0  # No splits
    assert result["continuation_token_pct"] == 0.0


def test_mixed_script_text_character_based():
    """Test character-based segmentation with mixed scripts."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4])  # 4 tokens

    chinese_lang_info = {"iso_code": "cmn", "script": "Hani"}
    mixed_text = "Hi你好"  # Mixed English and Chinese

    result = calculate_word_metrics(mock_tokenizer, mixed_text, chinese_lang_info)

    # Should use character-based segmentation because language is Chinese
    assert result["debug_info"]["segmentation_method"] == "character"
    assert result["debug_info"]["total_words"] == 4  # 4 characters: H, i, 你, 好
    assert result["subword_fertility"] == 1.0  # 4 tokens / 4 characters


def test_character_based_with_whitespace_and_punctuation():
    """Test character-based segmentation filtering out whitespace and punctuation."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(return_value=[1, 2])  # 2 tokens

    chinese_lang_info = {"iso_code": "cmn", "script": "Hani"}
    text_with_spaces = "你 好 ! "  # Chinese with spaces and punctuation

    result = calculate_word_metrics(mock_tokenizer, text_with_spaces, chinese_lang_info)

    # Should filter out spaces but keep punctuation
    assert result["debug_info"]["segmentation_method"] == "character"
    assert result["debug_info"]["total_words"] == 3  # 你, 好, ! (spaces filtered out)


def test_language_info_passed_to_debug():
    """Test that language info is preserved in debug output."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(return_value=[1, 2])

    lang_info = {"iso_code": "cmn", "script": "Hani", "name": "Chinese"}

    result = calculate_word_metrics(mock_tokenizer, "你好", lang_info)

    assert result["debug_info"]["language_info"] == lang_info


def test_backward_compatibility_no_language_info():
    """Test that function still works without language info (backward compatibility)."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(return_value=[1, 2])

    # Should default to whitespace segmentation
    result = calculate_word_metrics(mock_tokenizer, "hello world")

    assert result["debug_info"]["segmentation_method"] == "whitespace"
    assert result["debug_info"]["language_info"] is None
    assert result["debug_info"]["total_words"] == 2  # 2 words


# ===== GLOBAL METRICS TRACKER TESTS =====


def test_tracker_initial_state():
    """Test GlobalMetricsTracker initial state."""
    tracker = GlobalMetricsTracker()
    assert tracker.total_token_count == 0
    assert tracker.get_global_metrics() == {}


def test_tracker_basic_metrics(sample_tokens):
    """Test basic GlobalMetricsTracker functionality."""
    tracker = GlobalMetricsTracker()
    tracker.add_tokens(sample_tokens)

    metrics = tracker.get_global_metrics()

    assert metrics["total_tokens_analyzed"] == 3
    assert metrics["tokens_starting_with_space_pct"] == pytest.approx(33.33, rel=1e-2)
    assert metrics["tokens_with_whitespace_in_middle_pct"] == 0.0
    assert metrics["tokens_with_script_overlap_pct"] == 0.0
    assert metrics["tokens_with_latin_unicode_pct"] == pytest.approx(66.67, rel=1e-2)
    assert metrics["tokens_with_punctuation_unicode_pct"] == pytest.approx(
        33.33, rel=1e-2
    )


def test_tracker_script_overlap(mixed_script_tokens):
    """Test GlobalMetricsTracker with script overlap."""
    tracker = GlobalMetricsTracker()
    tracker.add_tokens(mixed_script_tokens)

    metrics = tracker.get_global_metrics()

    assert metrics["tokens_with_script_overlap_pct"] == pytest.approx(33.33, rel=1e-2)
    assert metrics["tokens_with_whitespace_in_middle_pct"] == pytest.approx(
        33.33, rel=1e-2
    )
    assert metrics["tokens_starting_with_space_pct"] == pytest.approx(33.33, rel=1e-2)
    assert metrics["tokens_with_latin_unicode_pct"] == pytest.approx(66.67, rel=1e-2)
    assert metrics["tokens_with_chinese_unicode_pct"] == pytest.approx(33.33, rel=1e-2)
    assert metrics["tokens_with_cyrillic_unicode_pct"] == pytest.approx(33.33, rel=1e-2)


def test_tracker_incremental_updates():
    """Test that tracker correctly handles multiple add_tokens calls."""
    tracker = GlobalMetricsTracker()

    # Add first batch
    batch1 = [
        {
            "id": 1,
            "text": " test",
            "starts_with_space": True,
            "has_whitespace_in_middle": False,
            "scripts": ["Latin"],
            "script_overlap": False,
        }
    ]
    tracker.add_tokens(batch1)

    # Add second batch
    batch2 = [
        {
            "id": 2,
            "text": "data",
            "starts_with_space": False,
            "has_whitespace_in_middle": False,
            "scripts": ["Latin"],
            "script_overlap": False,
        }
    ]
    tracker.add_tokens(batch2)

    metrics = tracker.get_global_metrics()
    assert metrics["total_tokens_analyzed"] == 2
    assert metrics["tokens_starting_with_space_pct"] == 50.0
    assert metrics["tokens_with_latin_unicode_pct"] == 100.0


@pytest.mark.parametrize(
    "tokens_data,expected_scripts",
    [
        ([{"scripts": ["Latin"], "script_overlap": False}], {"latin": 100.0}),
        ([{"scripts": ["Chinese"], "script_overlap": False}], {"chinese": 100.0}),
        (
            [
                {"scripts": ["Latin"], "script_overlap": False},
                {"scripts": ["Chinese"], "script_overlap": False},
            ],
            {"latin": 50.0, "chinese": 50.0},
        ),
    ],
)
def test_tracker_script_percentages(tokens_data, expected_scripts):
    """Test script percentage calculations."""
    tracker = GlobalMetricsTracker()

    # Add required fields to test data
    full_tokens = []
    for i, token_data in enumerate(tokens_data):
        full_token = {
            "id": i + 1,
            "text": f"token{i}",
            "starts_with_space": False,
            "has_whitespace_in_middle": False,
            **token_data,
        }
        full_tokens.append(full_token)

    tracker.add_tokens(full_tokens)
    metrics = tracker.get_global_metrics()

    for script, expected_pct in expected_scripts.items():
        key = f"tokens_with_{script}_unicode_pct"
        assert metrics[key] == pytest.approx(expected_pct, rel=1e-2)


def test_tracker_empty_tokens():
    """Test tracker with empty token list."""
    tracker = GlobalMetricsTracker()
    tracker.add_tokens([])

    assert tracker.total_token_count == 0
    assert tracker.get_global_metrics() == {}


def test_tracker_malformed_tokens():
    """Test tracker robustness with malformed token data."""
    tracker = GlobalMetricsTracker()

    # Token missing some fields - should not crash
    malformed_tokens = [{"id": 1, "text": "test"}]  # Missing required fields

    # Should handle gracefully (implementation dependent)
    try:
        tracker.add_tokens(malformed_tokens)
        # If it doesn't crash, verify it handles it somehow
        metrics = tracker.get_global_metrics()
        assert isinstance(metrics, dict)
    except (KeyError, AttributeError):
        # Expected if implementation requires all fields
        pass
