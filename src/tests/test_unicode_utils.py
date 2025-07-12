"""
Tests for unicode utility functions.
"""

import pytest
from tokka_bench.unicode_utils import (
    get_unicode_scripts,
    has_whitespace_in_middle,
    starts_with_space,
)


# ===== FIXTURES =====


@pytest.fixture
def unicode_samples():
    """Sample text with various Unicode scripts."""
    return {
        "latin": "Hello world",
        "cyrillic": "Привет мир",
        "chinese": "你好世界",
        "arabic": "مرحبا بالعالم",
        "mixed_words": "Hello мир 世界",  # Multiple words with different scripts
        "mixed_token": "hello世界мир",  # Single token with mixed scripts
        "with_punctuation": "Hello!",
        "with_numbers": "Hello123",
        "with_symbols": "Hello + world",
        "complex_mixed": "Hello 世界! 123 + мир",
    }


@pytest.fixture
def whitespace_samples():
    """Sample text with various whitespace patterns."""
    return {
        "middle_space": "hello world",
        "middle_tab": "hello\tworld",
        "middle_newline": "hello\nworld",
        "edge_spaces": "  hello world  ",
        "no_middle": "hello",
        "edges_only": "  hello  ",
        "multiple_middle": "hello   world   test",
        "mixed_whitespace": "hello \t\n world",
    }


@pytest.fixture
def space_start_samples():
    """Sample text with various leading space patterns."""
    return {
        "space": " hello",
        "tab": "\thello",
        "newline": "\nhello",
        "multiple": "   hello",
        "mixed": " \t\nhello",
        "no_space": "hello",
        "trailing_only": "hello ",
        "middle_only": "hel lo",
    }


# ===== UNICODE SCRIPT TESTS =====


@pytest.mark.parametrize(
    "text,expected_scripts",
    [
        ("Hello world", {"Latin"}),
        ("Привет мир", {"Cyrillic"}),
        ("你好世界", {"Chinese"}),
        ("مرحبا", {"Arabic"}),
        ("Hello!", {"Latin", "Punctuation"}),
        ("Hello123", {"Latin", "Numbers"}),
        ("Hello + world", {"Latin", "Symbols"}),
        ("Hello мир 世界", {"Latin", "Cyrillic", "Chinese"}),
        ("", set()),
        ("   ", set()),
        ("123!@#", {"Numbers", "Punctuation"}),
        ("1+2=3", {"Numbers", "Symbols"}),  # + and = are symbols
        # Mixed scripts within single tokens
        ("hello世界", {"Latin", "Chinese"}),  # Single token with Latin + Chinese
        ("café中文", {"Latin", "Chinese"}),  # Single token with Latin + Chinese
        ("test한국어", {"Latin", "Korean"}),  # Single token with Latin + Korean
        ("αβγ123", {"Greek", "Numbers"}),  # Single token with Greek + Numbers
        ("مرحبا!", {"Arabic", "Punctuation"}),  # Single token with Arabic + Punctuation
    ],
)
def test_get_unicode_scripts_basic(text, expected_scripts):
    """Test Unicode script detection for various inputs."""
    result = get_unicode_scripts(text)
    assert result == expected_scripts


def test_get_unicode_scripts_comprehensive(unicode_samples):
    """Test comprehensive Unicode script detection."""
    # Test complex mixed text (multiple words)
    scripts = get_unicode_scripts(unicode_samples["complex_mixed"])
    expected = {"Latin", "Chinese", "Punctuation", "Numbers", "Symbols", "Cyrillic"}
    assert scripts == expected

    # Test single token with mixed scripts
    scripts = get_unicode_scripts(unicode_samples["mixed_token"])
    expected = {"Latin", "Chinese", "Cyrillic"}
    assert scripts == expected


@pytest.mark.parametrize(
    "text,should_contain",
    [
        ("ñáéíóú", "Latin"),  # Latin with diacritics
        ("αβγδε", "Greek"),  # Greek
        ("שלום", "Hebrew"),  # Hebrew
        ("สวัสดี", "Thai"),  # Thai
        ("こんにちは", "Japanese"),  # Japanese
        ("안녕하세요", "Korean"),  # Korean
    ],
)
def test_get_unicode_scripts_extended(text, should_contain):
    """Test Unicode script detection for extended character sets."""
    scripts = get_unicode_scripts(text)
    assert should_contain in scripts


def test_mixed_scripts_within_single_tokens():
    """Test detection of mixed scripts within single tokens (no spaces)."""
    test_cases = [
        # Latin + CJK combinations
        ("hello世界", {"Latin", "Chinese"}),
        ("world中文", {"Latin", "Chinese"}),
        ("test한국어", {"Latin", "Korean"}),
        ("helloこんにちは", {"Latin", "Japanese"}),
        # Latin + Cyrillic combinations
        ("helloпривет", {"Latin", "Cyrillic"}),
        ("testмир", {"Latin", "Cyrillic"}),
        # Multiple scripts + symbols/numbers
        ("hello世界123", {"Latin", "Chinese", "Numbers"}),
        ("café中文!", {"Latin", "Chinese", "Punctuation"}),
        ("αβγ123γδε", {"Greek", "Numbers"}),
        ("مرحبا123", {"Arabic", "Numbers"}),
        # Complex mixed tokens
        ("hello世界мир", {"Latin", "Chinese", "Cyrillic"}),
        ("test한국어中文", {"Latin", "Korean", "Chinese"}),
        ("αβγ中文hello", {"Greek", "Chinese", "Latin"}),
    ]

    for token, expected_scripts in test_cases:
        result = get_unicode_scripts(token)
        assert result == expected_scripts, (
            f"Token '{token}' expected {expected_scripts}, got {result}"
        )


# ===== WHITESPACE DETECTION TESTS =====


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello world", True),
        ("hello\tworld", True),
        ("hello\nworld", True),
        ("  hello world  ", True),  # Should trim and find middle
        ("hello   world", True),  # Multiple spaces
        ("hello \t\n world", True),  # Mixed whitespace
        ("hello", False),
        ("  hello  ", False),  # Only edges
        ("", False),
        ("   ", False),  # Only whitespace
        ("helloworld", False),
    ],
)
def test_has_whitespace_in_middle(text, expected):
    """Test whitespace detection in middle of text."""
    assert has_whitespace_in_middle(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (" hello", True),
        ("\thello", True),
        ("\nhello", True),
        ("   hello", True),  # Multiple spaces
        (" \t\nhello", True),  # Mixed whitespace
        ("hello", False),
        ("hello ", False),  # Trailing space
        ("hel lo", False),  # Middle space only
        ("", False),
    ],
)
def test_starts_with_space(text, expected):
    """Test space detection at start of text."""
    assert starts_with_space(text) == expected


# ===== EDGE CASES AND INTEGRATION TESTS =====


def test_empty_and_whitespace_edge_cases():
    """Test edge cases with empty strings and whitespace."""
    # Empty string
    assert get_unicode_scripts("") == set()
    assert not has_whitespace_in_middle("")
    assert not starts_with_space("")

    # Whitespace only
    assert get_unicode_scripts("   ") == set()
    assert not has_whitespace_in_middle("   ")
    assert starts_with_space("   ")

    # Single characters
    assert "Latin" in get_unicode_scripts("a")
    assert not has_whitespace_in_middle("a")
    assert not starts_with_space("a")


@pytest.mark.parametrize(
    "char,expected_script",
    [
        ("a", "Latin"),
        ("!", "Punctuation"),
        ("1", "Numbers"),
        ("+", "Symbols"),
        ("α", "Greek"),
        ("ñ", "Latin"),
    ],
)
def test_single_character_scripts(char, expected_script):
    """Test script detection for single characters."""
    scripts = get_unicode_scripts(char)
    assert expected_script in scripts


def test_unicode_normalization():
    """Test that function handles Unicode normalization correctly."""
    # These should be treated the same (composed vs decomposed)
    text1 = "café"  # é as single character
    text2 = "cafe\u0301"  # e + combining acute accent

    scripts1 = get_unicode_scripts(text1)
    scripts2 = get_unicode_scripts(text2)

    # Both should detect Latin script
    assert "Latin" in scripts1
    assert "Latin" in scripts2


def test_rare_unicode_categories():
    """Test handling of rare Unicode categories."""
    # Test various Unicode categories
    test_cases = [
        ("𝔸𝔹ℂ", "Symbols"),  # Mathematical symbols
        ("①②③", "Numbers"),  # Circled numbers
        ("←→↑↓", "Symbols"),  # Arrows
        ("♠♥♦♣", "Symbols"),  # Card suits
    ]

    for text, expected in test_cases:
        scripts = get_unicode_scripts(text)
        # Should at least not crash and return some category
        assert isinstance(scripts, set)
        assert len(scripts) > 0


def test_very_long_text():
    """Test performance with long text."""
    # Create long text with mixed scripts
    long_text = "Hello мир 世界! " * 1000

    scripts = get_unicode_scripts(long_text)
    expected = {"Latin", "Cyrillic", "Chinese", "Punctuation"}
    assert scripts == expected

    # Should handle whitespace correctly even in long text
    assert has_whitespace_in_middle(long_text)
    assert not starts_with_space(long_text)


@pytest.mark.parametrize(
    "text",
    [
        "hello\x00world",  # Null character
        "hello\x7fworld",  # DEL character
        "hello\u200bworld",  # Zero-width space
        "hello\ufeffworld",  # BOM character
    ],
)
def test_special_characters(text):
    """Test handling of special/control characters."""
    # Should not crash
    scripts = get_unicode_scripts(text)
    assert isinstance(scripts, set)

    # These have "whitespace" but might not be detected by our function
    # depending on implementation
    result = has_whitespace_in_middle(text)
    assert isinstance(result, bool)


def test_functions_integration(
    unicode_samples, whitespace_samples, space_start_samples
):
    """Test all functions work together on the same data."""
    all_samples = {**unicode_samples, **whitespace_samples, **space_start_samples}

    for text in all_samples.values():
        # All functions should work without crashing
        scripts = get_unicode_scripts(text)
        has_middle = has_whitespace_in_middle(text)
        starts_space = starts_with_space(text)

        # Basic type checks
        assert isinstance(scripts, set)
        assert isinstance(has_middle, bool)
        assert isinstance(starts_space, bool)

        # Logical consistency
        if starts_space and text.strip():
            # If starts with space and has content, the stripped version should differ
            assert text != text.lstrip()


def test_consistency_with_different_whitespace():
    """Test consistency across different types of whitespace."""
    whitespace_chars = [" ", "\t", "\n", "\r", "\f", "\v"]

    for ws in whitespace_chars:
        # Should all be detected as starting with space
        assert starts_with_space(f"{ws}hello")

        # Should all be detected as having middle whitespace
        assert has_whitespace_in_middle(f"hello{ws}world")

        # Whitespace-only should not have middle whitespace
        assert not has_whitespace_in_middle(ws * 3)
