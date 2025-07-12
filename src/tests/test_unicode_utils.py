"""
Tests for unicode utility functions.

This demonstrates how to test the split utility functions.
"""

from tokka_bench.unicode_utils import (
    get_unicode_scripts,
    has_whitespace_in_middle,
    starts_with_space,
)


def test_get_unicode_scripts():
    """Test Unicode script detection."""
    # Test Latin text
    assert "Latin" in get_unicode_scripts("Hello world")

    # Test punctuation
    assert "Punctuation" in get_unicode_scripts("Hello!")

    # Test numbers
    assert "Numbers" in get_unicode_scripts("Hello123")

    # Test symbols
    assert "Symbols" in get_unicode_scripts("Hello + world")

    # Test mixed scripts
    scripts = get_unicode_scripts("Hello мир 世界")
    assert "Latin" in scripts
    assert "Cyrillic" in scripts
    assert "Chinese" in scripts

    # Test empty string
    assert get_unicode_scripts("") == set()

    # Test whitespace only
    assert get_unicode_scripts("   ") == set()


def test_has_whitespace_in_middle():
    """Test whitespace detection in middle of text."""
    # Should detect whitespace in middle
    assert has_whitespace_in_middle("hello world")
    assert has_whitespace_in_middle("  hello world  ")  # Trimmed internally
    assert has_whitespace_in_middle("hello\tworld")  # Tab
    assert has_whitespace_in_middle("hello\nworld")  # Newline

    # Should not detect whitespace if none in middle
    assert not has_whitespace_in_middle("hello")
    assert not has_whitespace_in_middle("  hello  ")  # Only at edges
    assert not has_whitespace_in_middle("")
    assert not has_whitespace_in_middle("   ")  # Only whitespace


def test_starts_with_space():
    """Test space detection at start of text."""
    # Should detect space at start
    assert starts_with_space(" hello")
    assert starts_with_space("\thello")  # Tab
    assert starts_with_space("\nhello")  # Newline

    # Should not detect space if none at start
    assert not starts_with_space("hello")
    assert not starts_with_space("hello ")  # Space at end
    assert not starts_with_space("")  # Empty string


def test_edge_cases():
    """Test edge cases for all functions."""
    # Empty string tests
    assert get_unicode_scripts("") == set()
    assert not has_whitespace_in_middle("")
    assert not starts_with_space("")

    # Single character tests
    assert "Latin" in get_unicode_scripts("a")
    assert "Punctuation" in get_unicode_scripts("!")
    assert "Numbers" in get_unicode_scripts("1")
    assert "Symbols" in get_unicode_scripts("+")

    # Complex Unicode test
    complex_text = "Hello 世界! 123 + мир"
    scripts = get_unicode_scripts(complex_text)
    expected_scripts = {
        "Latin",
        "Chinese",
        "Punctuation",
        "Numbers",
        "Symbols",
        "Cyrillic",
    }
    assert scripts == expected_scripts
