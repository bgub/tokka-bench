"""
Unicode script detection and text analysis utilities.

This module provides functions for analyzing Unicode scripts and text properties,
which are used throughout the tokenizer benchmarking process.
"""

import unicodedata
from typing import Dict, List, Set

# Unicode category constants
PUNCTUATION_CATEGORY_PREFIX = "P"
SYMBOL_CATEGORY_PREFIX = "S"
NUMBER_CATEGORY_PREFIX = "N"

# Special symbol keywords
MATHEMATICAL_KEYWORDS = {"MATHEMATICAL", "DOUBLE-STRUCK"}


# Unicode script mappings for major writing systems
# Order matters: more specific scripts should come first
UNICODE_SCRIPTS: Dict[str, List[str]] = {
    "Korean": ["HANGUL"],  # Must come before Chinese to avoid HAN matching in HANGUL
    "Japanese": ["HIRAGANA", "KATAKANA"],
    "Chinese": ["CJK", "HAN"],  # CJK covers Chinese ideographs
    "Latin": ["LATIN"],
    "Cyrillic": ["CYRILLIC"],
    "Arabic": ["ARABIC"],
    "Devanagari": ["DEVANAGARI"],
    "Thai": ["THAI"],
    "Hebrew": ["HEBREW"],
    "Greek": ["GREEK"],
}


def get_unicode_scripts(text: str) -> Set[str]:
    """Get the set of Unicode scripts present in the text."""
    if not text:
        return set()

    scripts: Set[str] = set()
    for char in text:
        if char.isspace():
            continue

        # Get character name and category with error handling
        try:
            char_name: str = unicodedata.name(char, "")
            category: str = unicodedata.category(char)
        except (ValueError, TypeError):
            # Skip characters that can't be analyzed
            continue

        # Check for mathematical symbols first (they're classified as Lu but should be Symbols)
        if any(keyword in char_name for keyword in MATHEMATICAL_KEYWORDS):
            scripts.add("Symbols")
            continue

        # Check for punctuation, symbols, numbers using category prefixes
        if category.startswith(
            PUNCTUATION_CATEGORY_PREFIX
        ):  # Punctuation (Po, Pc, Pd, Ps, Pe, Pi, Pf)
            scripts.add("Punctuation")
            continue
        elif category.startswith(SYMBOL_CATEGORY_PREFIX):  # Symbols (Sm, Sc, Sk, So)
            scripts.add("Symbols")
            continue
        elif category.startswith(NUMBER_CATEGORY_PREFIX):  # Numbers (Nd, Nl, No)
            scripts.add("Numbers")
            continue

        # Check for script families using the full character name
        for script_name, script_codes in UNICODE_SCRIPTS.items():
            if any(script_code in char_name for script_code in script_codes):
                scripts.add(script_name)
                break
    return scripts


def has_whitespace_in_middle(text: str) -> bool:
    """Check if text has whitespace characters in the middle (not at start/end)."""
    if not text:
        return False

    stripped = text.strip()
    return len(stripped) > 0 and any(char.isspace() for char in stripped)


def starts_with_space(text: str) -> bool:
    """Check if text starts with a whitespace character."""
    return bool(text and text[0].isspace())
