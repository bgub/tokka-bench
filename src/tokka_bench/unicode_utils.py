"""
Unicode script detection and text analysis utilities.

This module provides functions for analyzing Unicode scripts and text properties,
which are used throughout the tokenizer benchmarking process.
"""

import unicodedata
from typing import Dict, List, Set


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
    scripts: Set[str] = set()
    for char in text:
        if char.isspace():
            continue

        # Get character name and category
        char_name: str = unicodedata.name(char, "")
        category: str = unicodedata.category(char)

        # Check for mathematical symbols first (they're classified as Lu but should be Symbols)
        if "MATHEMATICAL" in char_name or "DOUBLE-STRUCK" in char_name:
            scripts.add("Symbols")
            continue

        # Check for punctuation, symbols, numbers
        if category.startswith("P"):  # Punctuation (Po, Pc, Pd, Ps, Pe, Pi, Pf)
            scripts.add("Punctuation")
            continue
        elif category.startswith("S"):  # Symbols (Sm, Sc, Sk, So)
            scripts.add("Symbols")
            continue
        elif category.startswith("N"):  # Numbers (Nd, Nl, No)
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
    stripped = text.strip()
    return len(stripped) > 0 and any(char.isspace() for char in stripped)


def starts_with_space(text: str) -> bool:
    """Check if text starts with a whitespace character."""
    return len(text) > 0 and text[0].isspace()
