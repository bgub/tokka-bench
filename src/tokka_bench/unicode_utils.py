"""
Unicode script detection and text analysis utilities.

This module provides functions for analyzing Unicode scripts and text properties,
which are used throughout the tokenizer benchmarking process.
"""

import unicodedata
from typing import Dict, List, Set


# Unicode script mappings for major writing systems
UNICODE_SCRIPTS: Dict[str, List[str]] = {
    "Latin": ["LATIN"],
    "Chinese": ["CJK", "HAN"],  # CJK covers Chinese ideographs
    "Cyrillic": ["CYRILLIC"],
    "Korean": ["HANGUL"],
    "Japanese": ["HIRAGANA", "KATAKANA"],
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

        # Check for punctuation, symbols, numbers first
        category: str = unicodedata.category(char)
        if category.startswith("P"):  # Punctuation (Po, Pc, Pd, Ps, Pe, Pi, Pf)
            scripts.add("Punctuation")
            continue
        elif category.startswith("S"):  # Symbols (Sm, Sc, Sk, So)
            scripts.add("Symbols")
            continue
        elif category.startswith("N"):  # Numbers (Nd, Nl, No)
            scripts.add("Numbers")
            continue

        # Check for script families
        script: str = (
            unicodedata.name(char, "").split()[0] if unicodedata.name(char, "") else ""
        )
        for script_name, script_codes in UNICODE_SCRIPTS.items():
            if any(script_code in script for script_code in script_codes):
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
