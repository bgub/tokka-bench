"""
Language categorization and filtering utilities.
"""

from typing import Dict, List

import pandas as pd

# SCRIPT_GROUPS is not needed in the streamlined categories


def detect_language_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect and categorize languages with a streamlined category set."""
    # Get languages in their original order from the data (preserves CSV order)
    all_languages = list(df["language"].unique())

    # Detect programming languages (those with "code" in script or name)
    programming_languages = []
    natural_languages = []
    english_languages = []

    for lang in all_languages:
        lang_data = df[df["language"] == lang].iloc[0]
        # Check if language name contains "(code)" or script is "code"
        if "(code)" in str(lang).lower() or str(lang_data["script"]).lower() == "code":
            programming_languages.append(lang)
        elif "english" in str(lang).lower() or "eng-fineweb" in str(lang).lower():
            english_languages.append(lang)
        else:
            natural_languages.append(lang)

    # Preserve proper ordering: English → Natural → Programming
    ordered_all_languages = (
        english_languages + natural_languages + programming_languages
    )

    # === SIZE-BASED TIERS (streamlined) ===
    # Top 10: English + top 9 natural languages
    top_10 = english_languages + natural_languages[:9]

    # === PROGRAMMING LANGUAGE CATEGORIES (streamlined) ===
    # Keep only broad programming categories

    # === SCRIPT-BASED CATEGORIES ===
    # Get script information for each language
    script_info = {}
    for lang in all_languages:
        lang_data = df[df["language"] == lang].iloc[0]
        script_info[lang] = str(lang_data["script"])

    # Group by script families
    latin_script = [
        lang for lang, script in script_info.items() if "latn" in script.lower()
    ]
    cyrillic_script = [
        lang for lang, script in script_info.items() if "cyrl" in script.lower()
    ]
    cjk_scripts = [
        lang
        for lang, script in script_info.items()
        if any(s in script.lower() for s in ["hani", "jpan", "hang"])
    ]
    # Optional: Arabic already separate; omit SE Asian to reduce noise
    arabic_script = [
        lang for lang, script in script_info.items() if "arab" in script.lower()
    ]

    # === REGIONAL/FAMILY CATEGORIES (streamlined) ===
    # European languages (Latin + Cyrillic + Greek)
    european_langs = [
        lang
        for lang, script in script_info.items()
        if any(s in script.lower() for s in ["latn", "cyrl", "grek"])
        and lang not in programming_languages
    ]

    return {
        # Core categories (always include)
        "All Languages": ordered_all_languages,
        "Top 10": top_10,
        # Language type categories
        "Natural Languages": natural_languages,
        "Programming Languages": programming_languages,
        # Script-based categories
        "Latin Script": latin_script,
        "Cyrillic Script": cyrillic_script,
        "CJK Scripts": cjk_scripts,
        "Arabic Script": arabic_script,
        # Regional/family categories
        "European Languages": european_langs,
        # Special categories
        "English": english_languages,
    }
