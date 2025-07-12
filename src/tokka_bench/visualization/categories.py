"""
Language categorization and filtering utilities.
"""

from typing import Dict, List

import pandas as pd

from tokka_bench.visualization.constants import SCRIPT_GROUPS


def detect_language_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect and categorize languages from the data with comprehensive category system."""
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

    # === SIZE-BASED TIERS ===
    # Top 10 (English + top 9 natural languages by size)
    top_10 = english_languages + natural_languages[:9]

    # 11-40: Next 30 natural languages (if we have enough)
    tier_11_40 = natural_languages[9:39] if len(natural_languages) > 9 else []

    # 41-100: Remaining natural languages (60 languages: positions 40-99 in natural_languages)
    tier_41_100 = natural_languages[39:] if len(natural_languages) > 39 else []

    # === PROGRAMMING LANGUAGE CATEGORIES ===
    # Core programming languages (most popular)
    core_programming = [
        lang
        for lang in programming_languages
        if any(
            tech in lang.lower()
            for tech in [
                "python",
                "javascript",
                "java",
                "c ",
                "cpp",
                "c-sharp",
                "php",
                "typescript",
            ]
        )
    ]

    # Systems programming languages
    systems_programming = [
        lang
        for lang in programming_languages
        if any(
            tech in lang.lower()
            for tech in ["rust", "go", "c", "cpp", "zig", "assembly", "fortran"]
        )
    ]

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
    asian_scripts_langs = [
        lang
        for lang, script in script_info.items()
        if any(
            s in script.lower()
            for s in ["hani", "jpan", "hang", "thai", "laoo", "khmr"]
        )
    ]
    arabic_script = [
        lang for lang, script in script_info.items() if "arab" in script.lower()
    ]

    # === REGIONAL/FAMILY CATEGORIES ===
    # European languages (Latin + Cyrillic + Greek)
    european_langs = [
        lang
        for lang, script in script_info.items()
        if any(s in script.lower() for s in ["latn", "cyrl", "grek"])
        and lang not in programming_languages
    ]

    # Major world languages (most speakers/important) - top 20 by size
    major_world = (
        english_languages + natural_languages[:19]
        if len(natural_languages) >= 19
        else english_languages + natural_languages
    )

    return {
        # Core categories (always include)
        "All Languages": ordered_all_languages,
        "Top 10": top_10,
        "11-40": tier_11_40,
        "41-100": tier_41_100,
        # Language type categories
        "Natural Languages": natural_languages,
        "Programming Languages": programming_languages,
        "Core Programming": core_programming,
        "Systems Programming": systems_programming,
        # Script-based categories
        "Latin Script": latin_script,
        "Cyrillic Script": cyrillic_script,
        "Asian Scripts": asian_scripts_langs,
        "Arabic Script": arabic_script,
        # Regional/family categories
        "European Languages": european_langs,
        "Major World Languages": major_world,
        # Special categories
        "English": english_languages,
    }
