"""
Language categorization and filtering utilities.
"""

from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd

# SCRIPT_GROUPS is not needed in the streamlined categories


def detect_language_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build the exact category sets requested, with careful ordering and labeling.

    Categories returned (in order):
    - All Languages
    - Top 30 Natural (English first)
    - 31–60 Natural
    - 61–100 Natural
    - Coding
    - European
    - Non-European
    - Latin Script
    - Cyrillic Script
    - Arabic Script
    - CJK Scripts
    """

    # Languages present in the dataset
    all_languages = list(df["language"].unique())

    # Popularity order if available (lower rank = more popular)
    language_rank_map: Dict[str, Optional[float]] = (
        df.groupby("language")["language_rank"].min().to_dict()
        if "language_rank" in df.columns
        else {}
    )

    def rank_key(lang: str) -> float:
        rank = language_rank_map.get(lang)
        return rank if rank is not None else 1e9

    # Identify programming vs natural and English
    programming_languages: List[str] = []
    natural_languages: List[str] = []
    english_languages: List[str] = []
    for lang in all_languages:
        lang_row = df[df["language"] == lang].iloc[0]
        script_value = str(lang_row.get("script", "")).lower()
        if "(code)" in str(lang).lower() or script_value == "code":
            programming_languages.append(lang)
        elif "english" in str(lang).lower():
            english_languages.append(lang)
        else:
            natural_languages.append(lang)

    # Ordering
    natural_by_rank = sorted(natural_languages, key=rank_key)
    programming_by_rank = sorted(programming_languages, key=rank_key)

    # All Languages overall order: English (if present) → natural by rank → coding by rank
    ordered_all_languages = english_languages + natural_by_rank + programming_by_rank

    # Natural ranges with English at the front of Top 30
    top_natural_with_english: List[str] = english_languages[:1] + [
        l for l in natural_by_rank if l not in english_languages
    ]
    top_30_natural = top_natural_with_english[:30]
    natural_31_60 = top_natural_with_english[30:60]
    natural_61_100 = top_natural_with_english[60:100]

    # Script groupings from df
    script_map: Dict[str, str] = {
        lang: str(df[df["language"] == lang].iloc[0].get("script", ""))
        for lang in all_languages
    }
    latin_script = [l for l, s in script_map.items() if "latn" in s.lower()]
    cyrillic_script = [l for l, s in script_map.items() if "cyrl" in s.lower()]
    arabic_script = [l for l, s in script_map.items() if "arab" in s.lower()]
    cjk_scripts = [
        l
        for l, s in script_map.items()
        if any(tag in s.lower() for tag in ["hani", "jpan", "hang"])
    ]

    # European vs Non-European using CSV families plus script as a guard
    # Load FineWeb-2 CSV to access the Language Family column
    try:
        repo_root = Path(__file__).resolve().parents[1]
        fineweb_csv = repo_root / "fineweb-2-languages.csv"
        name_to_family: Dict[str, str] = {}
        if fineweb_csv.exists():
            fw = pd.read_csv(fineweb_csv)
            for _, row in fw.iterrows():
                name_to_family[str(row.get("Name", ""))] = str(
                    row.get("Language Family", "")
                )
        european_families = {"Indo-European", "Uralic", "Turkic", "Kartvelian"}
        european_name_exceptions = {"Basque", "Maltese"}

        def is_european(lang: str) -> bool:
            # Ignore coding
            if lang in programming_languages:
                return False
            fam = name_to_family.get(lang, "")
            scr = script_map.get(lang, "").lower()
            if lang in european_name_exceptions:
                return True
            # Require European-associated script and qualifying family
            if any(tag in scr for tag in ["latn", "cyrl", "grek"]) and (
                fam in european_families
            ):
                return True
            return False

        european = [l for l in natural_by_rank if is_european(l)]
    except Exception:
        # Fallback: script-only heuristic
        european = [
            l
            for l, s in script_map.items()
            if any(tag in s.lower() for tag in ["latn", "cyrl", "grek"])
            and l in natural_languages
        ]

    non_european = [l for l in natural_by_rank if l not in set(european)]

    return {
        "All Languages": ordered_all_languages,
        "Top 30 Natural": top_30_natural,
        "31–60 Natural": natural_31_60,
        "61–100 Natural": natural_61_100,
        "Coding": programming_by_rank,
        "European": european,
        "Non-European": non_european,
        "Latin Script": latin_script,
        "Cyrillic Script": cyrillic_script,
        "Arabic Script": arabic_script,
        "CJK Scripts": cjk_scripts,
    }
