"""
Data loading and processing utilities for visualization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from .constants import RESULTS_DIR


def load_all_results(results_dir: str = RESULTS_DIR) -> Dict[str, Any]:
    """Load all JSON result files from the results directory."""
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        return results

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results[json_file.stem] = data
        except Exception as e:
            st.warning(f"Could not load {json_file}: {e}")

    return results


def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert results dictionary to a pandas DataFrame for analysis."""
    rows = []

    for tokenizer_key, result in results.items():
        tokenizer_name = result.get("tokenizer", tokenizer_key)
        benchmark_size = result.get("benchmark_size_mb", 1.0)
        timestamp = result.get("timestamp", 0)
        vocab_size = result.get("vocab_size", None)

        # Extract vocab metrics if available
        vocab_metrics = result.get("vocab_metrics", {})
        global_metrics = result.get("global_metrics", {})

        for lang_key, lang_data in result.get("languages", {}).items():
            lang_info = lang_data["language_info"]
            metrics = lang_data["metrics"]
            sample = lang_data.get("sample", {})

            row = {
                "tokenizer_key": tokenizer_key,
                "tokenizer_name": tokenizer_name,
                "language": lang_info["name"],
                "iso_code": lang_info["iso_code"],
                "script": lang_info["script"],
                "lang_key": lang_key,
                "bytes_per_token": metrics["bytes_per_token"],
                "total_bytes": metrics["total_bytes"],
                "total_tokens": metrics["total_tokens"],
                "unique_tokens": metrics["unique_tokens"],
                "vocab_size": vocab_size,
                "benchmark_size_mb": benchmark_size,
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp) if timestamp else None,
            }

            # Add new per-language metrics if available
            if "subword_fertility" in metrics:
                row["subword_fertility"] = metrics["subword_fertility"]
            # Prefer new word_split_pct; keep backward-compat if only continued_word_rate exists
            if "word_split_pct" in metrics:
                row["word_split_pct"] = metrics["word_split_pct"]
            elif "continued_word_rate" in metrics:
                row["word_split_pct"] = metrics["continued_word_rate"]

            # Add vocab metrics (same for all languages of a tokenizer)
            if vocab_metrics:
                row["tokens_without_leading_space_pct"] = vocab_metrics.get(
                    "tokens_without_leading_space_pct"
                )
                row["tokens_without_leading_space_count"] = vocab_metrics.get(
                    "tokens_without_leading_space_count"
                )
                row["analyzed_sample_size"] = vocab_metrics.get("analyzed_sample_size")

            # Add global metrics (same for all languages of a tokenizer)
            if global_metrics:
                row["total_tokens_analyzed"] = global_metrics.get(
                    "total_tokens_analyzed"
                )
                row["tokens_starting_with_space_pct"] = global_metrics.get(
                    "tokens_starting_with_space_pct"
                )
                row["tokens_with_whitespace_in_middle_pct"] = global_metrics.get(
                    "tokens_with_whitespace_in_middle_pct"
                )
                row["tokens_with_script_overlap_pct"] = global_metrics.get(
                    "tokens_with_script_overlap_pct"
                )
                row["tokens_with_symbols_unicode_pct"] = global_metrics.get(
                    "tokens_with_symbols_unicode_pct"
                )
                row["tokens_with_latin_unicode_pct"] = global_metrics.get(
                    "tokens_with_latin_unicode_pct"
                )
                row["tokens_with_punctuation_unicode_pct"] = global_metrics.get(
                    "tokens_with_punctuation_unicode_pct"
                )
                row["tokens_with_numbers_unicode_pct"] = global_metrics.get(
                    "tokens_with_numbers_unicode_pct"
                )
                row["tokens_with_japanese_unicode_pct"] = global_metrics.get(
                    "tokens_with_japanese_unicode_pct"
                )
                row["tokens_with_chinese_unicode_pct"] = global_metrics.get(
                    "tokens_with_chinese_unicode_pct"
                )
                row["tokens_with_cyrillic_unicode_pct"] = global_metrics.get(
                    "tokens_with_cyrillic_unicode_pct"
                )
                row["tokens_with_greek_unicode_pct"] = global_metrics.get(
                    "tokens_with_greek_unicode_pct"
                )
                row["tokens_with_korean_unicode_pct"] = global_metrics.get(
                    "tokens_with_korean_unicode_pct"
                )
                row["tokens_with_arabic_unicode_pct"] = global_metrics.get(
                    "tokens_with_arabic_unicode_pct"
                )
                row["tokens_with_hebrew_unicode_pct"] = global_metrics.get(
                    "tokens_with_hebrew_unicode_pct"
                )

            # Add sample preview fields if available
            if sample:
                row["sample_text"] = sample.get("text")
                row["sample_byte_offset"] = sample.get("byte_offset")
                row["sample_text_bytes"] = sample.get("text_bytes")
                tokens_list = sample.get("tokens") or []
                if tokens_list:
                    row["sample_tokens"] = tokens_list
                    row["sample_token_count"] = len(tokens_list)
                    # Join a short preview for hover; avoid huge hovers (no spaces around separator)
                    row["sample_tokens_preview"] = "|".join(tokens_list[:20])

            rows.append(row)

    df = pd.DataFrame(rows)

    # Attach a stable language presentation order using project CSV metadata
    try:
        order_map = build_language_order_map()
        english_key = "English (FineWeb)"
        language_ranks: Dict[str, int] = {}

        english_present = english_key in set(df["language"].unique())
        current_rank = 0
        if english_present:
            language_ranks[english_key] = current_rank
            current_rank += 1

        # Natural languages in CSV order (size desc)
        natural_items = sorted(
            [
                (name, idx)
                for name, idx in order_map.items()
                if not name.endswith("(code)")
            ],
            key=lambda x: x[1],
        )
        for name, _ in natural_items:
            if name not in language_ranks:
                language_ranks[name] = current_rank
                current_rank += 1

        # Programming languages in StarCoder CSV order
        prog_items = sorted(
            [(name, idx) for name, idx in order_map.items() if name.endswith("(code)")],
            key=lambda x: x[1],
        )
        for name, _ in prog_items:
            if name not in language_ranks:
                language_ranks[name] = current_rank
                current_rank += 1

        # Any remaining languages not covered by CSVs go last in alpha order
        for name in sorted(set(df["language"].unique())):
            if name not in language_ranks:
                language_ranks[name] = current_rank
                current_rank += 1

        df["language_rank"] = df["language"].map(language_ranks)
    except Exception:
        # Fallback: no ranking
        df["language_rank"] = None

    return df


def build_language_order_map() -> Dict[str, int]:
    """Build a mapping of language display name -> stable rank.

    - English (FineWeb) handled in results_to_dataframe to be first
    - Natural languages: ordered by FineWeb-2 disk size (desc)
    - Programming languages: StarCoder CSV order (top to bottom)
    """
    repo_root = Path(__file__).resolve().parents[1]
    fineweb_csv = repo_root / "fineweb-2-languages.csv"
    starcoder_csv = repo_root / "starcoderdata-dirs.csv"

    order_map: Dict[str, int] = {}

    # Natural languages by size desc
    if fineweb_csv.exists():
        fw = pd.read_csv(fineweb_csv)

        def parse_size(size_str: str) -> float:
            s = str(size_str).strip()
            if "TB" in s:
                return float(s.replace("TB", "")) * 1000.0
            if "GB" in s:
                return float(s.replace("GB", ""))
            if "MB" in s:
                return float(s.replace("MB", "")) / 1000.0
            return 0.0

        fw = fw.dropna(subset=["Name", "Disk size"])  # basic hygiene
        fw["size_gb"] = fw["Disk size"].apply(parse_size)
        fw = fw[fw["ISO 639-3 code"] != "Total"]
        fw_sorted = fw.sort_values("size_gb", ascending=False)
        for idx, (_, row) in enumerate(fw_sorted.iterrows()):
            order_map[str(row["Name"])] = idx

    # Programming languages by CSV row order
    if starcoder_csv.exists():
        sc = pd.read_csv(starcoder_csv)
        prog_names: List[str] = [
            f"{str(x).strip().title()} (code)" for x in sc["Language"].tolist()
        ]
        for idx, name in enumerate(prog_names):
            order_map[name] = idx

    # English is not added here; handled in caller
    return order_map


def get_tokenizer_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """Extract tokenizer-level summary information including vocab metrics."""
    summary_rows = []

    for tokenizer_key, result in results.items():
        tokenizer_name = result.get("tokenizer", tokenizer_key)
        vocab_size = result.get("vocab_size", None)
        timestamp = result.get("timestamp", 0)

        # Extract vocab metrics
        vocab_metrics = result.get("vocab_metrics", {})
        global_metrics = result.get("global_metrics", {})

        row = {
            "tokenizer_key": tokenizer_key,
            "tokenizer_name": tokenizer_name,
            "vocab_size": vocab_size,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp) if timestamp else None,
        }

        # Add vocab metrics
        if vocab_metrics:
            row.update(
                {
                    "tokens_without_leading_space_pct": vocab_metrics.get(
                        "tokens_without_leading_space_pct"
                    ),
                    "tokens_without_leading_space_count": vocab_metrics.get(
                        "tokens_without_leading_space_count"
                    ),
                    "analyzed_sample_size": vocab_metrics.get("analyzed_sample_size"),
                    "sample_non_space_tokens": vocab_metrics.get(
                        "sample_non_space_tokens", []
                    ),
                    "sample_space_tokens": vocab_metrics.get("sample_space_tokens", []),
                }
            )

        # Add global metrics
        if global_metrics:
            row.update(
                {
                    "total_tokens_analyzed": global_metrics.get(
                        "total_tokens_analyzed"
                    ),
                    "tokens_starting_with_space_pct": global_metrics.get(
                        "tokens_starting_with_space_pct"
                    ),
                    "tokens_with_whitespace_in_middle_pct": global_metrics.get(
                        "tokens_with_whitespace_in_middle_pct"
                    ),
                    "tokens_with_script_overlap_pct": global_metrics.get(
                        "tokens_with_script_overlap_pct"
                    ),
                    "tokens_with_symbols_unicode_pct": global_metrics.get(
                        "tokens_with_symbols_unicode_pct"
                    ),
                    "tokens_with_latin_unicode_pct": global_metrics.get(
                        "tokens_with_latin_unicode_pct"
                    ),
                    "tokens_with_punctuation_unicode_pct": global_metrics.get(
                        "tokens_with_punctuation_unicode_pct"
                    ),
                    "tokens_with_numbers_unicode_pct": global_metrics.get(
                        "tokens_with_numbers_unicode_pct"
                    ),
                    "tokens_with_japanese_unicode_pct": global_metrics.get(
                        "tokens_with_japanese_unicode_pct"
                    ),
                    "tokens_with_chinese_unicode_pct": global_metrics.get(
                        "tokens_with_chinese_unicode_pct"
                    ),
                    "tokens_with_cyrillic_unicode_pct": global_metrics.get(
                        "tokens_with_cyrillic_unicode_pct"
                    ),
                    "tokens_with_greek_unicode_pct": global_metrics.get(
                        "tokens_with_greek_unicode_pct"
                    ),
                    "tokens_with_korean_unicode_pct": global_metrics.get(
                        "tokens_with_korean_unicode_pct"
                    ),
                    "tokens_with_arabic_unicode_pct": global_metrics.get(
                        "tokens_with_arabic_unicode_pct"
                    ),
                    "tokens_with_hebrew_unicode_pct": global_metrics.get(
                        "tokens_with_hebrew_unicode_pct"
                    ),
                }
            )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
