"""
Data loading and processing utilities for visualization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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
            if "continued_word_rate" in metrics:
                row["continued_word_rate"] = metrics["continued_word_rate"]

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

            rows.append(row)

    return pd.DataFrame(rows)


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
