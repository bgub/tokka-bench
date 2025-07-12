"""
Data loading and processing utilities for visualization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from tokka_bench.visualization.constants import RESULTS_DIR


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

        for lang_key, lang_data in result.get("languages", {}).items():
            lang_info = lang_data["language_info"]
            metrics = lang_data["metrics"]

            rows.append(
                {
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
                    "datetime": datetime.fromtimestamp(timestamp)
                    if timestamp
                    else None,
                }
            )

    return pd.DataFrame(rows)
