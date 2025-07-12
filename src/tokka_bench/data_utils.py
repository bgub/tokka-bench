"""
Data loading utilities for tokenizer benchmarking.

This module provides functions for loading language metadata and text samples
from various datasets (FineWeb-2, StarCoder, FineWeb).
"""

import gc
import os
from typing import Dict, List, Optional

import pandas as pd


def load_language_data() -> pd.DataFrame:
    """Load natural language data from CSV file."""
    # Get the path to the CSV file in the src directory
    current_dir: str = os.path.dirname(__file__)
    csv_path: str = os.path.join(current_dir, "..", "fineweb-2-languages.csv")

    df: pd.DataFrame = pd.read_csv(csv_path)
    # Clean up the column names and data
    df.columns = df.columns.str.strip()
    return df


def load_coding_languages(n: int = 10) -> List[Dict[str, str]]:
    """Load coding language data from CSV file."""
    # Get the path to the CSV file in the src directory
    current_dir: str = os.path.dirname(__file__)
    csv_path: str = os.path.join(current_dir, "..", "starcoderdata-dirs.csv")

    df: pd.DataFrame = pd.read_csv(csv_path)

    # Convert to list of language info dictionaries (first n languages only)
    coding_langs: List[Dict[str, str]] = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= n:  # Stop after n languages
            break
        lang: str = row["Language"].strip()
        coding_langs.append(
            {
                "iso_code": lang,  # Use language name as identifier
                "script": "code",  # Mark as coding language
                "name": f"{lang.title()} (code)",
                "source": "starcoder",
                "data_dir": lang,
            }
        )

    return coding_langs


def get_top_languages(df: pd.DataFrame, n: int = 5) -> List[Dict[str, str]]:
    """Get the top N natural languages by size."""
    # Filter out invalid rows (like Total row)
    df = df.dropna(subset=["Name", "Script"])
    df = df[df["ISO 639-3 code"] != "Total"]

    # Convert disk size to numeric for sorting
    def parse_size(size_str: str) -> float:
        size_str = str(size_str).strip()
        if "TB" in size_str:
            return float(size_str.replace("TB", "")) * 1000
        elif "GB" in size_str:
            return float(size_str.replace("GB", ""))
        elif "MB" in size_str:
            return float(size_str.replace("MB", "")) / 1000
        return 0.0

    df["size_gb"] = df["Disk size"].apply(parse_size)
    top_langs: pd.DataFrame = df.nlargest(n, "size_gb")

    return [
        {
            "iso_code": row["ISO 639-3 code"],
            "script": row["Script"],
            "name": row["Name"],
            "source": "fineweb2",
        }
        for _, row in top_langs.iterrows()
    ]


def get_english_fineweb() -> Dict[str, str]:
    """Get English from FineWeb sample-10BT."""
    return {
        "iso_code": "eng",
        "script": "Latn",
        "name": "English (FineWeb)",
        "source": "fineweb",
    }


def load_real_sample_text(
    language_info: Dict[str, str], sample_size_mb: float = 1.0
) -> str:
    """Load real sample text from appropriate dataset based on source."""
    from datasets import load_dataset

    target_bytes: int = int(sample_size_mb * 1024 * 1024)
    source: str = language_info.get("source", "fineweb2")

    print(f"    Loading real data from {source}...")

    try:
        # Load dataset based on source
        if source == "fineweb2":
            # FineWeb-2 dataset
            dataset_name: str = f"{language_info['iso_code']}_{language_info['script']}"
            fw = load_dataset(
                "HuggingFaceFW/fineweb-2",
                name=dataset_name,
                split="train",
                streaming=True,
            )
            content_key: str = "text"

        elif source == "fineweb":
            # FineWeb English dataset
            fw = load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                split="train",
                streaming=True,
            )
            content_key = "text"

        elif source == "starcoder":
            # StarCoder dataset
            data_dir: str = language_info.get("data_dir", language_info["iso_code"])
            fw = load_dataset(
                "bigcode/starcoderdata",
                data_dir=data_dir,
                split="train",
                streaming=True,
            )
            content_key = "content"

        else:
            raise ValueError(f"Unknown source: {source}")

        # Accumulate text until we reach target size
        accumulated_text: List[str] = []
        total_bytes: int = 0

        # Use iterator to ensure we can clean up properly
        dataset_iter = iter(fw)

        try:
            while total_bytes < target_bytes:
                sample = next(dataset_iter)
                text: str = sample.get(content_key, "")
                if text:
                    accumulated_text.append(text)
                    total_bytes += len(text.encode("utf-8"))
        except StopIteration:
            # End of dataset reached
            pass

        # Join all accumulated text
        full_text: str = "\n".join(accumulated_text)

        # Truncate to exact size if needed
        text_bytes: bytes = full_text.encode("utf-8")
        if len(text_bytes) > target_bytes:
            full_text = text_bytes[:target_bytes].decode("utf-8", errors="ignore")

        print(f"    Loaded {len(full_text.encode('utf-8')):,} bytes of real data")

        # Simple cleanup
        del fw
        del dataset_iter
        gc.collect()

        return full_text

    except Exception as e:
        print(f"    Warning: Could not load real data ({e}), using fallback text")
        # Fallback to a simple sample if dataset loading fails
        fallback_text: str = (
            f"Sample text for {language_info['name']} tokenizer testing. " * 1000
        )

        # Adjust size to target
        text_bytes = fallback_text.encode("utf-8")
        if len(text_bytes) > target_bytes:
            fallback_text = text_bytes[:target_bytes].decode("utf-8", errors="ignore")
        elif len(text_bytes) < target_bytes:
            repeat_count: int = (target_bytes // len(text_bytes)) + 1
            fallback_text = (fallback_text * repeat_count)[:target_bytes]

        return fallback_text
