"""
Tokka-Bench: Tokenizer benchmarking for multiple languages.

This module provides the core functionality for benchmarking HuggingFace tokenizers
across multiple languages using real data from the FineWeb-2 dataset.

Key Features:
- Universal tokenizer support for any HuggingFace model
- Real-world data from top 5 languages by FineWeb-2 size
- Efficiency metrics (bytes per token)
- JSON output with detailed results
"""

import gc
import json
import os
import time
from typing import Any, Dict, List

import pandas as pd
from transformers import AutoTokenizer


class UniversalTokenizer:
    """Universal tokenizer that can load any HuggingFace model."""

    def __init__(self, model_name: str):
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_metrics(self, text: str) -> Dict[str, float]:
        """Calculate metrics for given text."""
        # Encode the text
        token_ids = self.encode(text)

        # Basic metrics from the actual text we're tokenizing
        text_bytes = len(text.encode("utf-8"))
        num_tokens = len(token_ids)
        unique_tokens = len(set(token_ids))  # Count unique token IDs

        return {
            "bytes_per_token": text_bytes / num_tokens if num_tokens > 0 else 0,
            "total_bytes": text_bytes,
            "total_tokens": num_tokens,
            "unique_tokens": unique_tokens,
        }


def load_language_data() -> pd.DataFrame:
    """Load language data from CSV file."""
    # Get the path to the CSV file in the src directory
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "..", "fineweb-2-languages.csv")

    df = pd.read_csv(csv_path)
    # Clean up the column names and data
    df.columns = df.columns.str.strip()
    return df


def get_top_languages(df: pd.DataFrame, n: int = 5) -> List[Dict[str, str]]:
    """Get the top N languages by size."""
    # Filter out invalid rows (like Total row)
    df = df.dropna(subset=["Name", "Script"])
    df = df[df["ISO 639-3 code"] != "Total"]

    # Convert disk size to numeric for sorting
    def parse_size(size_str):
        size_str = str(size_str).strip()
        if "TB" in size_str:
            return float(size_str.replace("TB", "")) * 1000
        elif "GB" in size_str:
            return float(size_str.replace("GB", ""))
        elif "MB" in size_str:
            return float(size_str.replace("MB", "")) / 1000
        return 0

    df["size_gb"] = df["Disk size"].apply(parse_size)
    top_langs = df.nlargest(n, "size_gb")

    return [
        {
            "iso_code": row["ISO 639-3 code"],
            "script": row["Script"],
            "name": row["Name"],
        }
        for _, row in top_langs.iterrows()
    ]


def load_real_sample_text(
    language_info: Dict[str, str], sample_size_mb: float = 1.0
) -> str:
    """Load real sample text from FineWeb-2 dataset."""
    from datasets import load_dataset

    # Construct the dataset name for FineWeb-2
    dataset_name = f"{language_info['iso_code']}_{language_info['script']}"
    target_bytes = int(sample_size_mb * 1024 * 1024)

    print("    Loading real data from FineWeb-2...")

    try:
        # Load the dataset in streaming mode
        fw = load_dataset(
            "HuggingFaceFW/fineweb-2", name=dataset_name, split="train", streaming=True
        )

        # Accumulate text until we reach target size
        accumulated_text = []
        total_bytes = 0

        # Use iterator to ensure we can clean up properly
        dataset_iter = iter(fw)

        try:
            while total_bytes < target_bytes:
                sample = next(dataset_iter)
                text = sample.get("text", "")
                if text:
                    accumulated_text.append(text)
                    total_bytes += len(text.encode("utf-8"))
        except StopIteration:
            # End of dataset reached
            pass

        # Join all accumulated text
        full_text = "\n".join(accumulated_text)

        # Truncate to exact size if needed
        text_bytes = full_text.encode("utf-8")
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
        fallback_text = (
            f"Sample text for {language_info['name']} tokenizer testing. " * 1000
        )

        # Adjust size to target
        text_bytes = fallback_text.encode("utf-8")
        if len(text_bytes) > target_bytes:
            fallback_text = text_bytes[:target_bytes].decode("utf-8", errors="ignore")
        elif len(text_bytes) < target_bytes:
            repeat_count = (target_bytes // len(text_bytes)) + 1
            fallback_text = (fallback_text * repeat_count)[:target_bytes]

        return fallback_text


def benchmark_tokenizer(
    tokenizer: UniversalTokenizer,
    languages: List[Dict[str, str]],
    sample_size_mb: float = 1.0,
) -> Dict[str, Any]:
    """Benchmark a tokenizer on multiple languages."""

    print(f"Benchmarking {tokenizer.name}...")

    results = {
        "tokenizer": tokenizer.name,
        "benchmark_size_mb": sample_size_mb,
        "timestamp": time.time(),
        "languages": {},
    }

    for lang_info in languages:
        print(
            f"  Processing {lang_info['name']} ({lang_info['iso_code']}-{lang_info['script']})..."
        )

        # Load real sample text
        text = load_real_sample_text(lang_info, sample_size_mb)

        # Calculate metrics
        metrics = tokenizer.get_metrics(text)

        # Store results
        lang_key = f"{lang_info['iso_code']}-{lang_info['script']}"
        results["languages"][lang_key] = {
            "language_info": lang_info,
            "metrics": metrics,
        }

        print(
            f"    Bytes/token: {metrics['bytes_per_token']:.2f} | Unique tokens: {metrics['unique_tokens']:,d}"
        )

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def run_benchmark(
    tokenizer_name: str, output_name: str = None, sample_size_mb: float = 1.0
):
    """Run the complete benchmark process."""
    # Load language data
    df = load_language_data()
    languages = get_top_languages(df, n=30)  # Expand to top 30 languages

    print("Top 30 languages by size:")
    for i, lang in enumerate(languages, 1):
        # Get size from CSV for display
        lang_size = df.loc[df["ISO 639-3 code"] == lang["iso_code"], "Disk size"].iloc[
            0
        ]
        print(
            f"  {i:2d}. {lang['name']:<25} ({lang['iso_code']}-{lang['script']}): {lang_size}"
        )
    print()

    # Initialize tokenizer
    tokenizer = UniversalTokenizer(tokenizer_name)

    # Run benchmark
    results = benchmark_tokenizer(tokenizer, languages, sample_size_mb)

    # Generate output path
    if output_name:
        filename = f"{output_name}.json"
    else:
        # Convert tokenizer name to safe filename
        safe_name = tokenizer_name.replace("/", "_").replace("-", "_")
        filename = f"{safe_name}.json"

    output_path = f"data/results/{filename}"

    # Ensure output directory exists
    os.makedirs("data/results", exist_ok=True)

    # Save results
    save_results(results, output_path)

    # Aggressive cleanup
    del tokenizer
    del languages
    del df
    gc.collect()

    print(f"Results saved to {output_path}")
    print("--------------------------------------------------")
    print("✅ Benchmark completed successfully!")
    print(f"Results saved for {len(results['languages'])} languages")

    return results
