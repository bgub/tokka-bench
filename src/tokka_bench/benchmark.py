"""
Tokka-Bench: Tokenizer benchmarking for multiple languages.

This module provides the core functionality for benchmarking HuggingFace tokenizers
across multiple natural languages (FineWeb-2), programming languages (StarCoder),
and English (FineWeb).

Key Features:
- Universal tokenizer support for any HuggingFace model
- Real-world data from multiple sources:
  - Natural languages: FineWeb-2 (top languages by size)
  - Programming languages: StarCoder dataset
  - English: FineWeb sample-10BT
- Efficiency metrics (bytes per token, unique tokens)
- JSON output with detailed results
- Parallel processing for faster benchmarks
"""

import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # Get vocabulary size
        self.vocab_size = len(self.tokenizer)

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
    """Load natural language data from CSV file."""
    # Get the path to the CSV file in the src directory
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "..", "fineweb-2-languages.csv")

    df = pd.read_csv(csv_path)
    # Clean up the column names and data
    df.columns = df.columns.str.strip()
    return df


def load_coding_languages(n: int = 10) -> List[Dict[str, str]]:
    """Load coding language data from CSV file."""
    # Get the path to the CSV file in the src directory
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "..", "starcoderdata-dirs.csv")

    df = pd.read_csv(csv_path)

    # Convert to list of language info dictionaries (first n languages only)
    coding_langs = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= n:  # Stop after n languages
            break
        lang = row["Language"].strip()
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

    target_bytes = int(sample_size_mb * 1024 * 1024)
    source = language_info.get("source", "fineweb2")

    print(f"    Loading real data from {source}...")

    try:
        # Load dataset based on source
        if source == "fineweb2":
            # FineWeb-2 dataset
            dataset_name = f"{language_info['iso_code']}_{language_info['script']}"
            fw = load_dataset(
                "HuggingFaceFW/fineweb-2",
                name=dataset_name,
                split="train",
                streaming=True,
            )
            content_key = "text"

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
            data_dir = language_info.get("data_dir", language_info["iso_code"])
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
        accumulated_text = []
        total_bytes = 0

        # Use iterator to ensure we can clean up properly
        dataset_iter = iter(fw)

        try:
            while total_bytes < target_bytes:
                sample = next(dataset_iter)
                text = sample.get(content_key, "")
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


def benchmark_language(
    tokenizer: UniversalTokenizer,
    lang_info: Dict[str, str],
    sample_size_mb: float = 1.0,
) -> Dict[str, Any]:
    """Benchmark a single language (for parallel processing)."""
    # Load real sample text
    text = load_real_sample_text(lang_info, sample_size_mb)

    # Calculate metrics
    metrics = tokenizer.get_metrics(text)

    # Return results for this language
    if lang_info.get("source") == "starcoder":
        lang_key = f"{lang_info['iso_code']}-code"
    elif lang_info.get("source") == "fineweb":
        lang_key = f"{lang_info['iso_code']}-fineweb"
    else:
        lang_key = f"{lang_info['iso_code']}-{lang_info['script']}"

    return {
        "lang_key": lang_key,
        "language_info": lang_info,
        "metrics": metrics,
    }


def benchmark_tokenizer(
    tokenizer: UniversalTokenizer,
    languages: List[Dict[str, str]],
    sample_size_mb: float = 1.0,
) -> Dict[str, Any]:
    """Benchmark a tokenizer on multiple languages using parallel processing."""

    print(f"Benchmarking {tokenizer.name} on {len(languages)} languages in parallel...")

    results = {
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "benchmark_size_mb": sample_size_mb,
        "timestamp": time.time(),
        "languages": {},
    }

    # Determine optimal number of workers (don't exceed number of languages or CPU cores)
    import os

    max_workers = min(
        len(languages), os.cpu_count() or 4, 10
    )  # Cap at 10 for good parallelism without overwhelming

    print(f"Using {max_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all language tasks
        future_to_lang = {
            executor.submit(
                benchmark_language, tokenizer, lang_info, sample_size_mb
            ): lang_info
            for lang_info in languages
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_lang):
            lang_info = future_to_lang[future]
            try:
                result = future.result()

                # Store results
                results["languages"][result["lang_key"]] = {
                    "language_info": result["language_info"],
                    "metrics": result["metrics"],
                }

                # Print progress
                print(
                    f"  ✓ {lang_info['name']:<25} | "
                    f"Bytes/token: {result['metrics']['bytes_per_token']:.2f} | "
                    f"Unique tokens: {result['metrics']['unique_tokens']:,d}"
                )

            except Exception as e:
                print(f"  ✗ Error processing {lang_info['name']}: {e}")

    print(
        f"Completed benchmarking {len(results['languages'])}/{len(languages)} languages"
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
    # Load all language sources
    print("Loading language data from multiple sources...")

    # 1. English from FineWeb (first)
    english_fineweb = get_english_fineweb()

    # 2. Top natural languages from FineWeb-2 (in CSV order)
    df = load_language_data()
    natural_languages = get_top_languages(df, n=30)  # Top 30 natural languages

    # 3. Programming languages from StarCoder (in CSV order, last)
    coding_languages = load_coding_languages()

    # Combine all languages in proper order: English → Natural → Programming
    all_languages = [english_fineweb] + natural_languages + coding_languages

    print(f"\n📊 Benchmarking {len(all_languages)} languages total:")
    print("  • 1 English (FineWeb sample-10BT)")
    print(f"  • {len(natural_languages)} natural languages (FineWeb-2)")
    print(f"  • {len(coding_languages)} programming languages (StarCoder - top 10)")
    print()

    print("🇺🇸 English (FineWeb) - First:")
    print(
        f"   1. {english_fineweb['name']:<25} ({english_fineweb['iso_code']}-{english_fineweb['script']}): sample-10BT"
    )

    print(
        f"\n🌍 Natural Languages (FineWeb-2) - In CSV order - showing first 10 of {len(natural_languages)}:"
    )
    for i, lang in enumerate(natural_languages[:10], 1):
        # Get size from CSV for display
        lang_size = df.loc[df["ISO 639-3 code"] == lang["iso_code"], "Disk size"].iloc[
            0
        ]
        print(
            f"  {i:2d}. {lang['name']:<25} ({lang['iso_code']}-{lang['script']}): {lang_size}"
        )
    if len(natural_languages) > 10:
        print(f"      ... and {len(natural_languages) - 10} more natural languages")

    print(
        f"\n💻 Programming Languages (StarCoder) - In CSV order, last - {len(coding_languages)} selected:"
    )
    for i, lang in enumerate(coding_languages, 1):
        print(f"  {i:2d}. {lang['name']:<25} ({lang['iso_code']}-{lang['script']})")
    print()

    # Initialize tokenizer
    tokenizer = UniversalTokenizer(tokenizer_name)

    # Run benchmark
    results = benchmark_tokenizer(tokenizer, all_languages, sample_size_mb)

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
    del all_languages
    del df
    gc.collect()

    print(f"Results saved to {output_path}")
    print("--------------------------------------------------")
    print("✅ Benchmark completed successfully!")
    print(f"Results saved for {len(results['languages'])} languages")

    return results
