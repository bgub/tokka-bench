"""
Benchmarking utility for tokenizer performance across languages.

This module provides the main benchmarking functionality for evaluating
tokenizer performance across different languages and scripts.
"""

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_utils import (
    get_english_fineweb,
    get_top_languages,
    load_coding_languages,
    load_language_data,
    load_real_sample_text,
)
from .metrics import GlobalMetricsTracker
from .tokenizer import UniversalTokenizer


def benchmark_language(
    tokenizer: UniversalTokenizer,
    lang_info: Dict[str, str],
    sample_size_mb: float = 1.0,
) -> Dict[str, Any]:
    """Benchmark a single language with the given tokenizer."""
    print(f"  ✓ {lang_info['name']:<25} | ", end="", flush=True)

    # Load real sample text
    sample_text: str = load_real_sample_text(lang_info, sample_size_mb)

    # Calculate metrics
    metrics: Dict[str, Any] = tokenizer.get_metrics(sample_text)

    # Print summary
    print(
        f"Bytes/token: {metrics['bytes_per_token']:.2f} | "
        f"Unique tokens: {metrics['unique_tokens']:,} | "
        f"Fertility: {metrics['subword_fertility']:.2f} | "
        f"Split rate: {metrics['continued_word_rate']:.1f}%"
    )

    # Return results
    return {
        "language_info": lang_info,
        "metrics": {
            "bytes_per_token": metrics["bytes_per_token"],
            "total_bytes": metrics["total_bytes"],
            "total_tokens": metrics["total_tokens"],
            "unique_tokens": metrics["unique_tokens"],
            "subword_fertility": metrics["subword_fertility"],
            "continued_word_rate": metrics["continued_word_rate"],
        },
    }


def benchmark_tokenizer(
    tokenizer: UniversalTokenizer,
    languages: List[Dict[str, str]],
    sample_size_mb: float = 1.0,
) -> Dict[str, Any]:
    """Benchmark a tokenizer across multiple languages."""
    global_tracker: GlobalMetricsTracker = GlobalMetricsTracker()
    language_results: Dict[str, Dict[str, Any]] = {}

    # Benchmark each language
    for lang_info in languages:
        try:
            # Get language results
            result: Dict[str, Any] = benchmark_language(
                tokenizer, lang_info, sample_size_mb
            )

            # Store results using a unique key
            if lang_info["source"] == "starcoder":
                key: str = f"{lang_info['iso_code']}-code"
            else:
                key = f"{lang_info['iso_code']}-{lang_info['script']}"

            language_results[key] = result

            # Load sample text again for global analysis
            sample_text = load_real_sample_text(lang_info, sample_size_mb)

            # Get detailed token analysis for global metrics
            token_analysis: Dict[str, Any] = tokenizer.get_token_analysis(sample_text)
            global_tracker.add_tokens(token_analysis["tokens"])

        except Exception as e:
            print(f"Error benchmarking {lang_info['name']}: {e}")
            continue

    # Calculate global metrics
    print(f"\n🔄 Calculating global metrics...")
    global_metrics: Dict[str, float] = global_tracker.get_global_metrics()

    return {
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "vocab_metrics": tokenizer.vocab_metrics,
        "benchmark_size_mb": sample_size_mb,
        "timestamp": time.time(),
        "languages": language_results,
        "global_metrics": global_metrics,
    }


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save benchmark results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_benchmark(
    tokenizer_name: str, output_name: Optional[str] = None, sample_size_mb: float = 1.0
) -> None:
    """Run a complete benchmark for a tokenizer."""
    print(f"🚀 Starting benchmark for {tokenizer_name}")
    print(f"📏 Sample size: {sample_size_mb}MB per language")

    # Load tokenizer
    print(f"\n🔧 Loading tokenizer...")
    tokenizer: UniversalTokenizer = UniversalTokenizer(tokenizer_name)

    # 1. English from FineWeb sample-10BT
    english_lang: Dict[str, str] = get_english_fineweb()

    # 2. Top natural languages from FineWeb-2 (in CSV order)
    df: pd.DataFrame = load_language_data()
    natural_languages: List[Dict[str, str]] = get_top_languages(
        df, n=20
    )  # Top 20 natural languages

    # 3. Programming languages from StarCoder (in CSV order, last)
    coding_languages: List[Dict[str, str]] = load_coding_languages(
        n=5
    )  # Top 5 programming languages

    # All languages to benchmark
    all_languages: List[Dict[str, str]] = (
        [english_lang] + natural_languages + coding_languages
    )

    print(f"\n📊 Benchmarking {len(all_languages)} languages total:")
    print("  • 1 English (FineWeb sample-10BT)")
    print(f"  • {len(natural_languages)} natural languages (FineWeb-2 - top 20)")
    print(f"  • {len(coding_languages)} programming languages (StarCoder - top 5)")
    print()

    # Print language lists
    print(f"\n🌍 Natural Languages (FineWeb-2) - In CSV order:")
    for i, lang in enumerate(natural_languages, 1):
        # Get size from CSV for display
        lang_size = df.loc[df["ISO 639-3 code"] == lang["iso_code"], "Disk size"].iloc[
            0
        ]
        print(
            f"  {i:2d}. {lang['name']:<25} ({lang['iso_code']}-{lang['script']}): {lang_size}"
        )

    print(f"\n💻 Programming Languages (StarCoder) - In CSV order:")
    for i, lang in enumerate(coding_languages, 1):
        print(f"  {i:2d}. {lang['name']:<25} ({lang['iso_code']}-{lang['script']})")

    # Run benchmark
    print(f"\n🏃 Running benchmark...")
    results: Dict[str, Any] = benchmark_tokenizer(
        tokenizer, all_languages, sample_size_mb
    )

    # Save results
    print(f"Completed benchmarking {len(all_languages)}/{len(all_languages)} languages")
    print(
        f"Analyzed {results['global_metrics'].get('total_tokens_analyzed', 0):,} tokens globally"
    )

    # Determine output filename
    if output_name:
        output_filename: str = f"{output_name}.json"
    else:
        # Convert tokenizer name to safe filename
        safe_name: str = tokenizer_name.replace("/", "_").replace("-", "_")
        output_filename = f"{safe_name}.json"

    output_path: str = os.path.join("data", "results", output_filename)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results
    save_results(results, output_path)
    print(f"Results saved to {output_path}")

    # Clean up
    try:
        gc.collect()
        print("✅ Benchmark completed successfully!")
        print(f"Results saved for {len(all_languages)} languages")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")
    finally:
        sys.exit(0)
