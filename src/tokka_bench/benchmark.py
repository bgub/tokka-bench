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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
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

    # Calculate metrics with language information
    metrics: Dict[str, Any] = tokenizer.get_metrics(sample_text, lang_info)

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


def process_language_parallel(
    tokenizers: List[UniversalTokenizer],
    lang_info: Dict[str, str],
    sample_size_mb: float,
    global_trackers: Dict[str, GlobalMetricsTracker],
    lock: Lock,
) -> Dict[str, Any]:
    """Process a single language with all tokenizers (thread-safe)."""
    try:
        # Load real sample text ONCE for all tokenizers
        sample_text: str = load_real_sample_text(lang_info, sample_size_mb)

        # Evaluate all tokenizers on the same text
        results = {}
        metrics_summary = []

        for tokenizer in tokenizers:
            # Calculate metrics with language information
            metrics: Dict[str, Any] = tokenizer.get_metrics(sample_text, lang_info)

            # Store results for this tokenizer
            results[tokenizer.name] = {
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

            # Collect metrics for summary
            metrics_summary.append(
                f"{tokenizer.name.split('/')[-1]}: {metrics['bytes_per_token']:.2f}"
            )

        # Thread-safe progress reporting
        with lock:
            print(
                f"  ✓ {lang_info['name']:<25} | Bytes/token: {' | '.join(metrics_summary)}"
            )

        # Get detailed token analysis for global metrics for each tokenizer
        for tokenizer in tokenizers:
            token_analysis: Dict[str, Any] = tokenizer.get_token_analysis(sample_text)

            # Thread-safe global metrics update
            with lock:
                global_trackers[tokenizer.name].add_tokens(token_analysis["tokens"])

        return {
            "success": True,
            "lang_info": lang_info,
            "results": results,
        }

    except Exception as e:
        # Thread-safe error reporting
        with lock:
            print(f"  ❌ {lang_info['name']:<25} | Error: {e}")

        return {
            "success": False,
            "lang_info": lang_info,
            "error": str(e),
        }


def benchmark_multiple_tokenizers(
    tokenizers: List[UniversalTokenizer],
    languages: List[Dict[str, str]],
    sample_size_mb: float = 1.0,
    max_workers: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """Benchmark multiple tokenizers across multiple languages with parallel processing."""
    # Initialize global trackers for each tokenizer (thread-safe)
    global_trackers = {
        tokenizer.name: GlobalMetricsTracker() for tokenizer in tokenizers
    }

    # Initialize results structure for each tokenizer
    all_results = {}
    for tokenizer in tokenizers:
        all_results[tokenizer.name] = {
            "tokenizer": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "vocab_metrics": tokenizer.vocab_metrics,
            "benchmark_size_mb": sample_size_mb,
            "timestamp": time.time(),
            "languages": {},
            "global_metrics": {},
        }

    # Thread-safe lock for shared resources
    lock = Lock()

    # Process languages in parallel
    print(
        f"🚀 Processing {len(languages)} languages with {max_workers} parallel workers..."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all language processing tasks
        future_to_lang = {
            executor.submit(
                process_language_parallel,
                tokenizers,
                lang_info,
                sample_size_mb,
                global_trackers,
                lock,
            ): lang_info
            for lang_info in languages
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_lang):
            lang_info = future_to_lang[future]
            result = future.result()
            completed += 1

            if result["success"]:
                # Store results for each tokenizer
                for tokenizer_name, lang_result in result["results"].items():
                    # Store results using a unique key
                    if lang_info["source"] == "starcoder":
                        key: str = f"{lang_info['iso_code']}-code"
                    else:
                        key = f"{lang_info['iso_code']}-{lang_info['script']}"

                    all_results[tokenizer_name]["languages"][key] = lang_result

            # Progress update
            if completed % 10 == 0 or completed == len(languages):
                print(f"📊 Progress: {completed}/{len(languages)} languages completed")

    # Calculate global metrics for each tokenizer
    print("\n🔄 Calculating global metrics...")
    for tokenizer_name, tracker in global_trackers.items():
        all_results[tokenizer_name]["global_metrics"] = tracker.get_global_metrics()

    return all_results


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
    print("\n🔄 Calculating global metrics...")
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
    tokenizer_names: List[str],
    output_names: Optional[List[str]] = None,
    sample_size_mb: float = 1.0,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Run a complete benchmark for one or more tokenizers."""
    if len(tokenizer_names) == 1:
        print(f"🚀 Starting benchmark for {tokenizer_names[0]}")
    else:
        print(f"🚀 Starting benchmark for {len(tokenizer_names)} tokenizers:")
        for i, name in enumerate(tokenizer_names, 1):
            print(f"  {i}. {name}")

    print(f"📏 Sample size: {sample_size_mb}MB per language")
    if len(tokenizer_names) > 1:
        print(f"⚡ Parallel workers: {max_workers}")

    # Load tokenizers
    print("\n🔧 Loading tokenizers...")
    tokenizers: List[UniversalTokenizer] = []
    for name in tokenizer_names:
        print(f"  Loading {name}...")
        tokenizer = UniversalTokenizer(name)
        tokenizers.append(tokenizer)

    if len(tokenizers) > 1:
        print(f"✅ Loaded {len(tokenizers)} tokenizers successfully")

    # 1. English from FineWeb sample-10BT
    english_lang: Dict[str, str] = get_english_fineweb()

    # 2. Top natural languages from FineWeb-2 (in CSV order)
    df: pd.DataFrame = load_language_data()
    natural_languages: List[Dict[str, str]] = get_top_languages(
        df, n=99
    )  # Top 99 natural languages

    # 3. Programming languages from StarCoder (in CSV order, last)
    coding_languages: List[Dict[str, str]] = load_coding_languages(
        n=20
    )  # Top 20 programming languages

    # All languages to benchmark
    all_languages: List[Dict[str, str]] = (
        [english_lang] + natural_languages + coding_languages
    )

    print(f"\n📊 Benchmarking {len(all_languages)} languages total:")
    print("  • 1 English (FineWeb sample-10BT)")
    print(f"  • {len(natural_languages)} natural languages (FineWeb-2 - top 99)")
    print(f"  • {len(coding_languages)} programming languages (StarCoder - top 20)")
    print()

    # Print language lists
    print("\n🌍 Natural Languages (FineWeb-2) - In CSV order:")
    for i, lang in enumerate(natural_languages, 1):
        # Get size from CSV for display
        lang_size = df.loc[df["ISO 639-3 code"] == lang["iso_code"], "Disk size"].iloc[
            0
        ]
        print(
            f"  {i:2d}. {lang['name']:<25} ({lang['iso_code']}-{lang['script']}): {lang_size}"
        )

    print("\n💻 Programming Languages (StarCoder) - In CSV order:")
    for i, lang in enumerate(coding_languages, 1):
        print(f"  {i:2d}. {lang['name']:<25} ({lang['iso_code']}-{lang['script']})")

    # Run benchmark
    print("\n🏃 Running benchmark...")
    if len(tokenizers) == 1:
        # Single tokenizer - use existing function
        results = benchmark_tokenizer(tokenizers[0], all_languages, sample_size_mb)
        all_results = {tokenizers[0].name: results}
    else:
        # Multiple tokenizers - use optimized parallel function
        print(f"🔄 Optimized multi-tokenizer mode with {max_workers} parallel workers")
        all_results = benchmark_multiple_tokenizers(
            tokenizers, all_languages, sample_size_mb, max_workers
        )

    # Save results for each tokenizer
    saved_files = []
    for i, tokenizer_name in enumerate(tokenizer_names):
        results = all_results[tokenizer_name]

        # Determine output filename
        if output_names and i < len(output_names):
            output_filename: str = f"{output_names[i]}.json"
        else:
            # Convert tokenizer name to safe filename
            safe_name: str = tokenizer_name.replace("/", "_").replace("-", "_")
            output_filename = f"{safe_name}.json"

        output_path: str = os.path.join("data", "results", output_filename)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save results
        save_results(results, output_path)
        saved_files.append(output_path)
        print(f"Results saved to {output_path}")

    # Print summary
    print(f"\nCompleted benchmarking {len(all_languages)} languages")
    for tokenizer_name in tokenizer_names:
        results = all_results[tokenizer_name]
        print(
            f"  {tokenizer_name}: {results['global_metrics'].get('total_tokens_analyzed', 0):,} tokens analyzed"
        )

    # Clean up
    try:
        gc.collect()
        print("✅ Benchmark completed successfully!")
        print(
            f"Results saved for {len(tokenizer_names)} tokenizer(s) across {len(all_languages)} languages"
        )
        if len(saved_files) > 1:
            print(f"📁 Files saved: {', '.join(saved_files)}")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")

    return all_results
