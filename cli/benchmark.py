#!/usr/bin/env python3
"""CLI for running tokka-bench."""

import sys

from omegaconf import OmegaConf

from tokka_bench.benchmark import run_benchmark


def print_summary(results):
    """Print a summary of benchmark results."""
    print("📊 Summary (ranked by efficiency):")

    # Sort languages by bytes_per_token (descending - higher is better)
    lang_results = []
    for lang_key, lang_data in results["languages"].items():
        lang_name = lang_data["language_info"]["name"]
        bytes_per_token = lang_data["metrics"]["bytes_per_token"]
        unique_tokens = lang_data["metrics"]["unique_tokens"]
        lang_results.append((lang_name, bytes_per_token, unique_tokens))

    # Sort by efficiency (higher bytes/token = more efficient)
    lang_results.sort(key=lambda x: x[1], reverse=True)

    for i, (name, efficiency, unique_tokens) in enumerate(lang_results, 1):
        print(
            f"  {i:2d}. {name:<20} | Bytes/token: {efficiency:6.2f} | Unique tokens: {unique_tokens:>5,d}"
        )


def main():
    """Main CLI function."""
    print("🚀 Starting tokka-bench")

    # Parse arguments with OmegaConf
    config = OmegaConf.from_cli()

    # Get required and optional parameters
    tokenizer = config.get("tokenizer")
    sample_size = config.get("sample_size", 1.0)
    output_name = config.get("output_name", None)

    if not tokenizer:
        print("❌ Error: tokenizer is required")
        print("Example: uv run cli/benchmark.py tokenizer=Xenova/gpt-4 sample_size=1.0")
        sys.exit(1)

    print(f"Tokenizer: {tokenizer}")
    print(f"Sample size: {sample_size}MB per language")

    if output_name:
        print(f"Output filename: data/results/{output_name}.json")
    else:
        safe_name = tokenizer.replace("/", "_").replace("-", "_")
        print(f"Output filename: data/results/{safe_name}.json")

    print("--------------------------------------------------")

    try:
        print("🔄 Loading language data...")
        results = run_benchmark(tokenizer, output_name, sample_size)

        print("🔄 Printing summary...")
        print_summary(results)

        print("✅ Benchmark completed successfully!")

    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        sys.exit(1)

    # Simple exit
    print("🔄 Exiting...")
    sys.exit(0)


if __name__ == "__main__":
    main()
