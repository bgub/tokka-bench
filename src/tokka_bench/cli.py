#!/usr/bin/env python3
"""CLI for running tokka-bench."""

import sys

from omegaconf import OmegaConf

from tokka_bench.benchmark import run_benchmark


def print_summary(results):
    """Print a summary of benchmark results."""
    print("📊 Language Summary (ranked by efficiency):")

    # Sort languages by bytes_per_token (descending - higher is better)
    lang_results = []
    for lang_key, lang_data in results["languages"].items():
        lang_name = lang_data["language_info"]["name"]
        metrics = lang_data["metrics"]
        lang_results.append(
            (
                lang_name,
                metrics["bytes_per_token"],
                metrics["unique_tokens"],
                metrics["subword_fertility"],
                metrics["continued_word_rate"],
            )
        )

    # Sort by efficiency (higher bytes/token = more efficient)
    lang_results.sort(key=lambda x: x[1], reverse=True)

    print(
        f"{'Rank':>4} {'Language':<25} {'Bytes/Token':>11} {'Unique Tokens':>13} {'Fertility':>9} {'Split Rate':>10}"
    )
    print("─" * 80)

    for i, (name, efficiency, unique_tokens, fertility, split_rate) in enumerate(
        lang_results, 1
    ):
        print(
            f"  {i:2d}. {name:<25} {efficiency:>10.2f} {unique_tokens:>12,d} {fertility:>8.2f} {split_rate:>9.1f}%"
        )


def print_global_metrics(results):
    """Print global metrics summary."""
    global_metrics = results.get("global_metrics", {})
    vocab_metrics = results.get("vocab_metrics", {})

    if not global_metrics:
        print("⚠️  No global metrics available")
        return

    print(f"\n📊 Vocabulary Analysis:")
    print("─" * 60)
    if vocab_metrics:
        print(f"Vocabulary size: {results.get('vocab_size', 0):,}")
        print(
            f"Tokens without leading space: {vocab_metrics.get('tokens_without_leading_space_pct', 0):.1f}% ({vocab_metrics.get('tokens_without_leading_space_count', 0):,} tokens)"
        )
        print(
            f"Sample analyzed: {vocab_metrics.get('analyzed_sample_size', 0):,} tokens"
        )
        sample_non_space = vocab_metrics.get("sample_non_space_tokens", [])
        sample_space = vocab_metrics.get("sample_space_tokens", [])
        if sample_non_space:
            print(f"Sample non-space tokens: {sample_non_space}")
        if sample_space:
            print(f"Sample space tokens: {sample_space}")

    print(
        f"\n🌍 Global Metrics (across {global_metrics.get('total_tokens_analyzed', 0):,} tokens):"
    )
    print("─" * 60)

    # Space and whitespace metrics
    print(
        f"Tokens starting with space:        {global_metrics.get('tokens_starting_with_space_pct', 0):.1f}%"
    )
    print(
        f"Tokens with whitespace in middle:  {global_metrics.get('tokens_with_whitespace_in_middle_pct', 0):.1f}%"
    )
    print(
        f"Tokens with script overlap:        {global_metrics.get('tokens_with_script_overlap_pct', 0):.1f}%"
    )

    # Unicode script metrics
    print(f"\nUnicode Script Coverage:")

    # Show script families
    for script in [
        "latin",
        "chinese",
        "cyrillic",
        "korean",
        "japanese",
        "arabic",
        "devanagari",
        "thai",
        "hebrew",
        "greek",
    ]:
        key = f"tokens_with_{script}_unicode_pct"
        if key in global_metrics and global_metrics[key] > 0:
            print(f"  {script.title():<12}: {global_metrics[key]:>6.1f}%")

    # Show specific content categories
    for category in ["punctuation", "symbols", "numbers"]:
        key = f"tokens_with_{category}_unicode_pct"
        if key in global_metrics and global_metrics[key] > 0:
            print(f"  {category.title():<12}: {global_metrics[key]:>6.1f}%")

    print("─" * 60)


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
        print("Example: uv run benchmark tokenizer=Xenova/gpt-4 sample_size=1.0")
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
        print_global_metrics(results)

        print("\n✅ Benchmark completed successfully!")

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
