#!/usr/bin/env python3
"""CLI for running tokka-bench."""

import sys

from omegaconf import OmegaConf

from tokka_bench.fast_benchmark import run_benchmark as run_benchmark_fast

# Default configuration constants
DEFAULT_SAMPLE_SIZE = 2.0
DEFAULT_MAX_WORKERS = 4
DEFAULT_NATURAL_LANGUAGES = 99
DEFAULT_CODE_LANGUAGES = 20

# Output formatting constants
TABLE_WIDTH = 80
METRICS_SECTION_WIDTH = 60
RANK_COLUMN_WIDTH = 4
LANGUAGE_COLUMN_WIDTH = 25
BYTES_TOKEN_COLUMN_WIDTH = 11
UNIQUE_TOKENS_COLUMN_WIDTH = 13
FERTILITY_COLUMN_WIDTH = 9
SPLIT_RATE_COLUMN_WIDTH = 10
SCRIPT_NAME_WIDTH = 12
SCRIPT_VALUE_WIDTH = 6

# File path constants
RESULTS_DIR = "data/results"
JSON_EXTENSION = ".json"


def print_summary(results):
    """Print a summary of benchmark results."""
    if not results or "languages" not in results:
        print("⚠️  No language results available")
        return

    print("📊 Language Summary (ranked by efficiency):")

    # Sort languages by bytes_per_token (descending - higher is better)
    lang_results = []
    try:
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
    except (KeyError, TypeError) as e:
        print(f"⚠️  Error processing language results: {e}")
        return

    # Sort by efficiency (higher bytes/token = more efficient)
    lang_results.sort(key=lambda x: x[1], reverse=True)

    print(
        f"{'Rank':>{RANK_COLUMN_WIDTH}} {'Language':<{LANGUAGE_COLUMN_WIDTH}} {'Bytes/Token':>{BYTES_TOKEN_COLUMN_WIDTH}} {'Unique Tokens':>{UNIQUE_TOKENS_COLUMN_WIDTH}} {'Fertility':>{FERTILITY_COLUMN_WIDTH}} {'Split Rate':>{SPLIT_RATE_COLUMN_WIDTH}}"
    )
    print("─" * TABLE_WIDTH)

    for i, (name, efficiency, unique_tokens, fertility, split_rate) in enumerate(
        lang_results, 1
    ):
        print(
            f"  {i:2d}. {name:<{LANGUAGE_COLUMN_WIDTH}} {efficiency:>{BYTES_TOKEN_COLUMN_WIDTH - 1}.2f} {unique_tokens:>{UNIQUE_TOKENS_COLUMN_WIDTH - 1},d} {fertility:>{FERTILITY_COLUMN_WIDTH - 1}.2f} {split_rate:>{SPLIT_RATE_COLUMN_WIDTH - 1}.1f}%"
        )


def print_global_metrics(results):
    """Print global metrics summary."""
    if not results:
        print("⚠️  No results available")
        return

    global_metrics = results.get("global_metrics", {})
    vocab_metrics = results.get("vocab_metrics", {})

    if not global_metrics:
        print("⚠️  No global metrics available")
        return

    print("\n📊 Vocabulary Analysis:")
    print("─" * METRICS_SECTION_WIDTH)
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
    print("─" * METRICS_SECTION_WIDTH)

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
    print("\nUnicode Script Coverage:")

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
            print(
                f"  {script.title():<{SCRIPT_NAME_WIDTH}}: {global_metrics[key]:>{SCRIPT_VALUE_WIDTH}.1f}%"
            )

    # Show specific content categories
    for category in ["punctuation", "symbols", "numbers"]:
        key = f"tokens_with_{category}_unicode_pct"
        if key in global_metrics and global_metrics[key] > 0:
            print(
                f"  {category.title():<{SCRIPT_NAME_WIDTH}}: {global_metrics[key]:>{SCRIPT_VALUE_WIDTH}.1f}%"
            )

    print("─" * METRICS_SECTION_WIDTH)


def parse_config():
    """Parse CLI configuration."""
    try:
        config = OmegaConf.from_cli()
    except Exception as e:
        raise RuntimeError(f"Failed to parse CLI arguments: {e}") from e

    return {
        "tokenizer": config.get("tokenizer"),
        "tokenizers": config.get("tokenizers"),
        "sample_size": config.get("sample_size", DEFAULT_SAMPLE_SIZE),
        "output_name": config.get("output_name", None),
        "output_names": config.get("output_names", None),
        "max_workers": config.get("max_workers", DEFAULT_MAX_WORKERS),
        "natural_n": config.get("natural_n", None),
        "code_n": config.get("code_n", None),
    }


def parse_tokenizers(config):
    """Parse and validate tokenizer configuration."""
    tokenizer = config["tokenizer"]
    tokenizers = config["tokenizers"]
    # Engine selection: fast only (legacy classic removed)

    # Handle tokenizer input (support both single and multiple)
    if tokenizers:
        # Multiple tokenizers specified
        if isinstance(tokenizers, str):
            tokenizer_list = [t.strip() for t in tokenizers.split(",") if t.strip()]
        else:
            tokenizer_list = (
                tokenizers if isinstance(tokenizers, list) else [tokenizers]
            )
    elif tokenizer:
        # Single tokenizer specified
        if isinstance(tokenizer, str) and "," in tokenizer:
            tokenizer_list = [t.strip() for t in tokenizer.split(",") if t.strip()]
        else:
            tokenizer_list = [tokenizer]
    else:
        _print_usage_error()
        return None

    # Validate tokenizer list
    if not tokenizer_list or any(not t for t in tokenizer_list):
        _print_usage_error("Empty tokenizer names not allowed")
        return None

    return tokenizer_list


def _print_usage_error(message=None):
    """Print usage error and examples."""
    if message:
        print(f"❌ Error: {message}")
    else:
        print("❌ Error: tokenizer or tokenizers is required")
    print("Examples:")
    print("  Single:   uv run benchmark tokenizer=openai-community/gpt2")
    print(
        "  Multiple: uv run benchmark tokenizers=openai-community/gpt2,google/gemma-2-27b-it"
    )
    print(
        "  Multiple: uv run benchmark tokenizer=openai-community/gpt2,google/gemma-2-27b-it"
    )
    print("  Workers:  uv run benchmark tokenizers=gpt2,gemma max_workers=8")
    sys.exit(1)


def parse_output_names(config, tokenizer_count):
    """Parse and validate output names configuration."""
    output_name = config["output_name"]
    output_names = config["output_names"]

    output_name_list = None
    if output_names:
        if isinstance(output_names, str):
            output_name_list = [n.strip() for n in output_names.split(",") if n.strip()]
        else:
            output_name_list = (
                output_names if isinstance(output_names, list) else [output_names]
            )
    elif output_name:
        output_name_list = [output_name]

    # Validate output names count
    if output_name_list and len(output_name_list) != tokenizer_count:
        print(
            f"❌ Error: Number of output names ({len(output_name_list)}) must match number of tokenizers ({tokenizer_count})"
        )
        sys.exit(1)

    return output_name_list


def generate_output_filename(tokenizer_name, output_name=None):
    """Generate output filename for a tokenizer."""
    if output_name:
        return f"{RESULTS_DIR}/{output_name}{JSON_EXTENSION}"
    else:
        safe_name = tokenizer_name.replace("/", "_").replace("-", "_")
        return f"{RESULTS_DIR}/{safe_name}{JSON_EXTENSION}"


def print_configuration(tokenizer_list, output_name_list, sample_size, max_workers):
    """Print benchmark configuration."""
    if len(tokenizer_list) == 1:
        print(f"Tokenizer: {tokenizer_list[0]}")
        output_file = generate_output_filename(
            tokenizer_list[0], output_name_list[0] if output_name_list else None
        )
        print(f"Output filename: {output_file}")
    else:
        print(f"Tokenizers ({len(tokenizer_list)}):")
        for i, tok in enumerate(tokenizer_list):
            output_name = output_name_list[i] if output_name_list else None
            output_file = generate_output_filename(tok, output_name)
            print(f"  {i + 1}. {tok} → {output_file}")

    print(f"Sample size: {sample_size}MB per language")
    if len(tokenizer_list) > 1:
        print(f"Parallel workers: {max_workers}")
    print("Engine: fast")
    print("-" * 50)


def main():
    """Main CLI function."""
    print("🚀 Starting tokka-bench")

    try:
        # Parse configuration
        config = parse_config()

        # Parse tokenizers
        tokenizer_list = parse_tokenizers(config)
        if not tokenizer_list:
            return

        # Parse output names
        output_name_list = parse_output_names(config, len(tokenizer_list))

        # Print configuration
        print_configuration(
            tokenizer_list,
            output_name_list,
            config["sample_size"],
            config["max_workers"],
        )

        # Run benchmark
        print("🔄 Loading language data...")
        run_benchmark = run_benchmark_fast
        # Fast engine supports natural_n/code_n knobs
        all_results = run_benchmark(
            tokenizer_list,
            output_name_list,
            config["sample_size"],
            config["max_workers"],
            config["natural_n"]
            if config["natural_n"] is not None
            else DEFAULT_NATURAL_LANGUAGES,
            config["code_n"]
            if config["code_n"] is not None
            else DEFAULT_CODE_LANGUAGES,
        )

        # Print results
        print("🔄 Printing summary...")
        for tokenizer_name in tokenizer_list:
            results = all_results.get(tokenizer_name, {})
            print(f"\n📊 Results for {tokenizer_name}:")
            print_summary(results)
            print_global_metrics(results)

        print("\n✅ All benchmarks completed successfully!")

    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as e:
        print(f"❌ Error running benchmark: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please report this issue with the full error message.")
        sys.exit(1)


if __name__ == "__main__":
    main()
