#!/usr/bin/env python3
"""CLI script for running tokenizer benchmarks."""

import argparse
import os
import sys

from omegaconf import DictConfig, OmegaConf

# Add src to Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tokka_bench.benchmark import run_benchmark


def create_config() -> DictConfig:
    """Create default configuration."""
    default_config = OmegaConf.create(
        {
            "tokenizer": None,  # Required
            "output_dir": "data/results",
            "sample_size_mb": 1.0,
        }
    )
    return default_config


def main():
    """Run tokenizer benchmark on top 5 languages from FineWeb-2."""

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run tokenizer benchmark on top 5 languages from FineWeb-2"
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        required=True,
        help='Tokenizer model name (e.g., "openai-community/gpt2", "Xenova/gpt-4", "meta-llama/Meta-Llama-3-8B")',
    )
    parser.add_argument(
        "--output-dir", "-o", default="data/results", help="Directory to save results"
    )
    parser.add_argument(
        "--sample-size",
        "-s",
        type=float,
        default=1.0,
        help="Sample size in MB per language",
    )
    parser.add_argument("--config", help="Path to YAML config file (optional)")

    args = parser.parse_args()

    # Create base config
    config = create_config()

    # Load config file if provided
    if args.config:
        file_config = OmegaConf.load(args.config)
        config = OmegaConf.merge(config, file_config)

    # Override with command line arguments
    cli_config = OmegaConf.create(
        {
            "tokenizer": args.tokenizer,
            "output_dir": args.output_dir,
            "sample_size_mb": args.sample_size,
        }
    )
    config = OmegaConf.merge(config, cli_config)

    print("🚀 Starting tokka-bench")
    print(f"Tokenizer: {config.tokenizer}")
    print(f"Sample size: {config.sample_size_mb}MB per language")
    print(f"Output directory: {config.output_dir}")
    print("-" * 50)

    try:
        results = run_benchmark(
            tokenizer_name=config.tokenizer,
            output_dir=config.output_dir,
            sample_size_mb=config.sample_size_mb,
        )

        print("-" * 50)
        print("✅ Benchmark completed successfully!")
        print(f"Results saved for {len(results['languages'])} languages")

        # Print summary with efficiency ranking
        print("\n📊 Summary (ranked by efficiency):")

        # Sort languages by bytes_per_token (higher = more efficient)
        lang_items = list(results["languages"].items())
        lang_items.sort(key=lambda x: x[1]["metrics"]["bytes_per_token"], reverse=True)

        for i, (lang_key, lang_data) in enumerate(lang_items, 1):
            metrics = lang_data["metrics"]
            lang_info = lang_data["language_info"]
            print(
                f"  {i}. {lang_info['name']:20} | "
                f"Bytes/token: {metrics['bytes_per_token']:6.2f}"
            )

    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
