#!/usr/bin/env python3
"""
Simple launcher for the Tokka-Bench visualization dashboard.

This script launches the Streamlit app with proper error handling.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching Tokka-Bench Dashboard...")

    # Check if results exist
    results_dir = Path("data/results")
    if not results_dir.exists() or not list(results_dir.glob("*.json")):
        print("❌ No benchmark results found!")
        print("Run some benchmarks first:")
        print("  uv run cli/benchmark.py tokenizer=openai-community/gpt2")
        print("  uv run cli/benchmark.py tokenizer=Xenova/gpt-4")
        sys.exit(1)

    # Count available results
    result_files = list(results_dir.glob("*.json"))
    print(f"📊 Found {len(result_files)} benchmark result(s)")
    for file in result_files:
        print(f"  - {file.stem}")

    print("\n🌐 Starting dashboard server...")
    print("👉 Dashboard will open in your browser automatically")
    print("👉 Press Ctrl+C to stop the server")

    try:
        # Launch streamlit
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "cli/visualize.py",
                "--browser.gatherUsageStats",
                "false",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
