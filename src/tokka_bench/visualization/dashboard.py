"""
Dashboard launcher with pre-flight checks.
"""

import subprocess
import sys
from pathlib import Path

from tokka_bench.visualization.constants import RESULTS_DIR


def launch_dashboard():
    """Launch the Streamlit dashboard with checks - entry point for dashboard command."""
    print("🚀 Launching Tokka-Bench Dashboard...")

    # Check if results exist
    results_dir = Path(RESULTS_DIR)
    if not results_dir.exists() or not list(results_dir.glob("*.json")):
        print("❌ No benchmark results found!")
        print("Run some benchmarks first:")
        print("  uv run benchmark tokenizer=openai-community/gpt2")
        print("  uv run benchmark tokenizer=Xenova/gpt-4")
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
        # Launch streamlit pointing to the new app module
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app.py",
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
