"""
Tokka-bench visualization package.

This package provides a Streamlit-based dashboard for comparing tokenizer
efficiency across multiple languages.
"""

from tokka_bench.visualization.dashboard import launch_dashboard
from tokka_bench.visualization.app import main

__all__ = ["launch_dashboard", "main"]
