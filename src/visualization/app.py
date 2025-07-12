"""
Main Streamlit app for the tokka-bench visualization dashboard.
"""

import streamlit as st

from .categories import detect_language_types
from .controls import (
    render_sidebar_controls,
    render_main_content,
)
from .data import load_all_results, results_to_dataframe


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Tokka-Bench Visualizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 Tokka-Bench Dashboard")
    st.markdown(
        "**Compare tokenizer efficiency across natural and programming languages**"
    )

    # Load data
    with st.spinner("Loading benchmark results..."):
        results = load_all_results()

    if not results:
        st.error(
            "No benchmark results found in `data/results/`. Run some benchmarks first!"
        )
        st.info("Example: `uv run benchmark tokenizer=openai-community/gpt2`")
        return

    df = results_to_dataframe(results)

    # Detect language types and categories
    language_categories = detect_language_types(df)

    # Render sidebar controls
    selected_tokenizers, selected_languages = render_sidebar_controls(
        df, language_categories
    )

    # Render main content
    render_main_content(df, selected_tokenizers, selected_languages)

    # Footer
    st.markdown("---")
    st.markdown(
        "*Higher bytes/token = more efficient • Higher unique tokens = better coverage*"
    )


if __name__ == "__main__":
    main()
