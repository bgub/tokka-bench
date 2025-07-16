"""
Main Streamlit app for the tokka-bench visualization dashboard.
"""

import streamlit as st
import pandas as pd

from .categories import detect_language_types
from .controls import (
    render_sidebar_controls,
    render_main_content,
)
from .data import load_all_results, results_to_dataframe, get_tokenizer_summary


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
    tokenizer_summary = get_tokenizer_summary(results)

    # Display tokenizer-level information at the top
    if not tokenizer_summary.empty:
        st.subheader("🔧 Tokenizer Overview")

        # Show key tokenizer metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Tokenizers", len(tokenizer_summary))
            if "vocab_size" in tokenizer_summary.columns:
                avg_vocab_size = tokenizer_summary["vocab_size"].mean()
                if not pd.isna(avg_vocab_size):
                    st.metric("Avg Vocab Size", f"{avg_vocab_size:,.0f}")

        with col2:
            total_languages = len(df["language"].unique())
            st.metric("Total Languages", total_languages)
            if "total_tokens_analyzed" in tokenizer_summary.columns:
                max_tokens = tokenizer_summary["total_tokens_analyzed"].max()
                if not pd.isna(max_tokens):
                    st.metric("Max Tokens Analyzed", f"{max_tokens:,.0f}")

        with col3:
            # Show script diversity info
            if "tokens_with_latin_unicode_pct" in tokenizer_summary.columns:
                avg_latin_support = tokenizer_summary[
                    "tokens_with_latin_unicode_pct"
                ].mean()
                if not pd.isna(avg_latin_support):
                    st.metric("Avg Latin Support", f"{avg_latin_support:.1f}%")

        # Show expanded tokenizer summary table
        with st.expander("📋 Detailed Tokenizer Information", expanded=False):
            # Select relevant columns for display
            display_cols = ["tokenizer_name", "vocab_size", "datetime"]

            # Add available metrics
            optional_cols = [
                "tokens_without_leading_space_pct",
                "total_tokens_analyzed",
                "tokens_with_latin_unicode_pct",
                "tokens_with_cyrillic_unicode_pct",
                "tokens_with_japanese_unicode_pct",
                "tokens_with_chinese_unicode_pct",
            ]

            for col in optional_cols:
                if (
                    col in tokenizer_summary.columns
                    and not tokenizer_summary[col].isna().all()
                ):
                    display_cols.append(col)

            display_summary = tokenizer_summary[display_cols].copy()

            # Format the dataframe for better display
            if "datetime" in display_summary.columns:
                display_summary["datetime"] = display_summary["datetime"].dt.strftime(
                    "%Y-%m-%d %H:%M"
                )

            st.dataframe(display_summary, use_container_width=True, hide_index=True)

        st.divider()

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
        "*Higher bytes/token = more efficient • Higher unique tokens = better coverage • New metrics: subword fertility & continued word rate*"
    )


if __name__ == "__main__":
    main()
