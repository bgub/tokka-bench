"""
UI controls and sidebar components for the dashboard.
"""

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from .charts import (
    create_efficiency_chart,
    create_coverage_chart,
    create_subword_fertility_chart,
    create_continued_word_rate_chart,
    create_script_distribution_chart,
    create_vocab_metrics_chart,
    create_vocab_efficiency_scatter,
    create_summary_table,
)


def render_sidebar_controls(
    df: pd.DataFrame, language_categories: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    """Render global sidebar controls with category presets."""
    st.sidebar.header("🎛️ Global Controls")

    # Initialize session state for global selections
    if "selected_tokenizers" not in st.session_state:
        st.session_state.selected_tokenizers = sorted(df["tokenizer_key"].unique())
    if "selected_languages" not in st.session_state:
        st.session_state.selected_languages = list(df["language"].unique())

    # Category preset buttons
    st.sidebar.subheader("📂 Language Presets")

    # Create preset buttons
    preset_descriptions = {
        "All Languages": "All available languages in proper order",
        "English": "English from FineWeb sample-10BT",
        "Natural Languages": "Human languages only",
        "Programming Languages": "Code languages only",
        "Top 10": "Top 10 languages by size",
        "11-40": "Next 30 natural languages",
        "41-100": "Remaining 60 natural languages (41st-100th largest)",
        "Core Programming": "Core programming languages",
        "Systems Programming": "Systems and compiled languages",
        "Latin Script": "Latin script languages",
        "Cyrillic Script": "Cyrillic script languages",
        "Asian Scripts": "Asian script languages",
        "Arabic Script": "Arabic script languages",
        "European Languages": "European languages",
        "Major World Languages": "Major world languages",
    }

    # Create buttons in a nice layout
    for category, languages in language_categories.items():
        if category in preset_descriptions:
            if st.sidebar.button(
                f"{category} ({len(languages)})",
                help=preset_descriptions[category],
                use_container_width=True,
            ):
                st.session_state.selected_languages = languages
                st.rerun()

    st.sidebar.divider()

    # Global tokenizer selection
    available_tokenizers = sorted(df["tokenizer_key"].unique())
    selected_tokenizers = st.sidebar.multiselect(
        "Select Tokenizers to Compare",
        available_tokenizers,
        default=st.session_state.selected_tokenizers,
        key="global_tokenizers",
    )

    # Global language selection
    all_languages = sorted(df["language"].unique())
    selected_languages = st.sidebar.multiselect(
        "Filter Languages",
        all_languages,
        default=st.session_state.selected_languages,
        key="global_languages",
    )

    st.sidebar.divider()

    # Quick stats
    st.sidebar.metric("Tokenizers Selected", len(selected_tokenizers))
    st.sidebar.write(f"📊 {len(available_tokenizers)} available")

    st.sidebar.metric("Languages Selected", len(selected_languages))
    st.sidebar.write(f"🌐 {len(all_languages)} available")

    return selected_tokenizers, selected_languages


def render_main_content(
    df: pd.DataFrame, selected_tokenizers: List[str], selected_languages: List[str]
):
    """Render the main dashboard content."""
    if not selected_tokenizers:
        st.warning("Please select at least one tokenizer to compare.")
        return

    if not selected_languages:
        st.warning("Please select at least one language to view.")
        return

    # Filter dataframe based on global selections
    display_df = df[
        (df["tokenizer_key"].isin(selected_tokenizers))
        & (df["language"].isin(selected_languages))
    ]

    if display_df.empty:
        st.warning(
            "No data available for the selected combination of tokenizers and languages."
        )
        return

    # Analyze current selection
    programming_langs = [
        lang for lang in selected_languages if "(code)" in str(lang).lower()
    ]
    natural_langs = [
        lang for lang in selected_languages if "(code)" not in str(lang).lower()
    ]

    # Selection summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Programming Languages", len(programming_langs))
        if len(programming_langs) > 0:
            st.write("🔧 Code Languages")
    with col2:
        st.metric("Natural Languages", len(natural_langs))
        if len(natural_langs) > 0:
            st.write("🌍 Human Languages")

    # Analysis type description
    if len(programming_langs) > 0 and len(natural_langs) > 0:
        st.info(
            "🌐 **Mixed Analysis** • Comparing code and natural language tokenization"
        )
    elif len(programming_langs) > 0:
        st.info("💻 **Code Focus** • Analyzing programming language efficiency")
    elif len(natural_langs) > 0:
        st.info("🌍 **Language Focus** • Analyzing human language tokenization")

    # Summary table
    st.subheader("📈 Summary Rankings")
    summary_df = create_summary_table(display_df, selected_tokenizers)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Charts section
    st.subheader("📊 Detailed Analysis")

    # Create tabs for different views
    efficiency_tab, coverage_tab, subword_tab, vocab_tab, analysis_tab, raw_tab = (
        st.tabs(
            [
                "🚀 Efficiency",
                "🎯 Coverage",
                "🔤 Subword Analysis",
                "📚 Vocab Analysis",
                "📏 Analysis",
                "🔍 Raw Data",
            ]
        )
    )

    with efficiency_tab:
        st.write("#### Tokenization Efficiency")
        st.caption("Higher values = more efficient tokenization")
        efficiency_chart = create_efficiency_chart(display_df, selected_tokenizers)
        st.plotly_chart(efficiency_chart, use_container_width=True)

        if len(programming_langs) > 0:
            st.info(
                f"💻 **{len(programming_langs)} coding languages** • Look for patterns in syntax complexity"
            )

    with coverage_tab:
        st.write("#### Vocabulary Coverage")
        st.caption("Higher values = better language support")
        coverage_chart = create_coverage_chart(display_df, selected_tokenizers)
        st.plotly_chart(coverage_chart, use_container_width=True)

        if len(natural_langs) > 0:
            st.info(
                f"🌍 **{len(natural_langs)} human languages** • Higher coverage indicates better script support"
            )

    with subword_tab:
        st.write("#### Subword Analysis")

        # Two columns for subword metrics
        col1, col2 = st.columns(2)

        with col1:
            st.write("##### Subword Fertility")
            st.caption("Subwords per word - Higher = more fragmented")
            subword_fertility_chart = create_subword_fertility_chart(
                display_df, selected_tokenizers
            )
            st.plotly_chart(subword_fertility_chart, use_container_width=True)

        with col2:
            st.write("##### Continued Word Rate")
            st.caption("% of tokens continuing words - Higher = more subword splitting")
            continued_word_rate_chart = create_continued_word_rate_chart(
                display_df, selected_tokenizers
            )
            st.plotly_chart(continued_word_rate_chart, use_container_width=True)

        st.info(
            "🔤 **Subword Metrics** • Subword fertility shows how many pieces each word breaks into. "
            "Continued word rate shows what percentage of tokens are continuations of words (not word-initial)."
        )

    with vocab_tab:
        st.write("#### Vocabulary Analysis")

        # Two columns for vocab metrics
        col1, col2 = st.columns(2)

        with col1:
            st.write("##### Vocabulary Composition")
            st.caption("% of tokens without leading space")
            vocab_metrics_chart = create_vocab_metrics_chart(
                display_df, selected_tokenizers
            )
            st.plotly_chart(vocab_metrics_chart, use_container_width=True)

        with col2:
            st.write("##### Script Distribution")
            st.caption("Script composition of tokenizer vocabulary")
            script_distribution_chart = create_script_distribution_chart(
                display_df, selected_tokenizers
            )
            st.plotly_chart(script_distribution_chart, use_container_width=True)

        st.info(
            "📚 **Vocabulary Metrics** • These show the composition of the tokenizer's vocabulary. "
            "Script distribution reveals which writing systems are better represented."
        )

    with analysis_tab:
        st.write("#### Efficiency vs Vocabulary Size")
        st.caption("How tokenizer size affects average efficiency across languages")

        # Overall scatter plot
        vocab_scatter = create_vocab_efficiency_scatter(display_df, selected_tokenizers)
        st.plotly_chart(vocab_scatter, use_container_width=True)

        # Category-specific analysis
        if len(programming_langs) > 0 and len(natural_langs) > 0:
            st.write("##### By Language Type")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**🌍 Natural Languages**")
                natural_df = display_df[display_df["language"].isin(natural_langs)]
                if not natural_df.empty:
                    natural_scatter = create_vocab_efficiency_scatter(
                        natural_df, selected_tokenizers
                    )
                    natural_scatter.update_layout(title="Natural Languages", height=400)
                    st.plotly_chart(natural_scatter, use_container_width=True)
                else:
                    st.info("No natural language data available")

            with col2:
                st.write("**💻 Programming Languages**")
                prog_df = display_df[display_df["language"].isin(programming_langs)]
                if not prog_df.empty:
                    prog_scatter = create_vocab_efficiency_scatter(
                        prog_df, selected_tokenizers
                    )
                    prog_scatter.update_layout(
                        title="Programming Languages", height=400
                    )
                    st.plotly_chart(prog_scatter, use_container_width=True)
                else:
                    st.info("No programming language data available")

    with raw_tab:
        st.write("#### Complete Dataset")
        st.dataframe(display_df, use_container_width=True)

        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"tokka_bench_data_{len(selected_tokenizers)}tokenizers_{len(selected_languages)}languages.csv",
            mime="text/csv",
        )
