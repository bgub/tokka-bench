"""
Main Streamlit app for the tokka-bench visualization dashboard.
Single-page design with:
- Collapsed top table
- Collapsed global vocab & scripts
- Main analysis chart
- Filters (tokenizers, languages, simple preset dropdown) below the chart
"""

import streamlit as st

from .categories import detect_language_types
from .charts import (
    create_script_distribution_chart,
    create_vocab_metrics_chart,
    create_efficiency_chart,
    create_coverage_chart,
    create_subword_fertility_chart,
    create_continued_word_rate_chart,
    create_vocab_efficiency_scatter,
)
from .data import load_all_results, results_to_dataframe, get_tokenizer_summary


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Tokka-Bench Visualizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        /* Subtle theming tweaks */
        .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 Tokka-Bench Dashboard")
    st.caption("Compare tokenizer efficiency across natural and programming languages")

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

    # Collapsed top table
    with st.expander("📋 Detailed Tokenizer Information", expanded=False):
        display_cols = ["tokenizer_name", "vocab_size", "datetime"]
        optional_cols = ["tokens_without_leading_space_pct", "total_tokens_analyzed"]
        for col in optional_cols:
            if (
                col in tokenizer_summary.columns
                and not tokenizer_summary[col].isna().all()
            ):
                display_cols.append(col)
        display_summary = tokenizer_summary[display_cols].copy()
        if "datetime" in display_summary.columns:
            display_summary["datetime"] = display_summary["datetime"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

    # Collapsed global vocab/scripts (stacked)
    with st.expander(
        "📚 Vocabulary Composition & Script Distribution (Global)", expanded=False
    ):
        all_tokenizers = list(df["tokenizer_key"].unique())
        st.write("##### Script Distribution")
        script_chart = create_script_distribution_chart(df, all_tokenizers)
        st.plotly_chart(script_chart, use_container_width=True)

        st.write("##### Vocabulary Composition")
        vocab_chart = create_vocab_metrics_chart(df, all_tokenizers)
        st.plotly_chart(vocab_chart, use_container_width=True)

    # Initialize selection state
    if "selected_tokenizers" not in st.session_state:
        st.session_state.selected_tokenizers = sorted(df["tokenizer_key"].unique())
    if "selected_languages" not in st.session_state:
        st.session_state.selected_languages = list(df["language"].unique())

    selected_tokenizers = st.session_state.selected_tokenizers
    selected_languages = st.session_state.selected_languages

    # Main detailed analysis charts (tabs)
    st.subheader("📈 Detailed Analysis")

    # Build subset according to current selection state
    display_df = df[
        (df["tokenizer_key"].isin(selected_tokenizers))
        & (df["language"].isin(selected_languages))
    ]

    efficiency_tab, coverage_tab, subword_tab, analysis_tab, raw_tab = st.tabs(
        [
            "🚀 Efficiency",
            "🎯 Coverage",
            "🔤 Subword Analysis",
            "📏 Analysis",
            "🔍 Raw Data",
        ]
    )

    with efficiency_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.caption("Higher values = more efficient tokenization")
            st.plotly_chart(
                create_efficiency_chart(display_df, selected_tokenizers),
                use_container_width=True,
            )

    with coverage_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.caption("Higher values = better language coverage")
            st.plotly_chart(
                create_coverage_chart(display_df, selected_tokenizers),
                use_container_width=True,
            )

    with subword_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            # Continued Word Rate first
            st.write("##### Continued Word Rate")
            st.caption("% of tokens continuing words - Higher = more subword splitting")
            st.plotly_chart(
                create_continued_word_rate_chart(display_df, selected_tokenizers),
                use_container_width=True,
            )

            # Subword Fertility below
            st.write("##### Subword Fertility")
            st.caption("Subwords per word - Higher = more fragmented")
            st.plotly_chart(
                create_subword_fertility_chart(display_df, selected_tokenizers),
                use_container_width=True,
            )

    with analysis_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.write("#### Efficiency vs Vocabulary Size")
            st.caption("How tokenizer size affects average efficiency across languages")
            st.plotly_chart(
                create_vocab_efficiency_scatter(display_df, selected_tokenizers),
                use_container_width=True,
            )

    with raw_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.write("#### Complete Dataset")
            st.dataframe(display_df, use_container_width=True)
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"tokka_bench_data_{len(selected_tokenizers)}tokenizers_{len(selected_languages)}languages.csv",
                mime="text/csv",
            )

    st.markdown("---")

    # Filters below chart (header + preset on same line)
    language_categories = detect_language_types(df)

    # Store categories for callbacks
    st.session_state["language_categories"] = language_categories

    def _apply_preset():
        preset_val = st.session_state.get("language_preset")
        cats = st.session_state.get("language_categories", {})
        if preset_val and preset_val in cats:
            st.session_state.selected_languages = cats[preset_val]

    header_col, preset_col = st.columns([6, 2])
    with header_col:
        st.markdown("### Filters")
    with preset_col:
        st.selectbox(
            "Language Preset",
            options=list(language_categories.keys()),
            index=list(language_categories.keys()).index("All Languages")
            if "All Languages" in language_categories
            else 0,
            key="language_preset",
            on_change=_apply_preset,
            label_visibility="collapsed",
            help="Language preset",
        )

    # Row 2: tokenizers and languages side by side
    tok_col, lang_col = st.columns(2)
    with tok_col:
        st.session_state.selected_tokenizers = st.multiselect(
            "Tokenizers",
            options=sorted(df["tokenizer_key"].unique()),
            default=st.session_state.selected_tokenizers,
            key="tokenizer_multiselect",
        )
    with lang_col:
        st.session_state.selected_languages = st.multiselect(
            "Languages",
            options=list(df["language"].unique()),
            default=st.session_state.selected_languages,
            key="language_multiselect",
        )

    # Footer
    st.markdown("---")
    st.caption(
        "Higher bytes/token = more efficient • Higher unique tokens = better coverage • Subword metrics show splitting behavior"
    )


if __name__ == "__main__":
    main()
