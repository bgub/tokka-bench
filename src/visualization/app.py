"""
Main Streamlit app for the tokka-bench visualization dashboard.
Single-page design with:
- Collapsed top table
- Collapsed global vocab & scripts
- Main analysis chart
- Filters (tokenizers, languages, simple preset dropdown) below the chart
"""

import streamlit as st
import html as _html

from .categories import detect_language_types
from .charts import (
    create_script_distribution_chart,
    create_vocab_metrics_chart,
    create_efficiency_chart,
    create_coverage_chart,
    create_subword_fertility_chart,
    create_word_splitting_rate_chart,
    create_vocab_efficiency_scatter,
)
from .data import load_all_results, results_to_dataframe, get_tokenizer_summary


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Tokka-Bench Visualizer",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        /* Subtle theming tweaks */
        .block-container { padding-top: 0.5rem; }
        /* Align the Filters header label to the top of its row */
        .filters-header { margin-top: 0; margin-bottom: 0.25rem; }
        /* Tighten spacing under the preset select so it aligns visually */
        .stSelectbox div[data-baseweb="select"] { margin-top: 0; }
        /* Make preset input non-typable while keeping it clickable */
        .stSelectbox div[data-baseweb="select"] input { pointer-events: none; }
        .stSelectbox div[data-baseweb="select"] input { caret-color: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Tokka-Bench Dashboard")

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

    # Build language categories early so we can set a friendly default selection
    language_categories = detect_language_types(df)
    st.session_state["language_categories"] = language_categories

    # Collapsed top table
    with st.expander("Benchmark Details", expanded=False):
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
        "Vocabulary Composition & Script Distribution (Global)", expanded=False
    ):
        all_tokenizers = list(df["tokenizer_key"].unique())
        st.subheader("Script Distribution", help=None)
        script_chart = create_script_distribution_chart(df, all_tokenizers)
        st.plotly_chart(script_chart, use_container_width=True)

        st.subheader("Vocabulary Composition", help=None)
        vocab_chart = create_vocab_metrics_chart(df, all_tokenizers)
        st.plotly_chart(vocab_chart, use_container_width=True)

    # Initialize selection state
    if "selected_tokenizers" not in st.session_state:
        desired_tokenizers = ["GPT-2", "gpt-oss", "Kimi K2"]
        available_tokenizers = list(df["tokenizer_key"].unique())
        defaults = [t for t in desired_tokenizers if t in available_tokenizers]
        if not defaults:
            defaults = sorted(available_tokenizers)[:3] if available_tokenizers else []
        st.session_state.selected_tokenizers = defaults
    if "selected_languages" not in st.session_state:
        top_30 = language_categories.get("Top 30 Natural", [])
        st.session_state.selected_languages = (
            top_30 if top_30 else list(df["language"].unique())
        )

    selected_tokenizers = st.session_state.selected_tokenizers
    selected_languages = st.session_state.selected_languages

    # Main detailed analysis charts (tabs)
    st.subheader("Detailed Analysis")

    # Build subset according to current selection state
    display_df = df[
        (df["tokenizer_key"].isin(selected_tokenizers))
        & (df["language"].isin(selected_languages))
    ]

    (
        efficiency_tab,
        coverage_tab,
        continued_tab,
        fertility_tab,
        analysis_tab,
        raw_tab,
    ) = st.tabs(
        [
            "Efficiency",
            "Coverage",
            "Word Splitting Rate",
            "Subword Fertility",
            "Comparisons",
            "Raw Data",
        ]
    )

    with efficiency_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.caption(
                "Bytes per token. Higher = more efficient. Not comparable across languages—compare tokenizers within the same language."
            )
            st.plotly_chart(
                create_efficiency_chart(display_df, selected_tokenizers),
                use_container_width=True,
            )
            # Optional: Show tokenization sample for hovered language via selection
            if "sample_tokens_preview" in display_df.columns:
                st.caption("Hover a bar to see sample details; or view samples below.")

    with coverage_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.caption(
                "Unique token IDs used in the sample. Higher generally indicates better script coverage for that language."
            )
            st.plotly_chart(
                create_coverage_chart(display_df, selected_tokenizers),
                use_container_width=True,
            )

    with continued_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.caption(
                "% of units (words/characters/syllables) that split into multiple tokens. Higher = more splitting. Not comparable across languages—compare tokenizers within the same language."
            )
            st.plotly_chart(
                create_word_splitting_rate_chart(display_df, selected_tokenizers),
                use_container_width=True,
            )

    with fertility_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.caption(
                "Average subwords per word. Higher = more fragmentation. Not comparable across languages—compare tokenizers within the same language."
            )
            st.plotly_chart(
                create_subword_fertility_chart(display_df, selected_tokenizers),
                use_container_width=True,
            )

    with analysis_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.caption(
                "Scatter of average bytes/token vs vocabulary size per tokenizer (log-scale x-axis)."
            )
            st.plotly_chart(
                create_vocab_efficiency_scatter(display_df, selected_tokenizers),
                use_container_width=True,
            )

    with raw_tab:
        if display_df.empty:
            st.info("No data for current selection. Adjust filters below.")
        else:
            st.caption("Use this table to export data and run your own analyses.")
            st.subheader("Raw Data")
            st.dataframe(display_df, use_container_width=True)
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"tokka_bench_data_{len(selected_tokenizers)}tokenizers_{len(selected_languages)}languages.csv",
                mime="text/csv",
            )

    st.markdown("---")

    # Filters below chart (header + preset on same line)

    def _apply_preset():
        preset_val = st.session_state.get("language_preset")
        cats = st.session_state.get("language_categories", {})
        if preset_val and preset_val in cats:
            st.session_state.selected_languages = cats[preset_val]

    header_col, preset_col = st.columns([6, 2])
    with header_col:
        # Add a class for CSS targeting to align to top
        st.markdown('<h3 class="filters-header">Filters</h3>', unsafe_allow_html=True)
    with preset_col:
        st.selectbox(
            "Language Preset",
            options=list(language_categories.keys()),
            index=(
                list(language_categories.keys()).index("Top 30 Natural")
                if "Top 30 Natural" in language_categories
                else 0
            ),
            key="language_preset",
            on_change=_apply_preset,
            label_visibility="collapsed",
            help="Language preset",
            disabled=False,
        )

    # Row 2: tokenizers and languages side by side
    tok_col, lang_col = st.columns(2)
    with tok_col:
        st.multiselect(
            "Tokenizers",
            options=sorted(df["tokenizer_key"].unique()),
            key="selected_tokenizers",
        )
    with lang_col:
        st.multiselect(
            "Languages",
            options=list(df["language"].unique()),
            key="selected_languages",
        )

    # Refresh current selections after widgets update
    selected_tokenizers = st.session_state.selected_tokenizers
    selected_languages = st.session_state.selected_languages

    # Tokenization Preview section
    st.markdown("---")
    st.subheader("Tokenization Preview")
    st.caption("Preview the exact sample text used for a tokenizer-language pair.")

    # Build dropdowns (names for display, map back to keys)
    tok_key_to_name = {
        k: v for k, v in df.groupby("tokenizer_key")["tokenizer_name"].first().items()
    }
    tok_name_to_key = {v: k for k, v in tok_key_to_name.items()}

    tok_names = sorted(tok_name_to_key.keys())
    lang_names = list(df["language"].unique())

    col_tok, col_lang = st.columns(2)
    with col_tok:
        selected_tok_name = st.selectbox(
            "Tokenizer",
            options=tok_names,
            index=(
                tok_names.index(
                    tok_key_to_name.get(selected_tokenizers[0], tok_names[0])
                )
                if selected_tokenizers
                else 0
            ),
            key="preview_tokenizer_name",
        )
    with col_lang:
        default_lang_index = (
            lang_names.index(selected_languages[0]) if selected_languages else 0
        )
        selected_lang_name = st.selectbox(
            "Language",
            options=lang_names,
            index=default_lang_index,
            key="preview_language_name",
        )

    # Retrieve matching row and show sample text
    preview_df = df[
        (df["tokenizer_key"] == tok_name_to_key.get(selected_tok_name, ""))
        & (df["language"] == selected_lang_name)
    ]

    if preview_df.empty or "sample_text" not in preview_df.columns:
        st.info("No sample available for this selection.")
    else:
        sample_text = preview_df["sample_text"].iloc[0]
        sample_tokens = (
            preview_df["sample_tokens"].iloc[0]
            if "sample_tokens" in preview_df.columns
            else None
        )
        if not sample_text and not sample_tokens:
            st.info("No sample available for this selection.")
        else:
            # Flag fallback synthetic samples
            if "Sample text for" in sample_text and "tokenizer testing" in sample_text:
                st.warning(
                    "Showing fallback sample text for this language (real dataset unavailable)."
                )

            # Render tokenized sample if available; otherwise show raw text
            if sample_tokens:
                # Color each token using a small repeating palette; allow natural wrapping
                palette = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ]
                spans = []
                for i, tok in enumerate(sample_tokens):
                    color = palette[i % len(palette)]
                    # Avoid adding spaces around the separator; rely on <wbr> for wrap
                    safe_tok = _html.escape(str(tok))
                    spans.append(
                        f'<span style="color:{color}; margin-right: 0.25rem">{safe_tok}</span>'
                    )
                # Join with zero-width space so tokens can wrap without visible separators
                zws = "\u200b"
                html = (
                    "<div style=\"font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; white-space: normal; word-break: break-word; border: 1px solid #e9ecef; padding: 0.75rem; background: var(--background-secondary); border-radius: 6px; line-height: 1.6;\">"
                    + zws.join(spans)
                    + "</div>"
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.code(sample_text)

    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
