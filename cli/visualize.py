#!/usr/bin/env python3
"""
Streamlit app for visualizing and comparing tokenizer benchmark results.

Run with: streamlit run cli/visualize.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Constants
RESULTS_DIR = "data/results"
CHART_HEIGHT = 500
LEGEND_CONFIG = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

# Script grouping mappings
SCRIPT_GROUPS = {
    "Latn": "Latin",
    "Cyrl": "Cyrillic",
    "Arab": "Arabic",
    "Hani": "CJK",
    "Jpan": "CJK",
    "Hang": "CJK",
    "Deva": "Indic",
    "Beng": "Indic",
    "Guru": "Indic",
    "Taml": "Indic",
    "Telu": "Indic",
    "Knda": "Indic",
    "Mlym": "Indic",
    "Gujr": "Indic",
    "Orya": "Indic",
    "Thai": "Southeast Asian",
    "Laoo": "Southeast Asian",
    "Khmr": "Southeast Asian",
    "Mymr": "Southeast Asian",
    "Grek": "Other European/Middle Eastern",
    "Armn": "Other European/Middle Eastern",
    "Geor": "Other European/Middle Eastern",
    "Hebr": "Other European/Middle Eastern",
}


def detect_language_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect and categorize languages from the data with comprehensive 15-category system."""
    # Get languages in their original order from the data (preserves CSV order)
    all_languages = list(df["language"].unique())

    # Detect programming languages (those with "code" in script or name)
    programming_languages = []
    natural_languages = []
    english_languages = []

    for lang in all_languages:
        lang_data = df[df["language"] == lang].iloc[0]
        # Check if language name contains "(code)" or script is "code"
        if "(code)" in str(lang).lower() or str(lang_data["script"]).lower() == "code":
            programming_languages.append(lang)
        elif "english" in str(lang).lower() and "fineweb" in str(lang).lower():
            english_languages.append(lang)
        else:
            natural_languages.append(lang)

    # Preserve proper ordering: English → Natural → Programming
    ordered_all_languages = (
        english_languages + natural_languages + programming_languages
    )

    # === SIZE-BASED TIERS ===
    # Top 10 (English + top 9 natural languages by size)
    top_10 = english_languages + natural_languages[:9]
    
    # 11-40: Next 30 natural languages
    tier_11_40 = natural_languages[9:39] if len(natural_languages) > 9 else []
    
    # 41-100: Remaining natural languages
    tier_41_100 = natural_languages[39:] if len(natural_languages) > 39 else []

    # === PROGRAMMING LANGUAGE CATEGORIES ===
    # Core programming languages (most popular)
    core_programming = [lang for lang in programming_languages 
                       if any(tech in lang.lower() for tech in 
                             ['python', 'javascript', 'java', 'c ', 'cpp', 'c-sharp', 'php', 'typescript'])]
    
    # Systems & compiled languages
    systems_programming = [lang for lang in programming_languages 
                          if any(tech in lang.lower() for tech in 
                                ['rust', 'go', 'kotlin', 'scala', 'swift', 'dart', 'c ', 'cpp', 'assembly'])]

    # === SCRIPT-BASED CATEGORIES ===
    # Latin script languages (most common)
    latin_script = df[df["script"] == "Latn"]["language"].tolist()
    
    # Cyrillic script languages
    cyrillic_script = df[df["script"] == "Cyrl"]["language"].tolist()
    
    # Asian scripts (CJK + Indic + SE Asian)
    asian_scripts = ["Hani", "Jpan", "Hang", "Thai", "Laoo", "Khmr", "Mymr", 
                    "Deva", "Beng", "Guru", "Taml", "Telu", "Knda", "Mlym", "Gujr", "Orya", "Sinh"]
    asian_scripts_langs = df[df["script"].isin(asian_scripts)]["language"].tolist()
    
    # Arabic script languages
    arabic_script = df[df["script"] == "Arab"]["language"].tolist()

    # === LANGUAGE FAMILY CATEGORIES (based on language names and knowledge) ===
    # Indo-European languages (pattern matching)
    indo_european_patterns = ['russian', 'german', 'spanish', 'french', 'italian', 'portuguese', 
                             'polish', 'dutch', 'czech', 'persian', 'romanian', 'ukrainian', 
                             'norwegian', 'greek', 'swedish', 'danish', 'hindi', 'bengali', 
                             'lithuanian', 'bosnian', 'hebrew', 'slovenian', 'estonian', 
                             'catalan', 'latvian', 'albanian', 'urdu', 'marathi', 'nepali',
                             'belarusian', 'icelandic', 'armenian', 'panjabi', 'gaelic', 'welsh',
                             'irish', 'pashto', 'kurdish', 'english']
    indo_european = [lang for lang in all_languages 
                    if any(pattern in lang.lower() for pattern in indo_european_patterns)]
    
    # Sino-Tibetan languages
    sino_tibetan_patterns = ['chinese', 'mandarin', 'japanese', 'tibetan', 'burmese', 'karen']
    sino_tibetan = [lang for lang in all_languages 
                   if any(pattern in lang.lower() for pattern in sino_tibetan_patterns)]
    
    # Niger-Congo languages (African)
    niger_congo_patterns = ['swahili', 'yoruba', 'igbo', 'zulu', 'xhosa', 'shona', 'kikuyu', 
                           'lingala', 'wolof', 'bambara', 'fula', 'akan', 'ewe', 'kinyarwanda']
    niger_congo = [lang for lang in all_languages 
                  if any(pattern in lang.lower() for pattern in niger_congo_patterns)]

    # === GEOGRAPHIC/REGIONAL CATEGORIES ===
    # European languages (combining script + linguistic knowledge)
    european_patterns = ['russian', 'german', 'spanish', 'french', 'italian', 'portuguese', 
                        'polish', 'dutch', 'czech', 'romanian', 'ukrainian', 'norwegian', 
                        'greek', 'swedish', 'danish', 'finnish', 'bulgarian', 'slovak', 
                        'croatian', 'lithuanian', 'bosnian', 'slovenian', 'estonian', 
                        'catalan', 'latvian', 'albanian', 'belarusian', 'icelandic', 
                        'hungarian', 'basque', 'galician', 'welsh', 'irish', 'gaelic',
                        'breton', 'corsican', 'sardinian', 'maltese']
    european_langs = [lang for lang in natural_languages 
                     if any(pattern in lang.lower() for pattern in european_patterns)]
    
    # Major world languages (most speakers/important) - top 20 by size
    major_world = english_languages + natural_languages[:19] if len(natural_languages) >= 19 else english_languages + natural_languages

    return {
        # Core categories (always include)
        "All Languages": ordered_all_languages,
        "Top 10": top_10,
        "11-40": tier_11_40,
        "41-100": tier_41_100,
        
        # Language type categories  
        "Natural Languages": natural_languages,
        "Programming Languages": programming_languages,
        "Core Programming": core_programming,
        "Systems Programming": systems_programming,
        
        # Script-based categories
        "Latin Script": latin_script,
        "Cyrillic Script": cyrillic_script,
        "Asian Scripts": asian_scripts_langs,
        "Arabic Script": arabic_script,
        
        # Regional/family categories
        "European Languages": european_langs,
        "Major World Languages": major_world,
        
        # Special categories
        "English": english_languages,
    }


def load_all_results(results_dir: str = RESULTS_DIR) -> Dict[str, Any]:
    """Load all JSON result files from the results directory."""
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        return results

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results[json_file.stem] = data
        except Exception as e:
            st.warning(f"Could not load {json_file}: {e}")

    return results


def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert results dictionary to a pandas DataFrame for analysis."""
    rows = []

    for tokenizer_key, result in results.items():
        tokenizer_name = result.get("tokenizer", tokenizer_key)
        benchmark_size = result.get("benchmark_size_mb", 1.0)
        timestamp = result.get("timestamp", 0)
        vocab_size = result.get("vocab_size", None)

        for lang_key, lang_data in result.get("languages", {}).items():
            lang_info = lang_data["language_info"]
            metrics = lang_data["metrics"]

            rows.append(
                {
                    "tokenizer_key": tokenizer_key,
                    "tokenizer_name": tokenizer_name,
                    "language": lang_info["name"],
                    "iso_code": lang_info["iso_code"],
                    "script": lang_info["script"],
                    "lang_key": lang_key,
                    "bytes_per_token": metrics["bytes_per_token"],
                    "total_bytes": metrics["total_bytes"],
                    "total_tokens": metrics["total_tokens"],
                    "unique_tokens": metrics["unique_tokens"],
                    "vocab_size": vocab_size,
                    "benchmark_size_mb": benchmark_size,
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp)
                    if timestamp
                    else None,
                }
            )

    return pd.DataFrame(rows)


def get_global_tokenizer_order(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> List[str]:
    """Get consistent global ordering of tokenizers for all charts."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]
    return sorted(filtered_df["tokenizer_name"].unique())


def prepare_chart_data(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> pd.DataFrame:
    """Prepare and sort dataframe with consistent tokenizer ordering for charts."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)].copy()

    # Apply consistent categorical ordering
    global_order = get_global_tokenizer_order(df, selected_tokenizers)
    filtered_df["tokenizer_name"] = pd.Categorical(
        filtered_df["tokenizer_name"], categories=global_order, ordered=True
    )

    # Sort to ensure consistent color assignment across all charts
    return filtered_df.sort_values("tokenizer_name")


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    y_label: str,
    x_label: str = "Language",
) -> go.Figure:
    """Create a standardized bar chart with consistent styling."""
    fig = px.bar(
        df,
        x=x,
        y=y,
        color="tokenizer_name",
        title=title,
        labels={y: y_label, x: x_label},
        barmode="group",
        height=CHART_HEIGHT,
    )

    fig.update_layout(xaxis_tickangle=-45, legend=LEGEND_CONFIG)
    return fig


def create_efficiency_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a bar chart comparing bytes per token across languages."""
    chart_data = prepare_chart_data(df, selected_tokenizers)
    return create_bar_chart(
        chart_data,
        x="language",
        y="bytes_per_token",
        title="Tokenization Efficiency (Bytes per Token)",
        y_label="Bytes per Token (Higher = More Efficient)",
    )


def create_coverage_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a bar chart comparing unique tokens (vocabulary coverage)."""
    chart_data = prepare_chart_data(df, selected_tokenizers)
    return create_bar_chart(
        chart_data,
        x="language",
        y="unique_tokens",
        title="Vocabulary Coverage (Unique Tokens Used)",
        y_label="Unique Tokens (Higher = Better Coverage)",
    )


def create_vocab_efficiency_scatter(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a scatter plot showing average efficiency vs vocabulary size."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]

    # Calculate average efficiency per tokenizer
    summary_data = []
    for tokenizer in selected_tokenizers:
        tokenizer_df = filtered_df[filtered_df["tokenizer_key"] == tokenizer]
        if not tokenizer_df.empty:
            vocab_size = tokenizer_df["vocab_size"].iloc[0]
            if pd.notna(vocab_size):  # Only include if vocab_size is available
                avg_efficiency = tokenizer_df["bytes_per_token"].mean()
                tokenizer_name = tokenizer_df["tokenizer_name"].iloc[0]

                summary_data.append(
                    {
                        "tokenizer_name": tokenizer_name,
                        "vocab_size": vocab_size,
                        "avg_efficiency": avg_efficiency,
                        "languages_count": len(tokenizer_df),
                    }
                )

    if not summary_data:
        # Return empty figure if no vocab size data available
        fig = go.Figure()
        fig.add_annotation(
            text="Vocabulary size data not available<br>Run benchmarks with updated script",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Average Efficiency vs Vocabulary Size",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=CHART_HEIGHT,
        )
        return fig

    summary_df = pd.DataFrame(summary_data)

    # Apply consistent ordering
    global_order = get_global_tokenizer_order(df, selected_tokenizers)
    summary_df["tokenizer_name"] = pd.Categorical(
        summary_df["tokenizer_name"], categories=global_order, ordered=True
    )
    summary_df = summary_df.sort_values("tokenizer_name")

    fig = px.scatter(
        summary_df,
        x="vocab_size",
        y="avg_efficiency",
        color="tokenizer_name",
        size="languages_count",
        hover_data=["languages_count"],
        title="Average Efficiency vs Vocabulary Size",
        labels={
            "vocab_size": "Vocabulary Size (tokens)",
            "avg_efficiency": "Average Efficiency (bytes/token)",
            "languages_count": "Languages Tested",
        },
        height=CHART_HEIGHT,
    )

    fig.update_layout(
        xaxis=dict(type="log", title="Vocabulary Size (log scale)"),
        yaxis=dict(title="Average Efficiency (bytes/token)"),
        showlegend=True,
    )

    return fig


def create_summary_table(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> pd.DataFrame:
    """Create a summary table with rankings."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]
    summary_rows = []

    for tokenizer in selected_tokenizers:
        tokenizer_df = filtered_df[filtered_df["tokenizer_key"] == tokenizer]
        tokenizer_name = tokenizer_df["tokenizer_name"].iloc[0]

        vocab_size = (
            tokenizer_df["vocab_size"].iloc[0]
            if not tokenizer_df["vocab_size"].isna().iloc[0]
            else None
        )

        summary_rows.append(
            {
                "Tokenizer": tokenizer_name,
                "Vocab Size": f"{vocab_size:,}" if vocab_size else "N/A",
                "Avg Efficiency (bytes/token)": f"{tokenizer_df['bytes_per_token'].mean():.3f}",
                "Avg Coverage (unique tokens)": f"{tokenizer_df['unique_tokens'].mean():.0f}",
                "Languages Tested": len(tokenizer_df),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Sort by average efficiency (descending)
    summary_df["_sort_key"] = summary_df["Avg Efficiency (bytes/token)"].astype(float)
    summary_df = summary_df.sort_values("_sort_key", ascending=False).drop(
        "_sort_key", axis=1
    )

    return summary_df


def render_global_sidebar_controls(
    df: pd.DataFrame, language_categories: Dict[str, List[str]]
) -> tuple[List[str], List[str]]:
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
        "41-100": "Remaining natural languages",
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

    # Update session state
    if selected_tokenizers != st.session_state.selected_tokenizers:
        st.session_state.selected_tokenizers = selected_tokenizers

    # Global language selection
    all_languages = sorted(df["language"].unique())
    selected_languages = st.sidebar.multiselect(
        "Filter Languages",
        all_languages,
        default=st.session_state.selected_languages,
        key="global_languages",
    )

    # Update session state
    if selected_languages != st.session_state.selected_languages:
        st.session_state.selected_languages = selected_languages

    st.sidebar.divider()

    # Quick stats
    st.sidebar.metric(
        "Tokenizers Selected",
        len(selected_tokenizers),
        f"{len(available_tokenizers)} available",
    )
    st.sidebar.metric(
        "Languages Selected", len(selected_languages), f"{len(all_languages)} available"
    )

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

    # Selection summary with cleaner layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Languages",
            len(selected_languages),
            f"{len(programming_langs)}🔧 + {len(natural_langs)}🌍",
        )
    with col2:
        st.metric("Programming", len(programming_langs), "🔧 Code Languages")
    with col3:
        st.metric("Natural", len(natural_langs), "🌍 Human Languages")

    # Cleaner selection description
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
    chart_tab1, chart_tab2, chart_tab3, raw_tab = st.tabs(
        ["🚀 Efficiency", "🎯 Coverage", "📏 Efficiency Analysis", "🔍 Raw Data"]
    )

    with chart_tab1:
        st.write("#### Tokenization Efficiency")
        st.caption("Higher values = more efficient tokenization")
        efficiency_chart = create_efficiency_chart(display_df, selected_tokenizers)
        st.plotly_chart(efficiency_chart, use_container_width=True)

        # Add concise insights
        if len(programming_langs) > 0:
            st.info(
                f"💻 **{len(programming_langs)} coding languages** • Look for patterns in syntax complexity"
            )

    with chart_tab2:
        st.write("#### Vocabulary Coverage")
        st.caption("Higher values = better language support")
        coverage_chart = create_coverage_chart(display_df, selected_tokenizers)
        st.plotly_chart(coverage_chart, use_container_width=True)

        # Add concise insights
        if len(natural_langs) > 0:
            st.info(
                f"🌍 **{len(natural_langs)} human languages** • Higher coverage indicates better script support"
            )

    with chart_tab3:
        st.write("#### Efficiency vs Vocabulary Size")
        st.caption("How tokenizer size affects average efficiency across languages")

        # Show overall scatter plot
        vocab_scatter = create_vocab_efficiency_scatter(display_df, selected_tokenizers)
        st.plotly_chart(vocab_scatter, use_container_width=True)

        # Category-specific scatter plots
        if len(programming_langs) > 0 and len(natural_langs) > 0:
            st.write("##### By Language Type")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**🌍 Natural Languages Only**")
                natural_df = display_df[display_df["language"].isin(natural_langs)]
                if not natural_df.empty:
                    natural_scatter = create_vocab_efficiency_scatter(
                        natural_df, selected_tokenizers
                    )
                    natural_scatter.update_layout(
                        title="Natural Languages Efficiency", height=400
                    )
                    st.plotly_chart(natural_scatter, use_container_width=True)
                    st.caption(f"Analysis of {len(natural_langs)} natural languages")
                else:
                    st.info("No natural language data available")

            with col2:
                st.write("**💻 Programming Languages Only**")
                prog_df = display_df[display_df["language"].isin(programming_langs)]
                if not prog_df.empty:
                    prog_scatter = create_vocab_efficiency_scatter(
                        prog_df, selected_tokenizers
                    )
                    prog_scatter.update_layout(
                        title="Programming Languages Efficiency", height=400
                    )
                    st.plotly_chart(prog_scatter, use_container_width=True)
                    st.caption(
                        f"Analysis of {len(programming_langs)} programming languages"
                    )
                else:
                    st.info("No programming language data available")

            st.info(
                "🔬 **Mixed Analysis**: Compare how tokenizer efficiency scales with vocabulary size for code vs. natural language."
            )

        elif len(programming_langs) > 0:
            st.info(
                "⚙️ **Code Analysis**: Examine how tokenizer vocabulary size affects code tokenization efficiency. Larger vocabularies may better handle diverse syntax patterns."
            )
        elif len(natural_langs) > 0:
            st.info(
                "🌐 **Language Analysis**: Observe how vocabulary size correlates with multilingual efficiency. Larger tokenizers typically handle more languages effectively."
            )

    with raw_tab:
        st.write("#### Complete Dataset")
        st.dataframe(display_df, use_container_width=True)

        # Add download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"tokka_bench_data_{len(selected_tokenizers)}tokenizers_{len(selected_languages)}languages.csv",
            mime="text/csv",
        )


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
        st.info("Example: `uv run cli/benchmark.py tokenizer=openai-community/gpt2`")
        return

    df = results_to_dataframe(results)

    # Detect language types and categories
    language_categories = detect_language_types(df)

    # Render global sidebar controls
    selected_tokenizers, selected_languages = render_global_sidebar_controls(
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
