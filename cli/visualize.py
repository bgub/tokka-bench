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


def load_all_results(results_dir: str = "data/results") -> Dict[str, Any]:
    """Load all JSON result files from the results directory."""
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        return results

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Use filename (without .json) as key
                key = json_file.stem
                results[key] = data
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
                    "benchmark_size_mb": benchmark_size,
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp)
                    if timestamp
                    else None,
                }
            )

    return pd.DataFrame(rows)


def create_efficiency_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a bar chart comparing bytes per token across languages."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]

    fig = px.bar(
        filtered_df,
        x="language",
        y="bytes_per_token",
        color="tokenizer_name",
        title="Tokenization Efficiency (Bytes per Token)",
        labels={
            "bytes_per_token": "Bytes per Token (Higher = More Efficient)",
            "language": "Language",
        },
        barmode="group",
        height=500,
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_coverage_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a bar chart comparing unique tokens (vocabulary coverage)."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]

    fig = px.bar(
        filtered_df,
        x="language",
        y="unique_tokens",
        color="tokenizer_name",
        title="Vocabulary Coverage (Unique Tokens Used)",
        labels={
            "unique_tokens": "Unique Tokens (Higher = Better Coverage)",
            "language": "Language",
        },
        barmode="group",
        height=500,
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_scatter_plot(df: pd.DataFrame, selected_tokenizers: List[str]) -> go.Figure:
    """Create a scatter plot showing efficiency vs coverage."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]

    fig = px.scatter(
        filtered_df,
        x="unique_tokens",
        y="bytes_per_token",
        color="tokenizer_name",
        size="total_tokens",
        hover_data=["language", "script"],
        title="Efficiency vs Coverage (size = total tokens)",
        labels={
            "unique_tokens": "Unique Tokens (Coverage)",
            "bytes_per_token": "Bytes per Token (Efficiency)",
        },
        height=500,
    )

    return fig


def create_summary_table(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> pd.DataFrame:
    """Create a summary table with rankings."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]

    # Calculate rankings for each metric
    summary_rows = []

    for tokenizer in selected_tokenizers:
        tokenizer_df = filtered_df[filtered_df["tokenizer_key"] == tokenizer]
        tokenizer_name = tokenizer_df["tokenizer_name"].iloc[0]

        # Calculate averages and rankings
        avg_efficiency = tokenizer_df["bytes_per_token"].mean()
        avg_coverage = tokenizer_df["unique_tokens"].mean()
        total_tokens = tokenizer_df["total_tokens"].sum()

        summary_rows.append(
            {
                "Tokenizer": tokenizer_name,
                "Avg Efficiency (bytes/token)": f"{avg_efficiency:.3f}",
                "Avg Coverage (unique tokens)": f"{avg_coverage:.0f}",
                "Total Tokens Processed": f"{total_tokens:,}",
                "Languages Tested": len(tokenizer_df),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Sort by average efficiency (descending)
    summary_df["_sort_key"] = summary_df["Avg Efficiency (bytes/token)"].astype(float)
    summary_df = summary_df.sort_values("_sort_key", ascending=False)
    summary_df = summary_df.drop("_sort_key", axis=1)

    return summary_df


def create_script_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """Add script grouping for better visualization of 30+ languages."""

    def get_script_group(script):
        if script == "Latn":
            return "Latin"
        elif script == "Cyrl":
            return "Cyrillic"
        elif script in ["Arab"]:
            return "Arabic"
        elif script in ["Hani", "Jpan", "Hang"]:
            return "CJK"
        elif script in [
            "Deva",
            "Beng",
            "Guru",
            "Taml",
            "Telu",
            "Knda",
            "Mlym",
            "Gujr",
            "Orya",
        ]:
            return "Indic"
        elif script in ["Thai", "Laoo", "Khmr", "Mymr"]:
            return "Southeast Asian"
        elif script in ["Grek", "Armn", "Geor", "Hebr"]:
            return "Other European/Middle Eastern"
        else:
            return "Other"

    df = df.copy()
    df["script_group"] = df["script"].apply(get_script_group)
    return df


def create_script_comparison_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a chart comparing tokenizer performance by script group."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]

    # Add script grouping
    filtered_df = create_script_grouping(filtered_df)

    # Calculate averages by script group and tokenizer
    script_summary = (
        filtered_df.groupby(["tokenizer_name", "script_group"])
        .agg({"bytes_per_token": "mean", "unique_tokens": "mean"})
        .reset_index()
    )

    fig = px.bar(
        script_summary,
        x="script_group",
        y="bytes_per_token",
        color="tokenizer_name",
        title="Average Efficiency by Script Family",
        labels={
            "bytes_per_token": "Average Bytes per Token",
            "script_group": "Script Family",
        },
        barmode="group",
        height=500,
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Tokka-Bench Visualizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 Tokka-Bench: Tokenizer Comparison Dashboard")
    st.markdown(
        "Compare tokenizer efficiency and language coverage across multiple models"
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

    # Sidebar controls
    st.sidebar.header("🎛️ Controls")

    # Tokenizer selection
    available_tokenizers = sorted(df["tokenizer_key"].unique())
    selected_tokenizers = st.sidebar.multiselect(
        "Select Tokenizers to Compare",
        available_tokenizers,
        default=available_tokenizers,  # Select all by default
    )

    if not selected_tokenizers:
        st.warning("Please select at least one tokenizer to compare.")
        return

    # Language filtering
    available_languages = sorted(df["language"].unique())
    selected_languages = st.sidebar.multiselect(
        "Filter Languages", available_languages, default=available_languages
    )

    # Filter dataframe
    display_df = df[
        (df["tokenizer_key"].isin(selected_tokenizers))
        & (df["language"].isin(selected_languages))
    ]

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Tokenizers Loaded",
            len(selected_tokenizers),
            delta=f"{len(available_tokenizers)} total",
        )

    with col2:
        st.metric(
            "Languages Tested",
            len(selected_languages),
            delta=f"{len(available_languages)} total",
        )

    # Summary table
    st.header("📈 Summary Rankings")
    summary_df = create_summary_table(df, selected_tokenizers)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Charts
    st.header("📊 Detailed Comparisons")

    # Efficiency chart
    st.subheader("🚀 Tokenization Efficiency")
    efficiency_chart = create_efficiency_chart(display_df, selected_tokenizers)
    st.plotly_chart(efficiency_chart, use_container_width=True)

    # Coverage chart
    st.subheader("🎯 Vocabulary Coverage")
    coverage_chart = create_coverage_chart(display_df, selected_tokenizers)
    st.plotly_chart(coverage_chart, use_container_width=True)

    # Scatter plot
    st.subheader("⚡ Efficiency vs Coverage")
    scatter_chart = create_scatter_plot(display_df, selected_tokenizers)
    st.plotly_chart(scatter_chart, use_container_width=True)

    # Script family comparison
    st.subheader("🌍 Performance by Script Family")
    if len(selected_tokenizers) > 1:
        script_chart = create_script_comparison_chart(display_df, selected_tokenizers)
        st.plotly_chart(script_chart, use_container_width=True)
        st.markdown(
            "*This chart groups languages by script family to reveal broader patterns in tokenizer bias.*"
        )
    else:
        st.info("Select multiple tokenizers to see script family comparisons.")

    # Raw data
    with st.expander("🔍 Raw Data"):
        st.dataframe(display_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "*Generated by Tokka-Bench • Higher bytes/token = more efficient • Higher unique tokens = better coverage*"
    )


if __name__ == "__main__":
    main()
