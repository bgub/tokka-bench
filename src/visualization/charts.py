"""
Chart creation utilities for the visualization dashboard.
"""

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .constants import CHART_HEIGHT, LEGEND_CONFIG


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

        # Safety check: skip if no data for this tokenizer
        if tokenizer_df.empty:
            continue

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
    if not summary_df.empty:
        summary_df["_sort_key"] = summary_df["Avg Efficiency (bytes/token)"].astype(
            float
        )
        summary_df = summary_df.sort_values("_sort_key", ascending=False).drop(
            "_sort_key", axis=1
        )

    return summary_df
