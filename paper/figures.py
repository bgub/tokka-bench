#!/usr/bin/env python3
"""Generate publication-quality figures for the Tokka-Bench paper.

Reads JSON results from data/results/ and produces PDF vector figures
in paper/figures/.

Usage:
    python paper/figures.py
"""

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
# Consistent color palette across all figures (7 tokenizers)
TOKENIZER_COLORS = {
    "GPT-2": "#d62728",
    "GPT-4": "#ff7f0e",
    "gpt-oss": "#2ca02c",
    "Llama 3.1": "#1f77b4",
    "Gemma 3": "#9467bd",
    "Qwen3": "#8c564b",
    "Kimi K2": "#e377c2",
}
TOKENIZER_ORDER = list(TOKENIZER_COLORS.keys())

# Highlight languages: maximal script diversity
HIGHLIGHT_LANGS = {
    "eng-Latn": "English",
    "fra-Latn": "French",
    "rus-Cyrl": "Russian",
    "arb-Arab": "Arabic",
    "hin-Deva": "Hindi",
    "cmn-Hani": "Chinese",
    "jpn-Jpan": "Japanese",
    "kor-Hang": "Korean",
    "khm-Khmr": "Khmer",
    "tha-Thai": "Thai",
}

PROG_LANGS = {
    "python": "Python",
    "javascript": "JavaScript",
    "java": "Java",
    "cpp": "C++",
    "go": "Go",
    "rust": "Rust",
    "typescript": "TypeScript",
    "ruby": "Ruby",
}


def _apply_style() -> None:
    """Set global matplotlib parameters for publication quality."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
    })
    sns.set_style("whitegrid", {
        "axes.edgecolor": "0.2",
        "grid.linewidth": 0.4,
    })


# ---------------------------------------------------------------------------
# Data loading (adapted from src/visualization/data.py without Streamlit)
# ---------------------------------------------------------------------------
def load_all_results() -> Dict[str, Any]:
    results = {}
    for json_file in RESULTS_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            results[json_file.stem] = json.load(f)
    return results


def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for tokenizer_key, result in results.items():
        # Use the file stem (e.g. "GPT-4") as the display name
        tokenizer_name = tokenizer_key
        vocab_size = result.get("vocab_size", None)
        for lang_key, lang_data in result.get("languages", {}).items():
            lang_info = lang_data["language_info"]
            metrics = lang_data["metrics"]
            row = {
                "tokenizer_key": tokenizer_key,
                "tokenizer_name": tokenizer_name,
                "language": lang_info["name"],
                "iso_code": lang_info.get("iso_code", ""),
                "script": lang_info.get("script", ""),
                "source": lang_info.get("source", ""),
                "lang_key": lang_key,
                "bytes_per_token": metrics["bytes_per_token"],
                "total_tokens": metrics["total_tokens"],
                "unique_tokens": metrics["unique_tokens"],
                "vocab_size": vocab_size,
            }
            if "subword_fertility" in metrics:
                row["subword_fertility"] = metrics["subword_fertility"]
            if "word_split_pct" in metrics:
                row["word_split_pct"] = metrics["word_split_pct"]
            elif "continued_word_rate" in metrics:
                row["word_split_pct"] = metrics["continued_word_rate"]
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1: Efficiency (bytes per token) – grouped bar chart
# ---------------------------------------------------------------------------
def fig_efficiency(df: pd.DataFrame) -> None:
    sub = df[df["lang_key"].isin(HIGHLIGHT_LANGS)]
    sub = sub[sub["tokenizer_name"].isin(TOKENIZER_ORDER)]

    # Pivot: rows = lang_key, cols = tokenizer_name
    pivot = sub.pivot_table(
        index="lang_key", columns="tokenizer_name", values="bytes_per_token"
    )
    # Order languages and tokenizers
    lang_order = [k for k in HIGHLIGHT_LANGS if k in pivot.index]
    pivot = pivot.loc[lang_order, [t for t in TOKENIZER_ORDER if t in pivot.columns]]

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    x = np.arange(len(lang_order))
    n_tok = len(pivot.columns)
    width = 0.8 / n_tok

    for i, tok in enumerate(pivot.columns):
        vals = pivot[tok].values
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=tok,
            color=TOKENIZER_COLORS[tok],
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_ylabel("Bytes per Token")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [HIGHLIGHT_LANGS[k] for k in lang_order], rotation=45, ha="right"
    )
    ax.legend(
        ncol=2, loc="upper left", framealpha=0.9, edgecolor="0.8",
        borderpad=0.3, handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
    )
    ax.set_title("Tokenization Efficiency Across Languages")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "efficiency.pdf")
    plt.close(fig)
    print("  -> efficiency.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Coverage (unique tokens) – grouped bar chart
# ---------------------------------------------------------------------------
def fig_coverage(df: pd.DataFrame) -> None:
    sub = df[df["lang_key"].isin(HIGHLIGHT_LANGS)]
    sub = sub[sub["tokenizer_name"].isin(TOKENIZER_ORDER)]

    pivot = sub.pivot_table(
        index="lang_key", columns="tokenizer_name", values="unique_tokens"
    )
    lang_order = [k for k in HIGHLIGHT_LANGS if k in pivot.index]
    pivot = pivot.loc[lang_order, [t for t in TOKENIZER_ORDER if t in pivot.columns]]

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    x = np.arange(len(lang_order))
    n_tok = len(pivot.columns)
    width = 0.8 / n_tok

    for i, tok in enumerate(pivot.columns):
        vals = pivot[tok].values
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=tok,
            color=TOKENIZER_COLORS[tok],
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_ylabel("Unique Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [HIGHLIGHT_LANGS[k] for k in lang_order], rotation=45, ha="right"
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(
        ncol=2, loc="upper left", framealpha=0.9, edgecolor="0.8",
        borderpad=0.3, handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
    )
    ax.set_title("Vocabulary Coverage Across Languages")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "coverage.pdf")
    plt.close(fig)
    print("  -> coverage.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Fertility heatmap – full width
# ---------------------------------------------------------------------------
def fig_fertility_heatmap(df: pd.DataFrame) -> None:
    # Use a broader set of languages for the heatmap
    heatmap_langs = {
        "eng-Latn": "English",
        "fra-Latn": "French",
        "deu-Latn": "German",
        "spa-Latn": "Spanish",
        "por-Latn": "Portuguese",
        "ita-Latn": "Italian",
        "ces-Latn": "Czech",
        "pol-Latn": "Polish",
        "rus-Cyrl": "Russian",
        "ukr-Cyrl": "Ukrainian",
        "arb-Arab": "Arabic",
        "hin-Deva": "Hindi",
        "cmn-Hani": "Chinese",
        "jpn-Jpan": "Japanese",
        "kor-Hang": "Korean",
        "khm-Khmr": "Khmer",
        "tha-Thai": "Thai",
    }

    sub = df[df["lang_key"].isin(heatmap_langs)]
    sub = sub[sub["tokenizer_name"].isin(TOKENIZER_ORDER)]
    if "subword_fertility" not in sub.columns:
        print("  [skip] fertility heatmap – subword_fertility not in data")
        return

    pivot = sub.pivot_table(
        index="lang_key", columns="tokenizer_name", values="subword_fertility"
    )
    lang_order = [k for k in heatmap_langs if k in pivot.index]
    tok_order = [t for t in TOKENIZER_ORDER if t in pivot.columns]
    pivot = pivot.loc[lang_order, tok_order]
    pivot.index = [heatmap_langs[k] for k in lang_order]

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        center=1.0,
        vmin=0.5,
        vmax=3.5,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 6.5},
        cbar_kws={"label": "Subword Fertility", "shrink": 0.8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Subword Fertility: Tokens per Language Unit")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fertility_heatmap.pdf")
    plt.close(fig)
    print("  -> fertility_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Code convergence – programming languages
# ---------------------------------------------------------------------------
def fig_code_convergence(df: pd.DataFrame) -> None:
    # Programming languages have source == "starcoder"
    code_df = df[df["source"] == "starcoder"]
    code_df = code_df[code_df["iso_code"].isin(PROG_LANGS)]

    # Show GPT-2 vs 3 modern tokenizers for contrast
    show_tokenizers = ["GPT-2", "Llama 3.1", "Kimi K2", "gpt-oss"]
    code_df = code_df[code_df["tokenizer_name"].isin(show_tokenizers)]

    pivot = code_df.pivot_table(
        index="iso_code", columns="tokenizer_name", values="bytes_per_token"
    )
    lang_order = [k for k in PROG_LANGS if k in pivot.index]
    tok_order = [t for t in show_tokenizers if t in pivot.columns]
    pivot = pivot.loc[lang_order, tok_order]

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(lang_order))
    n_tok = len(tok_order)
    width = 0.8 / n_tok

    for i, tok in enumerate(tok_order):
        vals = pivot[tok].values
        ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=tok,
            color=TOKENIZER_COLORS[tok],
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_ylabel("Bytes per Token")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [PROG_LANGS[k] for k in lang_order], rotation=45, ha="right"
    )
    ax.legend(
        ncol=2, loc="upper left", framealpha=0.9, edgecolor="0.8",
        borderpad=0.3, handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
    )
    ax.set_title("Code Tokenization: GPT-2 vs. Modern Tokenizers")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "code_convergence.pdf")
    plt.close(fig)
    print("  -> code_convergence.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Vocab size vs. mean efficiency – scatter
# ---------------------------------------------------------------------------
def fig_vocab_vs_efficiency(df: pd.DataFrame) -> None:
    # Filter to natural languages only (non-code)
    nat_df = df[df["source"] != "starcoder"]
    nat_df = nat_df[nat_df["tokenizer_name"].isin(TOKENIZER_ORDER)]

    grouped = nat_df.groupby("tokenizer_name").agg(
        vocab_size=("vocab_size", "first"),
        mean_bpt=("bytes_per_token", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for _, row in grouped.iterrows():
        tok = row["tokenizer_name"]
        ax.scatter(
            row["vocab_size"],
            row["mean_bpt"],
            color=TOKENIZER_COLORS.get(tok, "gray"),
            s=50,
            zorder=5,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.annotate(
            tok,
            (row["vocab_size"], row["mean_bpt"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=6.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Mean Bytes per Token")
    ax.set_title("Vocabulary Size vs. Efficiency")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vocab_vs_efficiency.pdf")
    plt.close(fig)
    print("  -> vocab_vs_efficiency.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    print("Loading results...")
    results = load_all_results()
    print(f"  Loaded {len(results)} tokenizer result files")

    df = results_to_dataframe(results)
    print(f"  Built DataFrame: {len(df)} rows, {df['tokenizer_name'].nunique()} tokenizers")

    print("Generating figures...")
    fig_efficiency(df)
    fig_coverage(df)
    fig_fertility_heatmap(df)
    fig_code_convergence(df)
    fig_vocab_vs_efficiency(df)
    print("Done.")


if __name__ == "__main__":
    main()
