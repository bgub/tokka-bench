# Tokka-Bench

A tokenizer benchmarking tool that compares different tokenizers across multiple languages using real-world data from FineWeb-2.

## Overview

Tokka-bench evaluates how efficiently different tokenizers handle text from various language families using:

- **Bytes per token**: How many UTF-8 bytes each token represents (higher = more efficient)
- **Unique tokens**: How many different tokens were used (reveals vocabulary coverage)
- Real data from **141 languages total**: 100 natural languages (FineWeb-2) + 40 programming languages (StarCoder) + 1 English (FineWeb)

### What the Metrics Reveal

- **Bytes per token**: Overall compression efficiency
- **Unique tokens**: Language coverage quality
  - Well-supported languages use 800-1000+ unique tokens
  - Poorly-supported languages fall back to byte-level encoding (200-400 tokens)
  - Reveals which scripts/languages the tokenizer handles well

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tokka-bench

# Install dependencies with uv
uv sync
```

## Usage

### Basic Usage (Fast Engine)

```bash
# Benchmark GPT-2 tokenizer with 2MB samples per language
uv run benchmark tokenizer=openai-community/gpt2

# Quick test with smaller samples
uv run benchmark tokenizer=Xenova/gpt-4 sample_size=0.1

# Custom output filename
uv run benchmark tokenizer=meta-llama/Meta-Llama-3-8B output_name=llama-results
```

### Parameters

- `tokenizer`: HuggingFace model name (required)
- `sample_size`: Size in MB to test per language (default: 2.0)
- `output_name`: Custom filename for results (optional)

## Visualization Dashboard

After running benchmarks, launch the interactive web dashboard to compare results:

```bash
# Launch interactive comparison dashboard
uv run streamlit run cli/visualize.py
```

The dashboard provides:

- **Summary rankings** - Compare tokenizers across all metrics
- **Interactive charts** - Filter by tokenizer and language
- **Efficiency comparison** - Bytes per token across languages
- **Coverage analysis** - Unique tokens used (vocabulary coverage)
- **Scatter plots** - Efficiency vs coverage relationships
- **Raw data access** - Detailed metrics for analysis

### Dashboard Features

- 🎛️ **Interactive filters** - Select tokenizers and languages to compare
- 📊 **Multiple visualizations** - Bar charts, scatter plots, summary tables
- 🚀 **Real-time updates** - Change selections and see instant updates
- 📈 **Rankings** - Automatic sorting by efficiency and coverage
- 🔍 **Raw data** - Access to underlying benchmark data

## Test Languages

Benchmarks run on **141 languages total** from multiple sources, providing comprehensive coverage across natural and programming languages:

- **1 English** from FineWeb sample-10BT
- **Top 100 natural languages** by FineWeb-2 dataset size
- **Top 40 programming languages** from StarCoder dataset

## Natural Language Coverage (Top 100 FineWeb-2)

> **Note**: We benchmark 100 natural languages from FineWeb-2 plus English separately from FineWeb sample-10BT (which provides higher quality English data). This gives us comprehensive coverage while ensuring the highest quality English benchmark.

### Latin Script (17 languages)

1. **German** - 640.76GB
2. **Spanish** - 554.08GB
3. **French** - 476.55GB
4. **Italian** - 305.96GB
5. **Portuguese** - 246.33GB
6. **Polish** - 193.34GB
7. **Dutch** - 162.98GB
8. **Indonesian** - 134.84GB
9. **Turkish** - 116.64GB
10. **Czech** - 98.30GB
11. **Hungarian** - 85.72GB
12. **Romanian** - 81.30GB
13. **Vietnamese** - 78.95GB
14. **Norwegian Bokmål** - 74.48GB
15. **Swedish** - 63.27GB
16. **Danish** - 63.04GB
17. **Finnish** - 56.79GB
18. **Slovak** - 40.43GB
19. **Croatian** - 32.91GB

### Cyrillic Script (4 languages)

1. **Russian** - 1.65TB _(largest dataset)_
2. **Ukrainian** - 77.40GB
3. **Bulgarian** - 43.04GB

### Other Scripts (9 languages)

- **Mandarin Chinese** (Hani) - 1.34TB _(2nd largest)_
- **Japanese** (Jpan) - 636.71GB
- **Korean** (Hang) - 94.73GB
- **Standard Arabic** (Arab) - 94.52GB
- **Persian** (Arab) - 85.16GB
- **Thai** (Thai) - 70.86GB
- **Modern Greek** (Grek) - 68.91GB
- **Hindi** (Deva) - 30.59GB

## Programming Language Coverage (Top 40 StarCoder)

The benchmark also includes the **40 most popular programming languages** from the StarCoder dataset:

1. **Python** - Most popular language
2. **JavaScript** - Web development
3. **Java** - Enterprise applications
4. **C** - Systems programming
5. **C++** - Performance-critical applications
6. **C#** - .NET ecosystem
7. **PHP** - Web backends
8. **TypeScript** - Type-safe JavaScript
9. **Go** - Modern systems language
10. **Rust** - Memory-safe systems language
    ... and 30 more including **HTML, CSS, SQL, Shell, Ruby, YAML, JSON, Markdown, Docker**, and specialized languages like **CUDA, Verilog, Solidity, Fortran**, and more.

## Comprehensive Language Analysis

This expanded coverage reveals tokenizer performance across:

**Natural Languages:**

- **Multiple scripts**: Latin, Cyrillic, Arabic, CJK, Thai, Greek, Devanagari
- **Language families**: Indo-European, Sino-Tibetan, Turkic, Austronesian, etc.
- **Writing systems**: Alphabetic, logographic, syllabic, abjad

**Programming Languages:**

- **Popular languages**: Python, JavaScript, Java, C/C++
- **Modern languages**: Rust, Go, TypeScript, Kotlin
- **Specialized domains**: CUDA (GPU), Solidity (blockchain), Verilog (hardware)
- **Markup/Config**: HTML, CSS, YAML, JSON, Dockerfile

## Output

Results are saved as JSON files in `data/results/` containing:

- Tokenizer metadata and timestamp
- Per-language efficiency metrics (bytes per token, unique tokens)
- Total bytes and tokens processed

Example output shows Spanish typically most efficient (~3.01 bytes/token), followed by German (~2.71), Japanese (~2.25), Russian (~1.71), and Chinese (~1.40) for GPT-2.

Unique token counts reveal vocabulary coverage: German (~1000+ tokens) vs Russian (~150 tokens) showing Cyrillic script limitations.

## Architecture

```
tokka-bench/
├── src/tokka_bench/          # Core benchmarking logic
│   ├── fast_benchmark.py     # Fast benchmark implementation (default)
│   ├── tokenizer.py          # Universal tokenizer wrapper
│   ├── metrics.py            # Metrics and global analysis
│   └── fineweb-2-languages.csv # Language metadata
├── src/tokka_bench/cli.py    # Command-line interface (fast by default)
└── data/results/             # JSON output files
```

## Dependencies

- **transformers**: HuggingFace tokenizers
- **datasets**: FineWeb-2 data loading
- **pandas**: Language metadata processing
- **omegaconf**: CLI configuration

## Contributing

We welcome contributions to tokka-bench! Whether you're fixing bugs, adding new features, improving documentation, or adding support for new tokenizers, your help is appreciated.

### Quick Start for Contributors

1. **Fork the repository** and clone your fork
2. **Set up the development environment**: `uv sync`
3. **Make your changes** following our coding standards
4. **Test your changes** with existing benchmark data
5. **Submit a pull request** with a clear description

For detailed contribution guidelines, please see [CONTRIBUTING.md](CONTRIBUTING.md).

### Ways to Contribute

- 🐛 **Bug reports**: Found an issue? [Open an issue](../../issues/new)
- 🚀 **Feature requests**: Have an idea? [Start a discussion](../../discussions)
- 📖 **Documentation**: Improve our docs and examples
- 🔧 **Code contributions**: Fix bugs or add features
- 🧪 **Testing**: Add test coverage for better reliability
- 🌍 **Language support**: Help improve tokenizer coverage for different languages
