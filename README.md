# Tokka-Bench

A tokenizer benchmarking tool that compares different tokenizers across multiple languages using real-world data from FineWeb-2.

## Overview

Tokka-bench evaluates how efficiently different tokenizers handle text from various language families using:

- **Bytes per token**: How many UTF-8 bytes each token represents (higher = more efficient)
- **Unique tokens**: How many different tokens were used (reveals vocabulary coverage)
- Real data from the top 5 languages by FineWeb-2 dataset size

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

### Basic Usage

```bash
# Benchmark GPT-2 tokenizer with 1MB samples per language
uv run cli/benchmark.py tokenizer=openai-community/gpt2

# Quick test with smaller samples
uv run cli/benchmark.py tokenizer=Xenova/gpt-4 sample_size=0.1

# Custom output filename
uv run cli/benchmark.py tokenizer=meta-llama/Meta-Llama-3-8B output_name=llama-results
```

### Parameters

- `tokenizer`: HuggingFace model name (required)
- `sample_size`: Size in MB to test per language (default: 1.0)
- `output_name`: Custom filename for results (optional)

## Test Languages

Benchmarks run on the top 5 languages by FineWeb-2 size:

1. **Russian (rus-Cyrl)** - 1.65TB
2. **Mandarin Chinese (cmn-Hani)** - 1.34TB
3. **German (deu-Latn)** - 640.76GB
4. **Japanese (jpn-Jpan)** - 636.71GB
5. **Spanish (spa-Latn)** - 554.08GB

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
│   ├── benchmark.py          # Main benchmark implementation
│   └── fineweb-2-languages.csv # Language metadata
├── cli/benchmark.py          # Command-line interface
└── data/results/             # JSON output files
```

## Dependencies

- **transformers**: HuggingFace tokenizers
- **datasets**: FineWeb-2 data loading
- **pandas**: Language metadata processing
- **omegaconf**: CLI configuration
