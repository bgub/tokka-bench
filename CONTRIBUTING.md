# Contributing to Tokka-Bench

Thank you for your interest in contributing to tokka-bench! This guide will help you get started with contributing to our tokenizer benchmarking tool.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Fork and Clone

1. Fork the tokka-bench repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tokka-bench.git
   cd tokka-bench
   ```

## Development Environment

### Setting Up

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Verify the installation**:
   ```bash
   # Test basic functionality
   uv run cli/benchmark.py tokenizer=openai-community/gpt2 sample_size=0.01
   ```

3. **Install development tools** (optional but recommended):
   ```bash
   # Install pre-commit hooks for code quality
   uv add --dev pre-commit black isort flake8 mypy pytest
   ```

### Environment Variables

No special environment variables are required for basic development. However, you may want to:

- Set `HF_TOKEN` if working with private HuggingFace models
- Configure `STREAMLIT_SERVER_PORT` for dashboard development

## Project Structure

```
tokka-bench/
├── src/
│   ├── tokka_bench/
│   │   ├── __init__.py
│   │   └── benchmark.py          # Core benchmarking logic
│   ├── fineweb-2-languages.csv   # Language metadata
│   └── starcoderdata-dirs.csv     # Additional language data
├── cli/
│   ├── benchmark.py              # CLI interface
│   ├── dashboard.py              # Streamlit dashboard
│   └── visualize.py              # Visualization logic
├── data/
│   └── results/                  # Benchmark output files
├── pyproject.toml                # Project configuration
├── README.md                     # Main documentation
└── CONTRIBUTING.md               # This file
```

### Key Components

- **`src/tokka_bench/benchmark.py`**: Contains the `UniversalTokenizer` class and core benchmarking logic
- **`cli/benchmark.py`**: Command-line interface using OmegaConf for configuration
- **`cli/visualize.py`**: Streamlit-based interactive dashboard
- **Language data**: CSV files containing metadata about supported languages

## Making Changes

### Types of Contributions

1. **Bug Fixes**: Address issues in the existing codebase
2. **Feature Additions**: New functionality like additional metrics or tokenizers
3. **Performance Improvements**: Optimizations to benchmarking speed or memory usage
4. **Documentation**: Improvements to README, docstrings, or examples
5. **Testing**: Adding test coverage for better reliability

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes** following the guidelines below

3. **Test your changes** thoroughly:
   ```bash
   # Run a quick benchmark test
   uv run cli/benchmark.py tokenizer=openai-community/gpt2 sample_size=0.01
   
   # Test the dashboard
   uv run streamlit run cli/visualize.py
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add support for custom metrics"
   # or
   git commit -m "fix: handle empty dataset gracefully"
   ```

## Testing

### Manual Testing

Since the project currently doesn't have automated tests, please test your changes manually:

1. **Basic functionality**:
   ```bash
   # Test with different tokenizers
   uv run cli/benchmark.py tokenizer=openai-community/gpt2 sample_size=0.1
   uv run cli/benchmark.py tokenizer=microsoft/DialoGPT-medium sample_size=0.1
   ```

2. **Dashboard functionality**:
   ```bash
   uv run streamlit run cli/visualize.py
   ```

3. **Edge cases**:
   - Test with very small sample sizes
   - Test with models that might not be available
   - Test dashboard with no existing results

### Adding Tests (Encouraged!)

We welcome contributions that add test coverage:

1. **Create a `tests/` directory**
2. **Add unit tests** for core functions in `benchmark.py`
3. **Add integration tests** for the CLI interface
4. **Use pytest** as the testing framework

Example test structure:
```python
# tests/test_benchmark.py
import pytest
from src.tokka_bench.benchmark import UniversalTokenizer

def test_tokenizer_initialization():
    tokenizer = UniversalTokenizer("openai-community/gpt2")
    assert tokenizer is not None

def test_benchmark_with_small_sample():
    # Test benchmarking with minimal data
    pass
```

## Code Style

### Python Style Guidelines

- **Follow PEP 8** for Python code style
- **Use meaningful variable names** that clearly describe their purpose
- **Add docstrings** to all functions and classes
- **Keep functions focused** on a single responsibility
- **Use type hints** where appropriate

### Formatting Tools

If you install development tools, use these for consistent formatting:

```bash
# Format code
uv run black src/ cli/

# Sort imports
uv run isort src/ cli/

# Check style
uv run flake8 src/ cli/

# Type checking
uv run mypy src/ cli/
```

### Documentation Style

- **Use clear, concise language**
- **Include examples** for new features
- **Update relevant documentation** when making changes
- **Follow the existing documentation patterns**

## Submitting Changes

### Pull Request Process

1. **Ensure your changes are complete**:
   - Code is tested and working
   - Documentation is updated
   - Commit messages are clear

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**:
   - Go to the original tokka-bench repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

### Pull Request Guidelines

**Title**: Use a clear, descriptive title
- ✅ "Add support for custom output directory"
- ✅ "Fix dashboard crash when no results exist"
- ❌ "Update code"
- ❌ "Fixes"

**Description**: Include:
- **What** changes you made
- **Why** you made them
- **How** to test the changes
- **Any breaking changes** or migration notes

**Example PR description**:
```markdown
## Summary
Adds support for custom output directories in the benchmark CLI.

## Changes
- Added `output_dir` parameter to benchmark.py
- Updated CLI to accept `output_dir` argument
- Modified result saving logic to use custom directory
- Updated documentation with new parameter

## Testing
- Tested with custom output directory: `uv run cli/benchmark.py tokenizer=gpt2 output_dir=my_results`
- Verified backward compatibility with default behavior
- Confirmed dashboard can read results from custom directories

## Breaking Changes
None - this is backward compatible.
```

### Commit Message Guidelines

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

Examples:
- `feat: add support for custom metrics in benchmark`
- `fix: handle empty dataset gracefully in FineWeb loader`
- `docs: update README with new visualization features`

## Issue Guidelines

### Reporting Bugs

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected vs actual behavior**
4. **Environment information**:
   - Python version
   - Operating system
   - Relevant package versions
5. **Error messages or logs** (if any)
6. **Minimal example** that demonstrates the issue

### Feature Requests

For feature requests, please include:

1. **Clear description** of the proposed feature
2. **Use case** - why is this feature needed?
3. **Proposed implementation** (if you have ideas)
4. **Examples** of how it would be used
5. **Consideration of alternatives**

### Template for Issues

```markdown
## Description
[Clear description of the issue or feature request]

## Environment
- Python version: 
- Operating system: 
- tokka-bench version/commit: 

## Steps to Reproduce (for bugs)
1. 
2. 
3. 

## Expected Behavior
[What should happen]

## Actual Behavior (for bugs)
[What actually happens]

## Additional Context
[Any other relevant information]
```

## Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive in all interactions
- **Be constructive** in feedback and discussions
- **Help others** learn and contribute
- **Focus on the code and ideas**, not the person

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Ask questions during PR review if anything is unclear

### Recognition

Contributors are recognized in several ways:
- **Git history**: Your commits will be permanently part of the project history
- **Release notes**: Significant contributions will be mentioned in release notes
- **Appreciation**: We value all contributions, from typo fixes to major features

## Advanced Contributing

### Adding New Language Support

To add support for additional languages:

1. **Update language metadata**: Add entries to `src/fineweb-2-languages.csv`
2. **Verify data availability**: Ensure the language exists in FineWeb-2
3. **Test tokenizer performance**: Run benchmarks to verify results
4. **Update documentation**: Include the new language in README

### Adding New Metrics

To add new evaluation metrics:

1. **Implement in `benchmark.py`**: Add calculation logic to the `UniversalTokenizer` class
2. **Update output format**: Include new metrics in JSON results
3. **Update dashboard**: Add visualization for new metrics in `visualize.py`
4. **Document the metric**: Explain what it measures and how to interpret it

### Performance Optimizations

When contributing performance improvements:

1. **Measure before and after**: Use consistent benchmarks to demonstrate improvement
2. **Consider memory usage**: Optimize for both speed and memory efficiency
3. **Maintain accuracy**: Ensure optimizations don't compromise result quality
4. **Document trade-offs**: Explain any trade-offs made in optimization

---

Thank you for contributing to tokka-bench! Your efforts help make tokenizer evaluation more accessible and comprehensive for everyone.