# Testing Guide for Tokka-Bench

## 🧪 Modular Structure for Testing

The tokka-bench codebase has been split into testable modules to enable independent verification of all functions:

## 📁 Module Structure

```
src/tokka_bench/
├── unicode_utils.py    # Unicode script detection functions
├── metrics.py          # Word metrics & global tracking
├── data_utils.py       # Dataset loading functions
├── tokenizer.py        # UniversalTokenizer wrapper
├── benchmark.py        # Main orchestration (much smaller now)
└── __init__.py
```

## 🔧 Key Testable Functions

### 1. Unicode Utilities (`unicode_utils.py`)

- `get_unicode_scripts(text)` - Detects Unicode scripts in text
- `has_whitespace_in_middle(text)` - Checks for internal whitespace
- `starts_with_space(text)` - Checks if text starts with space

### 2. Metrics Calculation (`metrics.py`)

- `calculate_word_metrics(tokenizer, text)` - **Main split rate calculation**
- `analyze_vocabulary(tokenizer)` - Vocabulary analysis
- `GlobalMetricsTracker` - Global statistics tracking
- `get_token_analysis(tokenizer, text)` - Token-level analysis

### 3. Data Loading (`data_utils.py`)

- `load_language_data()` - Load language CSV data
- `load_coding_languages(n)` - Load programming language data
- `get_top_languages(df, n)` - Get top N languages by size
- `load_real_sample_text(lang_info, size)` - Load dataset samples

### 4. Tokenizer Wrapper (`tokenizer.py`)

- `UniversalTokenizer` - HuggingFace tokenizer wrapper
- `get_metrics(text)` - Main metrics calculation entry point

## 🧪 How to Test Each Module

### Option 1: Direct Testing (Python REPL)

```python
# Test unicode utilities
from src.tokka_bench.unicode_utils import get_unicode_scripts
scripts = get_unicode_scripts("Hello! 123 世界")
print(scripts)  # Should show: {'Latin', 'Punctuation', 'Numbers', 'Chinese'}

# Test metrics with mock tokenizer
from src.tokka_bench.metrics import calculate_word_metrics
from unittest.mock import Mock

mock_tokenizer = Mock()
mock_tokenizer.encode.side_effect = lambda x: [1, 2] if x == "hello" else [1]
result = calculate_word_metrics(mock_tokenizer, "hello")
print(result["continued_word_rate"])  # Test split rate calculation
```

### Option 2: Unit Tests (pytest)

```bash
# Install pytest if not already installed
pip install pytest

# Run tests
pytest tests/test_unicode_utils.py -v
pytest tests/test_metrics.py -v
```

### Option 3: Simple Validation Scripts

```python
# Create validation_script.py
import sys
sys.path.insert(0, 'src')

from tokka_bench.unicode_utils import get_unicode_scripts
from tokka_bench.metrics import GlobalMetricsTracker

# Test 1: Unicode detection
result = get_unicode_scripts("Hello! 123 世界")
assert "Latin" in result
assert "Chinese" in result
assert "Punctuation" in result
print("✓ Unicode detection working")

# Test 2: Global metrics tracking
tracker = GlobalMetricsTracker()
tracker.add_tokens([{
    "starts_with_space": True,
    "has_whitespace_in_middle": False,
    "scripts": ["Latin"],
    "script_overlap": False
}])
metrics = tracker.get_global_metrics()
assert metrics["total_tokens_analyzed"] == 1
print("✓ Global metrics tracking working")

print("All validations passed!")
```

## 🎯 Critical Functions to Verify

These are the most important functions to test for accuracy:

### 1. **Word Split Rate Calculation** (`metrics.py`)

```python
def calculate_word_metrics(tokenizer, text):
    # This is the core function that was previously broken
    # It uses sampling to calculate accurate split rates
    # Test with known tokenizers and expected results
```

### 2. **Unicode Script Detection** (`unicode_utils.py`)

```python
def get_unicode_scripts(text):
    # Make sure it correctly identifies:
    # - Latin, Chinese, Cyrillic, etc.
    # - Punctuation, Numbers, Symbols
    # - Multiple scripts in mixed text
```

### 3. **Global Metrics Tracking** (`metrics.py`)

```python
class GlobalMetricsTracker:
    # Verify percentages add up correctly
    # Test with various language combinations
```

## 🔍 Testing Strategy

1. **Unit Tests**: Test individual functions with known inputs/outputs
2. **Integration Tests**: Test module interactions
3. **Validation Tests**: Compare results against expected values
4. **Edge Case Tests**: Empty strings, special characters, etc.

## 📊 Example Test Data

```python
# Test cases for word splitting
test_cases = [
    ("hello world", ["hello", "world"]),  # Should not split
    ("hello", ["hel", "lo"]),             # Should split
    ("", []),                             # Empty text
    ("a", ["a"]),                         # Single character
]

# Test cases for Unicode scripts
unicode_test_cases = [
    ("Hello", {"Latin"}),
    ("Hello!", {"Latin", "Punctuation"}),
    ("Hello 123", {"Latin", "Numbers"}),
    ("Hello 世界", {"Latin", "Chinese"}),
]
```

## 🚀 Running Tests

```bash
# Quick validation
python -c "
import sys; sys.path.insert(0, 'src')
from tokka_bench.unicode_utils import get_unicode_scripts
print('✓ Modules working:', get_unicode_scripts('Hello! 123'))
"

# Full test suite (if using pytest)
pytest tests/ -v

# Custom validation script
python validation_script.py
```

## 💡 Benefits of Modular Structure

1. **Independent Testing**: Each function can be tested in isolation
2. **Mock Testing**: Easy to create mock tokenizers for testing
3. **Debugging**: Easier to debug specific functionality
4. **Verification**: You can verify each calculation step-by-step
5. **Refactoring**: Safe to modify individual modules without breaking others

The modular structure makes it much easier to verify the correctness of all the bug fixes, especially the critical word splitting calculation that was previously returning 100% for all languages.
