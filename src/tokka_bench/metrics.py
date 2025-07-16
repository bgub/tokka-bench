"""
Metrics calculation utilities for tokenizer analysis.

This module provides functions for calculating word-level metrics (split rates, fertility)
and tracking global statistics across multiple languages.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol

from .unicode_utils import (
    get_unicode_scripts,
    has_whitespace_in_middle,
    starts_with_space,
)


# Languages that don't typically use spaces between words
# Format: (iso_code, script) tuples
CHARACTER_BASED_LANGUAGES = {
    ("cmn", "Hani"),  # Chinese (Mandarin)
    ("yue", "Hani"),  # Chinese (Cantonese)
    ("zho", "Hani"),  # Chinese (generic)
    ("jpn", "Jpan"),  # Japanese (mixed scripts but often no spaces)
    ("tha", "Thai"),  # Thai
    ("khm", "Khmr"),  # Khmer
    ("lao", "Laoo"),  # Lao
    ("mya", "Mymr"),  # Myanmar
    ("sat", "Olck"),  # Santali
    ("nod", "Lana"),  # Northern Thai
}

# Languages that use syllable-based segmentation (e.g., tsheg marks in Tibetan)
# Format: (iso_code, script) tuples
SYLLABLE_BASED_LANGUAGES = {
    ("bod", "Tibt"),  # Tibetan
    ("dzo", "Tibt"),  # Dzongkha (Bhutanese)
    ("lad", "Tibt"),  # Ladakhi
}


def is_character_based_language(language_info: Optional[Dict[str, str]]) -> bool:
    """
    Determine if a language uses character-based segmentation instead of spaces.

    Args:
        language_info: Dictionary containing 'iso_code' and 'script' keys

    Returns:
        True if the language should use character-based segmentation
    """
    if not language_info:
        return False

    iso_code = language_info.get("iso_code", "").lower()
    script = language_info.get("script", "")

    # Check exact matches first
    if (iso_code, script) in CHARACTER_BASED_LANGUAGES:
        return True

    # Check for Chinese variants (any script with Han/Hani)
    if script in ("Hani", "Hans", "Hant") or "Han" in script:
        return True

    # Check for other character-based scripts
    character_based_scripts = {
        "Thai",
        "Khmr",
        "Laoo",
        "Mymr",
        "Olck",
        "Lana",
        "Jpan",  # Japanese can be mixed but often has no spaces
    }

    return script in character_based_scripts


def is_syllable_based_language(language_info: Optional[Dict[str, str]]) -> bool:
    """
    Determine if a language uses syllable-based segmentation (e.g., tsheg marks).

    Args:
        language_info: Dictionary containing 'iso_code' and 'script' keys

    Returns:
        True if the language should use syllable-based segmentation
    """
    if not language_info:
        return False

    iso_code = language_info.get("iso_code", "").lower()
    script = language_info.get("script", "")

    # Check for syllable-based languages
    if (iso_code, script) in SYLLABLE_BASED_LANGUAGES:
        return True

    # Check for Tibetan script variants
    if script in ("Tibt", "Tibetan"):
        return True

    return False


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer objects with encode/decode methods."""

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        ...

    @property
    def name(self) -> str:
        """Tokenizer name."""
        ...

    @property
    def tokenizer(self) -> Any:
        """Underlying tokenizer object."""
        ...


def calculate_word_metrics(
    tokenizer: TokenizerProtocol,
    text: str,
    language_info: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Calculate sub-word fertility and split rates using efficient sampling.

    Args:
        tokenizer: Tokenizer to analyze
        text: Text to analyze
        language_info: Optional language information dict with 'iso_code' and 'script'

    Returns:
        Dictionary containing metrics and debug information
    """
    # Determine segmentation method based on language
    is_char_based = is_character_based_language(language_info)
    is_syllable_based = is_syllable_based_language(language_info)

    if is_syllable_based:
        # For syllable-based languages (e.g., Tibetan), split by tsheg marks
        # The tsheg mark (་) separates syllables in Tibetan

        # First, split by tsheg marks (་)
        tsheg_separated = re.split(r"[་]", text)

        # Filter out empty strings and whitespace-only strings
        syllables = []
        for segment in tsheg_separated:
            # Remove leading/trailing whitespace
            segment = segment.strip()
            # Skip empty segments
            if segment and not segment.isspace():
                syllables.append(segment)

        # If we don't have many syllables, fall back to character-based
        # This handles texts that might not have tsheg marks
        if len(syllables) < 3:
            syllables = []
            for char in text:
                if not char.isspace() and char.isprintable() and char != "་":
                    syllables.append(char)

        words = syllables
        segmentation_method = "syllable"

    elif is_char_based:
        # For character-based languages, split by character
        # Filter out whitespace and punctuation to get meaningful units
        words = []
        for char in text:
            if not char.isspace() and char.isprintable():
                words.append(char)
        segmentation_method = "character"
    else:
        # For space-based languages, split by whitespace
        words = re.findall(r"\S+", text)
        segmentation_method = "whitespace"

    if not words:
        return {
            "subword_fertility": 0.0,
            "continued_word_rate": 0.0,
            "debug_info": {
                "total_words": 0,
                "words_split": 0,
                "sampled_words": 0,
                "sample_words": [],
                "sample_word_tokenizations": [],
                "segmentation_method": segmentation_method,
                "language_info": language_info,
            },
        }

    # Tokenize the full text once for total token count
    token_ids: List[int] = tokenizer.encode(text)
    total_tokens_for_words: int = len(token_ids)

    # Sample words for accurate splitting analysis (much more efficient than tokenizing all words)
    sample_size: int = min(len(words), 1000)  # Sample up to 1000 words/characters
    step: int = max(1, len(words) // sample_size)
    sample_words: List[str] = words[::step][:sample_size]

    # Tokenize sample words individually to check for splitting
    words_split_in_sample: int = 0
    continuation_tokens_in_sample: int = 0
    total_tokens_in_sample: int = 0
    sample_word_tokenizations: List[Dict[str, Any]] = []

    for word in sample_words:
        # Use add_special_tokens=False to avoid BOS/EOS tokens affecting split detection
        word_token_ids: List[int] = tokenizer.encode(word, add_special_tokens=False)
        is_split: bool = len(word_token_ids) > 1

        # Count tokens for correct continued_word_rate calculation
        num_tokens = len(word_token_ids)
        total_tokens_in_sample += num_tokens

        if is_split:
            words_split_in_sample += 1
            # All tokens except the first are continuations
            continuation_tokens_in_sample += num_tokens - 1

        # Store sample tokenization for debugging (first 10 only)
        if len(sample_word_tokenizations) < 10:
            word_tokens: List[str] = []
            for token_id in word_token_ids:
                try:
                    token_text: str = tokenizer.decode(
                        [token_id], skip_special_tokens=True
                    )
                    word_tokens.append(token_text)
                except Exception:
                    word_tokens.append("<DECODE_ERROR>")

            sample_word_tokenizations.append(
                {"word": word, "tokens": word_tokens, "is_split": is_split}
            )

    # Calculate metrics based on sample
    sample_split_rate: float = (
        words_split_in_sample / len(sample_words) if sample_words else 0.0
    )

    # FIXED: Calculate percentage of tokens that are continuations (not percentage of words that are split)
    continued_word_rate: float = (
        continuation_tokens_in_sample / total_tokens_in_sample * 100
        if total_tokens_in_sample > 0
        else 0.0
    )

    # Estimate total words split based on sample
    estimated_total_words_split: int = int(sample_split_rate * len(words))

    subword_fertility: float = total_tokens_for_words / len(words) if words else 0.0

    return {
        "subword_fertility": subword_fertility,
        "continued_word_rate": continued_word_rate,
        "debug_info": {
            "total_words": len(words),
            "words_split": estimated_total_words_split,
            "sampled_words": len(sample_words),
            "words_split_in_sample": words_split_in_sample,
            "sample_split_rate": sample_split_rate,
            "continuation_tokens_in_sample": continuation_tokens_in_sample,
            "total_tokens_in_sample": total_tokens_in_sample,
            "sample_words": sample_words[:10],
            "sample_word_tokenizations": sample_word_tokenizations,
            "segmentation_method": segmentation_method,
            "language_info": language_info,
        },
    }


def analyze_vocabulary(tokenizer: TokenizerProtocol) -> Dict[str, Any]:
    """Analyze the tokenizer's vocabulary to get global statistics."""
    print(f"    📊 Analyzing vocabulary for {tokenizer.name}...")

    tokens_without_leading_space: int = 0
    sample_non_space_tokens: List[str] = []
    sample_space_tokens: List[str] = []

    # Analyze a sample of token IDs to avoid performance issues with huge vocabularies
    vocab_size: int = len(tokenizer.tokenizer)
    sample_size: int = min(vocab_size, 10000)  # Sample up to 10K tokens
    step: int = max(1, vocab_size // sample_size)

    analyzed_count: int = 0
    for token_id in range(0, vocab_size, step):
        try:
            token_text: str = tokenizer.tokenizer.decode(
                [token_id], skip_special_tokens=True
            )
            analyzed_count += 1

            if token_text and not token_text.startswith(" "):
                tokens_without_leading_space += 1
                if len(sample_non_space_tokens) < 10:
                    sample_non_space_tokens.append(token_text)
            else:
                if len(sample_space_tokens) < 10:
                    sample_space_tokens.append(token_text)

        except Exception:
            continue

    # Extrapolate to full vocabulary
    if analyzed_count > 0:
        tokens_without_leading_space_pct: float = (
            tokens_without_leading_space / analyzed_count
        ) * 100
        estimated_total_non_space: int = int(
            (tokens_without_leading_space / analyzed_count) * vocab_size
        )
    else:
        tokens_without_leading_space_pct = 0.0
        estimated_total_non_space = 0

    print(f"      Analyzed {analyzed_count:,} tokens from vocabulary of {vocab_size:,}")
    print(
        f"      Tokens without leading space: {tokens_without_leading_space_pct:.1f}% (~{estimated_total_non_space:,} tokens)"
    )
    print(f"      Sample non-space tokens: {sample_non_space_tokens}")
    print(f"      Sample space tokens: {sample_space_tokens}")

    return {
        "tokens_without_leading_space_count": estimated_total_non_space,
        "tokens_without_leading_space_pct": tokens_without_leading_space_pct,
        "analyzed_sample_size": analyzed_count,
        "sample_non_space_tokens": sample_non_space_tokens,
        "sample_space_tokens": sample_space_tokens,
    }


class GlobalMetricsTracker:
    """Tracks global metrics across all languages."""

    def __init__(self) -> None:
        self.all_tokens: List[Dict[str, Any]] = []
        self.script_counts: defaultdict[str, int] = defaultdict(int)
        self.space_start_count: int = 0
        self.whitespace_middle_count: int = 0
        self.script_overlap_count: int = 0
        self.total_token_count: int = 0

    def add_tokens(self, tokens_info: List[Dict[str, Any]]) -> None:
        """Add token information from a language sample."""
        for token_info in tokens_info:
            self.total_token_count += 1

            # Track space-starting tokens
            if token_info["starts_with_space"]:
                self.space_start_count += 1

            # Track tokens with whitespace in middle
            if token_info["has_whitespace_in_middle"]:
                self.whitespace_middle_count += 1

            # Track script usage
            for script in token_info["scripts"]:
                self.script_counts[script] += 1

            # Track script overlap
            if token_info["script_overlap"]:
                self.script_overlap_count += 1

            # Store token for detailed analysis if needed
            self.all_tokens.append(token_info)

    def get_global_metrics(self) -> Dict[str, float]:
        """Calculate and return global metrics."""
        if self.total_token_count == 0:
            return {}

        metrics: Dict[str, float] = {
            "total_tokens_analyzed": self.total_token_count,
            "tokens_starting_with_space_pct": (
                self.space_start_count / self.total_token_count
            )
            * 100,
            "tokens_with_whitespace_in_middle_pct": (
                self.whitespace_middle_count / self.total_token_count
            )
            * 100,
            "tokens_with_script_overlap_pct": (
                self.script_overlap_count / self.total_token_count
            )
            * 100,
        }

        # Add script-specific percentages
        for script, count in self.script_counts.items():
            metrics[f"tokens_with_{script.lower()}_unicode_pct"] = (
                count / self.total_token_count
            ) * 100

        return metrics


def get_token_analysis(tokenizer: TokenizerProtocol, text: str) -> Dict[str, Any]:
    """Get detailed token analysis for global metrics."""
    token_ids: List[int] = tokenizer.encode(text)
    tokens_info: List[Dict[str, Any]] = []

    for token_id in token_ids:
        # Decode individual token
        try:
            token_text: str = tokenizer.tokenizer.decode(
                [token_id], skip_special_tokens=True
            )

            # Analyze token properties
            scripts: set[str] = get_unicode_scripts(token_text)
            tokens_info.append(
                {
                    "id": token_id,
                    "text": token_text,
                    "starts_with_space": starts_with_space(token_text),
                    "has_whitespace_in_middle": has_whitespace_in_middle(token_text),
                    "scripts": list(scripts),
                    "script_overlap": len(scripts) > 1,
                }
            )
        except Exception:
            # Skip problematic tokens
            continue

    return {"tokens": tokens_info}
