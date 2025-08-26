"""
Metrics utilities for tokenizer analysis.

Goals of this module:
- Provide a clear, minimal surface for computing word-level metrics.
- Offer a lightweight global metrics tracker for aggregated stats.
- Keep compatibility with existing callers and tests.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .unicode_utils import (
    get_unicode_scripts,
    has_whitespace_in_middle,
    starts_with_space,
)

# Module constants
DEFAULT_WORD_SAMPLE_SIZE = 1000
MIN_SYLLABLE_THRESHOLD = 3
VOCAB_ANALYSIS_SAMPLE_SIZE = 10000
SAMPLE_TOKEN_DISPLAY_LIMIT = 10


# Languages that don't typically use spaces between words
# Format: (iso_code, script) tuples
CHARACTER_BASED_LANGUAGES = {
    ("cmn", "Hani"),  # Chinese (Mandarin)
    ("yue", "Hani"),  # Chinese (Cantonese)
    ("zho", "Hani"),  # Chinese (generic)
    ("jpn", "Jpan"),  # Japanese (often no spaces)
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
    """Minimal tokenizer protocol used by metrics functions."""

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]: ...

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str: ...

    @property
    def name(self) -> str:  # pragma: no cover - optional for some callers
        ...

    @property
    def tokenizer(self) -> Any:  # pragma: no cover - optional for some callers
        ...


def calculate_word_metrics(
    tokenizer: TokenizerProtocol,
    text: str,
    language_info: Optional[Dict[str, str]] = None,
    pretokenized_text_token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute word-level metrics with simple, clear logic.

    Returns a dict containing:
    - subword_fertility
    - continued_word_rate (alias)
    - continuation_token_pct (primary key)
    - word_split_pct
    - debug_info
    """

    # Decide segmentation strategy
    words, segmentation_method = _segment_units(text, language_info)

    # Handle empty inputs uniformly
    if not words:
        return {
            "subword_fertility": 0.0,
            "continued_word_rate": 0.0,
            "word_split_pct": 0.0,
            "continuation_token_pct": 0.0,
            "debug_info": {
                "total_words": 0,
                "words_split": 0,
                "segmentation_method": segmentation_method,
                "language_info": language_info,
            },
        }

    # Tokenize full text once for fertility (or reuse provided tokenization)
    if pretokenized_text_token_ids is not None:
        total_tokens_for_words = len(pretokenized_text_token_ids)
    else:
        # Exclude special tokens so fertility reflects only content tokens
        try:
            total_tokens_for_words = len(
                tokenizer.encode(text, add_special_tokens=False)
            )
        except TypeError:
            # Some tokenizers may not accept add_special_tokens; fall back
            total_tokens_for_words = len(tokenizer.encode(text))

    # Sample units for splitting/continuation analysis
    sample_words = _even_sample(words, max_items=DEFAULT_WORD_SAMPLE_SIZE)

    words_split_in_sample = 0
    continuation_tokens_in_sample = 0
    total_tokens_in_sample = 0

    for unit in sample_words:
        try:
            unit_token_ids = tokenizer.encode(unit, add_special_tokens=False)
        except (TypeError, AttributeError):
            # Fallback for tokenizers that don't support add_special_tokens parameter
            unit_token_ids = tokenizer.encode(unit)

        num_tokens = len(unit_token_ids)
        total_tokens_in_sample += num_tokens

        is_split = num_tokens > 1
        if is_split:
            words_split_in_sample += 1
            continuation_tokens_in_sample += max(0, num_tokens - 1)

        # Omit per-unit debug tokenization details to keep metrics lightweight

    # Aggregate metrics
    sample_split_rate = (
        words_split_in_sample / len(sample_words) if sample_words else 0.0
    )
    continuation_token_pct = (
        (continuation_tokens_in_sample / total_tokens_in_sample) * 100
        if total_tokens_in_sample > 0
        else 0.0
    )
    estimated_total_words_split = int(sample_split_rate * len(words))
    subword_fertility = total_tokens_for_words / len(words) if words else 0.0

    return {
        "subword_fertility": subword_fertility,
        "continued_word_rate": continuation_token_pct,  # Back-compat alias
        "continuation_token_pct": continuation_token_pct,
        "word_split_pct": sample_split_rate * 100,
        "debug_info": {
            "total_words": len(words),
            "words_split": estimated_total_words_split,
            "segmentation_method": segmentation_method,
            "language_info": language_info,
        },
    }


def _segment_units(
    text: str, language_info: Optional[Dict[str, str]]
) -> Tuple[List[str], str]:
    """Return (units, method) according to language segmentation preferences.

    Methods: "syllable" (Tibetan-like), "character" (CJK/Thai-like), "whitespace".
    """
    if is_syllable_based_language(language_info):
        segments = [seg.strip() for seg in re.split(r"[་]", text)]
        syllables = [seg for seg in segments if seg and not seg.isspace()]
        if len(syllables) < MIN_SYLLABLE_THRESHOLD:
            # Fallback to character-based if too few syllable segments found
            syllables = [
                c for c in text if (not c.isspace()) and c.isprintable() and c != "་"
            ]
        return syllables, "syllable"

    if is_character_based_language(language_info):
        return [c for c in text if (not c.isspace()) and c.isprintable()], "character"

    return re.findall(r"\S+", text), "whitespace"


def _even_sample(items: Sequence[Any], max_items: int) -> List[Any]:
    """Evenly sample up to max_items from a sequence, preserving order."""
    if not items:
        return []
    if len(items) <= max_items:
        return list(items)
    step = max(1, len(items) // max_items)
    return list(items[::step])[:max_items]


def analyze_vocabulary(tokenizer: TokenizerProtocol) -> Dict[str, Any]:
    """Analyze a sample of the tokenizer's vocabulary.

    Kept for compatibility; primarily used for ad-hoc inspection.
    """
    print(
        f"    📊 Analyzing vocabulary for {getattr(tokenizer, 'name', '<tokenizer>')}..."
    )

    tokens_without_leading_space = 0
    sample_non_space_tokens: List[str] = []
    sample_space_tokens: List[str] = []

    vocab_size = len(getattr(tokenizer, "tokenizer"))
    sample_size = min(vocab_size, VOCAB_ANALYSIS_SAMPLE_SIZE)
    step = max(1, vocab_size // sample_size)

    analyzed_count = 0
    for token_id in range(0, vocab_size, step):
        try:
            token_text = getattr(tokenizer, "tokenizer").decode(
                [token_id], skip_special_tokens=True
            )
        except (ValueError, KeyError, RuntimeError, AttributeError):
            # Skip tokens that can't be decoded
            continue

        analyzed_count += 1
        if token_text and not token_text.startswith(" "):
            tokens_without_leading_space += 1
            if len(sample_non_space_tokens) < SAMPLE_TOKEN_DISPLAY_LIMIT:
                sample_non_space_tokens.append(token_text)
        else:
            if len(sample_space_tokens) < SAMPLE_TOKEN_DISPLAY_LIMIT:
                sample_space_tokens.append(token_text)

    if analyzed_count > 0:
        pct = (tokens_without_leading_space / analyzed_count) * 100
        est_count = int((tokens_without_leading_space / analyzed_count) * vocab_size)
    else:
        pct = 0.0
        est_count = 0

    print(f"      Analyzed {analyzed_count:,} tokens from vocabulary of {vocab_size:,}")
    print(f"      Tokens without leading space: {pct:.1f}% (~{est_count:,} tokens)")
    print(f"      Sample non-space tokens: {sample_non_space_tokens}")
    print(f"      Sample space tokens: {sample_space_tokens}")

    return {
        "tokens_without_leading_space_count": est_count,
        "tokens_without_leading_space_pct": pct,
        "analyzed_sample_size": analyzed_count,
        "sample_non_space_tokens": sample_non_space_tokens,
        "sample_space_tokens": sample_space_tokens,
    }


class GlobalMetricsTracker:
    """Aggregate simple global metrics across tokens."""

    def __init__(self) -> None:
        self.all_tokens: List[Dict[str, Any]] = []
        self.script_counts: defaultdict[str, int] = defaultdict(int)
        self.space_start_count: int = 0
        self.whitespace_middle_count: int = 0
        self.script_overlap_count: int = 0
        self.total_token_count: int = 0

    def add_tokens(self, tokens_info: List[Dict[str, Any]]) -> None:
        for token_info in tokens_info:
            self.total_token_count += 1

            if token_info.get("starts_with_space", False):
                self.space_start_count += 1

            if token_info.get("has_whitespace_in_middle", False):
                self.whitespace_middle_count += 1

            for script in token_info.get("scripts", []) or []:
                self.script_counts[script] += 1

            if token_info.get("script_overlap", False):
                self.script_overlap_count += 1

            self.all_tokens.append(token_info)

    def get_global_metrics(self) -> Dict[str, float]:
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

        for script, count in self.script_counts.items():
            metrics[f"tokens_with_{script.lower()}_unicode_pct"] = (
                count / self.total_token_count
            ) * 100

        return metrics


def get_token_analysis(tokenizer: TokenizerProtocol, text: str) -> Dict[str, Any]:
    """Decode tokens and annotate simple properties for analysis."""
    # Exclude special tokens and skip empty decoded tokens
    try:
        token_ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    tokens_info: List[Dict[str, Any]] = []

    for token_id in token_ids:
        try:
            token_text: str = getattr(tokenizer, "tokenizer").decode(
                [token_id], skip_special_tokens=True
            )
        except (ValueError, KeyError, RuntimeError, AttributeError):
            # Skip tokens that can't be decoded
            continue

        if not token_text:
            # Skip empty decoded tokens
            continue

        scripts = get_unicode_scripts(token_text)
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

    return {"tokens": tokens_info}
