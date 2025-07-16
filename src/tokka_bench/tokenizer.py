"""
Universal tokenizer wrapper for HuggingFace models.

This module provides the UniversalTokenizer class that can load any HuggingFace
tokenizer and calculate various metrics including vocabulary analysis and
word-level statistics.
"""

from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from .metrics import analyze_vocabulary, calculate_word_metrics, get_token_analysis


class UniversalTokenizer:
    """Universal tokenizer that can load any HuggingFace model."""

    def __init__(self, model_name: str) -> None:
        self.name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Get vocabulary size
        self.vocab_size: int = len(self.tokenizer)

        # Calculate vocabulary-level metrics once
        self.vocab_metrics: Dict[str, Any] = analyze_vocabulary(self)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_metrics(
        self, text: str, language_info: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Calculate metrics for given text."""
        # Encode the text
        token_ids: List[int] = self.encode(text)

        # Basic metrics from the actual text we're tokenizing
        text_bytes: int = len(text.encode("utf-8"))
        num_tokens: int = len(token_ids)
        unique_tokens: int = len(set(token_ids))  # Count unique token IDs

        # Calculate word-level metrics with language information
        word_metrics: Dict[str, Any] = calculate_word_metrics(self, text, language_info)

        # Debug logging
        debug_info: Dict[str, Any] = word_metrics.get("debug_info", {})
        segmentation_method = debug_info.get("segmentation_method", "whitespace")

        print("    🔍 Debug Info:")
        print(
            f"      Text bytes: {text_bytes:,}, Tokens: {num_tokens:,}, Unique tokens: {unique_tokens:,}"
        )
        print(
            f"      {segmentation_method.title()} units: {debug_info.get('total_words', 0):,}, Units split: {debug_info.get('words_split', 0):,}"
        )
        print(
            f"      Sample analysis: {debug_info.get('sampled_words', 0):,} units, {debug_info.get('words_split_in_sample', 0):,} split"
        )
        print(f"      Split rate: {debug_info.get('sample_split_rate', 0) * 100:.1f}%")
        print(
            f"      Continuation tokens: {debug_info.get('continuation_tokens_in_sample', 0):,}/{debug_info.get('total_tokens_in_sample', 0):,}"
        )
        print(f"      Continued word rate: {word_metrics['continued_word_rate']:.1f}%")

        # Show sample words and tokenizations
        sample_words: List[str] = debug_info.get("sample_words", [])
        sample_tokenizations: List[Dict[str, Any]] = debug_info.get(
            "sample_word_tokenizations", []
        )
        if sample_words:
            print(f"      Sample {segmentation_method} units: {sample_words}")
        if sample_tokenizations:
            for tok in sample_tokenizations[:5]:  # Show first 5 tokenizations
                split_indicator = "🔀" if tok["is_split"] else "✓"
                print(f"        {split_indicator} '{tok['word']}' → {tok['tokens']}")

        return {
            "bytes_per_token": text_bytes / num_tokens if num_tokens > 0 else 0,
            "total_bytes": text_bytes,
            "total_tokens": num_tokens,
            "unique_tokens": unique_tokens,
            "subword_fertility": word_metrics["subword_fertility"],
            "continued_word_rate": word_metrics["continued_word_rate"],
            "debug_info": debug_info,  # Include debug info in output
        }

    def get_token_analysis(self, text: str) -> Dict[str, Any]:
        """Get detailed token analysis for global metrics."""
        return get_token_analysis(self, text)
