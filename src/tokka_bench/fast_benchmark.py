"""
Fast, clean multi-tokenizer benchmark.

Design goals:
- Default: benchmark multiple tokenizers concurrently
- Load each language's text once, reuse across all tokenizers
- Minimal logging and no noisy per-tokenizer prints
- Lightweight global metrics via sampling (fast) while preserving output shape
"""

from __future__ import annotations

import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from .data_utils import (
    get_english_fineweb,
    get_top_languages,
    load_coding_languages,
    load_language_data,
    load_real_sample_text,
)
from .metrics import GlobalMetricsTracker, calculate_word_metrics


@dataclass
class FastTokenizer:
    """Lightweight tokenizer wrapper without verbose analysis on init."""

    name: str

    def __post_init__(self) -> None:
        # trust_remote_code=True allows custom tokenizers (e.g., tiktoken-wrapped)
        self._tok = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
        self.vocab_size: int = len(self._tok)

    # Expose a minimal protocol expected by metrics functions
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def tokenizer(self):  # for compatibility with GlobalMetricsTracker helpers
        return self._tok


def _compute_vocab_metrics_quiet(tokenizer: FastTokenizer) -> Dict[str, Any]:
    """Sample-based vocab analysis without printing (compatible keys)."""
    vocab_size: int = len(tokenizer.tokenizer)
    sample_size: int = min(vocab_size, 10000)
    step: int = max(1, vocab_size // max(1, sample_size))

    tokens_without_leading_space = 0
    analyzed_count = 0
    sample_non_space_tokens: List[str] = []
    sample_space_tokens: List[str] = []

    for token_id in range(0, vocab_size, step):
        try:
            token_text: str = tokenizer.tokenizer.decode(
                [token_id], skip_special_tokens=True
            )
        except Exception:
            continue

        analyzed_count += 1
        if token_text and not token_text.startswith(" "):
            tokens_without_leading_space += 1
            if len(sample_non_space_tokens) < 10:
                sample_non_space_tokens.append(token_text)
        else:
            if len(sample_space_tokens) < 10:
                sample_space_tokens.append(token_text)

    if analyzed_count > 0:
        pct = (tokens_without_leading_space / analyzed_count) * 100
        est_count = int((tokens_without_leading_space / analyzed_count) * vocab_size)
    else:
        pct = 0.0
        est_count = 0

    return {
        "tokens_without_leading_space_count": est_count,
        "tokens_without_leading_space_pct": pct,
        "analyzed_sample_size": analyzed_count,
        "sample_non_space_tokens": sample_non_space_tokens,
        "sample_space_tokens": sample_space_tokens,
    }


def _compute_language_metrics(
    tokenizer: FastTokenizer, text: str, lang_info: Dict[str, str]
) -> Dict[str, Any]:
    token_ids: List[int] = tokenizer.encode(text)
    text_bytes: int = len(text.encode("utf-8"))
    num_tokens: int = len(token_ids)
    unique_tokens: int = len(set(token_ids))

    word_metrics: Dict[str, Any] = calculate_word_metrics(tokenizer, text, lang_info)

    return {
        "bytes_per_token": (text_bytes / num_tokens) if num_tokens > 0 else 0.0,
        "total_bytes": text_bytes,
        "total_tokens": num_tokens,
        "unique_tokens": unique_tokens,
        "subword_fertility": word_metrics["subword_fertility"],
        "continued_word_rate": word_metrics["continued_word_rate"],
    }


def _sample_token_info_for_global(
    tokenizer: FastTokenizer, token_ids: List[int], max_samples: int = 2000
) -> List[Dict[str, Any]]:
    """Decode a sample of token IDs to reduce cost of global metrics."""
    if not token_ids:
        return []

    if len(token_ids) > max_samples:
        step = max(1, len(token_ids) // max_samples)
        sampled_ids = token_ids[::step][:max_samples]
    else:
        sampled_ids = token_ids

    tokens_info: List[Dict[str, Any]] = []
    for tid in sampled_ids:
        try:
            token_text = tokenizer.tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            continue

        # Minimal Unicode analysis inline to avoid importing heavy helpers repeatedly
        from .unicode_utils import (
            get_unicode_scripts,
            starts_with_space,
            has_whitespace_in_middle,
        )

        scripts = get_unicode_scripts(token_text)
        tokens_info.append(
            {
                "id": tid,
                "text": token_text,
                "starts_with_space": starts_with_space(token_text),
                "has_whitespace_in_middle": has_whitespace_in_middle(token_text),
                "scripts": list(scripts),
                "script_overlap": len(scripts) > 1,
            }
        )

    return tokens_info


def _process_single_language(
    tokenizers: List[FastTokenizer],
    lang_info: Dict[str, str],
    sample_size_mb: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Return per-tokenizer metrics and sampled token infos for global metrics."""
    text: str = load_real_sample_text(lang_info, sample_size_mb, verbose=False)

    per_tokenizer_metrics: Dict[str, Dict[str, Any]] = {}
    per_tokenizer_sampled_tokens: Dict[str, List[Dict[str, Any]]] = {}

    for tok in tokenizers:
        # Compute metrics
        metrics = _compute_language_metrics(tok, text, lang_info)
        per_tokenizer_metrics[tok.name] = {
            "language_info": lang_info,
            "metrics": metrics,
        }

        # Global metrics sampling
        token_ids = tok.encode(text)
        tokens_info = _sample_token_info_for_global(tok, token_ids, max_samples=2000)
        per_tokenizer_sampled_tokens[tok.name] = tokens_info

    return per_tokenizer_metrics, per_tokenizer_sampled_tokens


def run_benchmark(
    tokenizer_names: List[str],
    output_names: Optional[List[str]] = None,
    sample_size_mb: float = 1.0,
    max_workers: int = 8,
    natural_n: int = 99,
    code_n: int = 20,
) -> Dict[str, Any]:
    """Benchmark multiple tokenizers across many languages quickly.

    Returns a dict keyed by tokenizer name with the same shape as the classic benchmark.
    """
    if not tokenizer_names:
        raise ValueError("tokenizer_names must be a non-empty list")

    # Quiet header; a single compact line
    print(
        f"🚀 Fast benchmark | tokenizers={len(tokenizer_names)} | sample={sample_size_mb}MB | workers={max_workers}"
    )

    # Load tokenizers (quiet)
    print("🔧 Loading tokenizers...")
    tokenizers: List[FastTokenizer] = []
    for name in tokenizer_names:
        # concise per-tokenizer load message
        print(f"  • {name}")
        tokenizers.append(FastTokenizer(name))
    print(f"✅ Loaded: {len(tokenizers)}")

    # Pre-compute quiet vocab metrics once per tokenizer
    vocab_metrics_by_tok: Dict[str, Dict[str, Any]] = {}
    for tok in tokenizers:
        vocab_metrics_by_tok[tok.name] = _compute_vocab_metrics_quiet(tok)

    # Languages: English + configurable natural/code language counts (defaults match classic)
    english = get_english_fineweb()
    df = load_language_data()
    natural_languages = get_top_languages(df, n=natural_n)
    coding_languages = load_coding_languages(n=code_n)
    all_languages: List[Dict[str, str]] = (
        [english] + natural_languages + coding_languages
    )

    print(
        f"📊 Languages: {len(all_languages)} (1 Eng, {len(natural_languages)} nat, {len(coding_languages)} code)"
    )

    # Prepare per-tokenizer result containers
    all_results: Dict[str, Dict[str, Any]] = {
        tok.name: {
            "tokenizer": tok.name,
            "vocab_size": tok.vocab_size,
            "vocab_metrics": vocab_metrics_by_tok[tok.name],
            "benchmark_size_mb": sample_size_mb,
            "timestamp": time.time(),
            "languages": {},
            "global_metrics": {},
        }
        for tok in tokenizers
    }

    # Global trackers per tokenizer
    global_trackers: Dict[str, GlobalMetricsTracker] = {
        tok.name: GlobalMetricsTracker() for tok in tokenizers
    }

    # Process languages in parallel (one unit of work per language)
    print("🏃 Running...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_language, tokenizers, lang, sample_size_mb
            ): lang
            for lang in all_languages
        }

        completed = 0
        total = len(all_languages)
        for future in as_completed(futures):
            lang_info = futures[future]
            try:
                per_tok_metrics, per_tok_tokens = future.result()
            except Exception as e:
                print(f"  ❌ {lang_info.get('name', 'Unknown')}: {e}")
                completed += 1
                continue

            # Unique key per language
            if lang_info["source"] == "starcoder":
                key = f"{lang_info['iso_code']}-code"
            else:
                key = f"{lang_info['iso_code']}-{lang_info['script']}"

            # Store per-tokenizer language results and update trackers
            for tok_name, lang_result in per_tok_metrics.items():
                all_results[tok_name]["languages"][key] = lang_result
            for tok_name, tokens_info in per_tok_tokens.items():
                global_trackers[tok_name].add_tokens(tokens_info)

            completed += 1
            # Simple inline progress bar
            if completed % 1 == 0:
                width = 30
                filled = int(width * completed / total)
                bar = "#" * filled + "-" * (width - filled)
                print(f"\r📈 [{bar}] {completed}/{total}", end="", flush=True)
        print()  # newline after progress bar

    # Finalize global metrics
    print("🔄 Finalizing global metrics...")
    for tok_name, tracker in global_trackers.items():
        all_results[tok_name]["global_metrics"] = tracker.get_global_metrics()

    # Save results per tokenizer
    saved_files: List[str] = []
    for i, tok_name in enumerate(tokenizer_names):
        results = all_results[tok_name]

        # Determine output filename
        if output_names and i < len(output_names):
            output_filename = f"{output_names[i]}.json"
        else:
            safe_name = tok_name.replace("/", "_").replace("-", "_")
            output_filename = f"{safe_name}.json"

        output_path = os.path.join("data", "results", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        saved_files.append(output_path)
        print(f"💾 {output_path}")

    # Cleanup
    try:
        gc.collect()
    except Exception:
        pass

    print("✅ Done")
    if len(saved_files) > 1:
        print(f"📁 Files: {', '.join(saved_files)}")

    return all_results
