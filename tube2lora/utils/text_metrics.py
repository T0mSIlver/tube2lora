from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

import textstat
import tiktoken


WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def tokenize_words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def count_words(text: str) -> int:
    return len(tokenize_words(text))


def count_sentences(text: str) -> int:
    return len(SENTENCE_RE.findall(text))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def char_similarity(left: str, right: str) -> float:
    left_norm = normalize_whitespace(left)
    right_norm = normalize_whitespace(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(a=left_norm, b=right_norm).ratio()


def _encoding_candidates(model: str | None) -> list[str]:
    if model is None:
        return []

    value = model.strip()
    if not value:
        return []

    candidates = [value]
    if "/" in value:
        candidates.append(value.rsplit("/", 1)[-1])
    if value.lower().endswith(".gguf"):
        candidates.append(value[:-5])
        if "/" in value:
            short = value.rsplit("/", 1)[-1]
            if short.lower().endswith(".gguf"):
                candidates.append(short[:-5])

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


@lru_cache(maxsize=128)
def _resolve_encoding(model: str | None) -> tiktoken.Encoding:
    for candidate in _encoding_candidates(model):
        try:
            return tiktoken.encoding_for_model(candidate)
        except KeyError:
            continue
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, *, model: str | None = None) -> int:
    if not text:
        return 0
    encoding = _resolve_encoding(model)
    return len(encoding.encode(text))


def lexical_metrics(text: str) -> dict[str, float | int]:
    words = tokenize_words(text)
    word_count = len(words)
    if word_count == 0:
        return {
            "word_count": 0,
            "unique_word_count": 0,
            "unique_ratio": 0.0,
            "hapax_ratio": 0.0,
            "avg_word_length": 0.0,
            "long_word_ratio": 0.0,
        }

    counts = Counter(words)
    unique_word_count = len(counts)
    hapax_count = sum(1 for freq in counts.values() if freq == 1)
    total_word_chars = sum(len(word) for word in words)
    long_word_count = sum(1 for word in words if len(word) >= 7)

    return {
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "unique_ratio": safe_ratio(unique_word_count, word_count),
        "hapax_ratio": safe_ratio(hapax_count, word_count),
        "avg_word_length": safe_ratio(total_word_chars, word_count),
        "long_word_ratio": safe_ratio(long_word_count, word_count),
    }


def _safe_readability_value(func, text: str) -> float | None:
    try:
        value = func(text)
    except Exception:
        return None
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def readability_metrics(text: str) -> dict[str, float | None]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return {
            "flesch_reading_ease": None,
            "automated_readability_index": None,
            "gunning_fog": None,
        }

    return {
        "flesch_reading_ease": _safe_readability_value(textstat.flesch_reading_ease, normalized),
        "automated_readability_index": _safe_readability_value(
            textstat.automated_readability_index,
            normalized,
        ),
        "gunning_fog": _safe_readability_value(textstat.gunning_fog, normalized),
    }


def flesch_to_unit_interval(flesch_reading_ease: float | None) -> float:
    if flesch_reading_ease is None:
        return 0.5
    # Typical Flesch range is roughly 0..100; clamp to that and scale.
    return max(0.0, min(1.0, flesch_reading_ease / 100.0))
