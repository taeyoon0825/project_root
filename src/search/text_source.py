from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Iterable

import pandas as pd

from src.adaptive.normalization_resources import NormalizationResources, build_normalization_resources


SUPPORTED_TEXT_SOURCES = ("stt_transcript", "original_transcript", "combined")
DEFAULT_TEXT_SOURCE = "stt_transcript"
SUPPORTED_DENSE_NORMALIZATION_MODES = ("baseline", "adaptive_corpus")
DEFAULT_DENSE_NORMALIZATION_MODE = "adaptive_corpus"


@lru_cache(maxsize=1)
def _fallback_resources() -> NormalizationResources:
    return build_normalization_resources(pd.DataFrame())


def validate_text_source(text_source: str) -> str:
    if text_source not in SUPPORTED_TEXT_SOURCES:
        raise ValueError(f"Unsupported text source: {text_source}")
    return text_source


def validate_dense_normalization_mode(mode: str) -> str:
    if mode not in SUPPORTED_DENSE_NORMALIZATION_MODES:
        raise ValueError(f"Unsupported dense normalization mode: {mode}")
    return mode


def resolve_dense_normalization_mode(
    mode: str | None = None,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    candidate = str(mode or "").strip()
    if candidate:
        return validate_dense_normalization_mode(candidate)
    adaptive_resources = resources or _fallback_resources()
    return validate_dense_normalization_mode(adaptive_resources.recommended_mode or DEFAULT_DENSE_NORMALIZATION_MODE)


def normalize_text_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _collapse_repeated_characters(value: str) -> str:
    return re.sub(r"(.)\1{3,}", r"\1\1", value)


def _collapse_repeated_tokens(value: str) -> str:
    return re.sub(r"\b([0-9a-z\uac00-\ud7a3]+)(?:\s+\1){1,}\b", r"\1", value)


def _replace_surface(value: str, surface: str, replacement: str) -> str:
    if not surface or not replacement or surface == replacement:
        return value
    escaped = re.escape(surface)
    pattern = re.compile(rf"(?<![0-9A-Za-z\uac00-\ud7a3]){escaped}(?![0-9A-Za-z\uac00-\ud7a3])", re.IGNORECASE)
    return pattern.sub(replacement, value)


def _apply_alias_map(value: str, resources: NormalizationResources) -> str:
    alias_items = sorted(resources.alias_map.items(), key=lambda item: len(item[0]), reverse=True)
    for surface, replacement in alias_items:
        value = _replace_surface(value, str(surface).lower(), str(replacement).lower())
    return value


def _strip_filler_terms(value: str, resources: NormalizationResources) -> str:
    result = value
    for filler in sorted(resources.filler_terms, key=len, reverse=True):
        result = _replace_surface(result, str(filler).lower(), " ")
    return result


def _expand_spoken_years(value: str, resources: NormalizationResources) -> str:
    year_suffix_words = dict(resources.number_words.get("year_suffix", {}))
    if not year_suffix_words:
        return value

    suffix_choices = "|".join(re.escape(key) for key in year_suffix_words)

    def _replace(match: re.Match[str]) -> str:
        suffix_word = str(match.group(1) or "").lower()
        suffix_value = year_suffix_words.get(suffix_word)
        if suffix_value is None:
            return match.group(0)
        return f"{match.group(0)} 20{int(suffix_value):02d}"

    return re.sub(
        rf"\btwenty\s+twenty\s+({suffix_choices})\b",
        _replace,
        value,
        flags=re.IGNORECASE,
    )


def _parse_spoken_number(
    tokens: list[str],
    start: int,
    resources: NormalizationResources,
) -> tuple[str, str, int] | None:
    simple_words = dict(resources.number_words.get("simple", {}))
    tens_words = dict(resources.number_words.get("tens", {}))
    scale_words = dict(resources.number_words.get("scales", {}))
    total = 0
    current = 0
    decimal_digits: list[str] | None = None
    index = start
    saw_number_word = False

    while index < len(tokens):
        token = re.sub(r"[^a-z]", "", tokens[index].lower())
        if not token:
            break
        if token in simple_words:
            saw_number_word = True
            value = int(simple_words[token])
            if decimal_digits is not None:
                decimal_digits.append(str(value))
            else:
                current += value
            index += 1
            continue
        if token in tens_words:
            saw_number_word = True
            value = int(tens_words[token])
            if decimal_digits is not None:
                decimal_digits.append(str(value))
            else:
                current += value
            index += 1
            continue
        if token == "hundred" and decimal_digits is None:
            saw_number_word = True
            current = max(1, current) * 100
            index += 1
            continue
        if token in scale_words and decimal_digits is None:
            saw_number_word = True
            total += max(1, current) * int(scale_words[token])
            current = 0
            index += 1
            continue
        if token == "point" and decimal_digits is None and saw_number_word:
            decimal_digits = []
            total += current
            current = 0
            index += 1
            continue
        break

    consumed = index - start
    if not saw_number_word or consumed < 2:
        return None

    raw_text = " ".join(tokens[start:index])
    if decimal_digits is None:
        numeric_text = str(int(total + current))
    else:
        numeric_text = f"{int(total)}.{''.join(decimal_digits)}".rstrip("0").rstrip(".")
    return raw_text, numeric_text, consumed


def _expand_spoken_numbers(value: str, resources: NormalizationResources) -> str:
    tokens = value.split()
    if not tokens:
        return value

    normalized_tokens: list[str] = []
    index = 0
    while index < len(tokens):
        parsed = _parse_spoken_number(tokens, index, resources)
        if parsed is None:
            normalized_tokens.append(tokens[index])
            index += 1
            continue

        raw_text, numeric_text, consumed = parsed
        normalized_tokens.append(raw_text)
        if numeric_text and numeric_text != raw_text:
            normalized_tokens.append(numeric_text)
        index += consumed

    expanded = " ".join(normalized_tokens)
    scale_words = dict(resources.number_words.get("scales", {}))
    if not scale_words:
        return expanded
    scale_choices = "|".join(re.escape(key) for key in scale_words)

    def _replace(match: re.Match[str]) -> str:
        number_text = str(match.group(1))
        scale_word = str(match.group(2)).lower()
        factor = int(scale_words.get(scale_word, 1))
        try:
            expanded_number = int(float(number_text) * factor)
        except ValueError:
            return match.group(0)
        return f"{match.group(0)} {expanded_number}"

    return re.sub(
        rf"\b([0-9]+(?:\.[0-9]+)?)\s+({scale_choices})\b",
        _replace,
        expanded,
        flags=re.IGNORECASE,
    )


def _normalize_dense_core(text: str) -> str:
    value = normalize_text_for_search(text).lower()
    if not value:
        return value
    value = re.sub(r"([0-9])[, ]+([0-9])", r"\1\2", value)
    value = _collapse_repeated_characters(value)
    value = _collapse_repeated_tokens(value)
    return value


def normalize_text_for_dense(
    text: str,
    mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    resolved_mode = resolve_dense_normalization_mode(mode, resources=resources)
    adaptive_resources = resources or _fallback_resources()
    value = _normalize_dense_core(text)
    if resolved_mode == "adaptive_corpus":
        value = _apply_alias_map(value, adaptive_resources)
        value = _expand_spoken_years(value, adaptive_resources)
        value = _expand_spoken_numbers(value, adaptive_resources)
        value = _strip_filler_terms(value, adaptive_resources)
        value = _collapse_repeated_tokens(value)
    value = re.sub(r"[^0-9a-z\uac00-\ud7a3\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_text_for_reranker(
    text: str,
    mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    resolved_mode = resolve_dense_normalization_mode(mode, resources=resources)
    adaptive_resources = resources or _fallback_resources()
    value = _normalize_dense_core(text)
    if resolved_mode == "adaptive_corpus":
        value = _apply_alias_map(value, adaptive_resources)
        value = _expand_spoken_years(value, adaptive_resources)
        value = _expand_spoken_numbers(value, adaptive_resources)
        value = _strip_filler_terms(value, adaptive_resources)
    value = re.sub(r"[^0-9a-z\uac00-\ud7a3\s:|/\-]", " ", value)
    return normalize_text_for_search(value)


def prepare_query_for_dense(
    query: str,
    mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    return normalize_text_for_dense(query, mode=mode, resources=resources)


def prepare_query_for_reranker(
    query: str,
    mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    return normalize_text_for_reranker(query, mode=mode, resources=resources)


def _safe_value(row: pd.Series, column: str) -> str:
    value = row.get(column, "")
    if pd.isna(value):
        return ""
    return str(value)


def resolve_primary_text(row: pd.Series, text_source: str = DEFAULT_TEXT_SOURCE) -> str:
    validate_text_source(text_source)
    original_text = normalize_text_for_search(_safe_value(row, "original_transcript"))
    stt_text = normalize_text_for_search(_safe_value(row, "stt_transcript"))
    if text_source == "original_transcript":
        return original_text
    if text_source == "combined":
        return normalize_text_for_search(" ".join(part for part in [stt_text, original_text] if part))
    return stt_text or original_text


def _join_non_empty(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part).strip()


def build_search_text(
    row: pd.Series,
    text_source: str = DEFAULT_TEXT_SOURCE,
    for_dense: bool = False,
    normalization_mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    primary_text = resolve_primary_text(row, text_source=text_source)
    if not for_dense:
        return _join_non_empty(
            [
                normalize_text_for_search(_safe_value(row, "title")),
                normalize_text_for_search(_safe_value(row, "description")),
                normalize_text_for_search(_safe_value(row, "tags")),
                normalize_text_for_search(_safe_value(row, "category")),
                normalize_text_for_search(_safe_value(row, "keywords")),
                normalize_text_for_search(primary_text),
            ]
        )

    alternate_text_source = "original_transcript" if text_source != "original_transcript" else "stt_transcript"
    alternate_text = resolve_primary_text(row, text_source=alternate_text_source)
    dense_parts = [
        normalize_text_for_dense(_safe_value(row, "title"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(_safe_value(row, "description"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(_safe_value(row, "tags"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(_safe_value(row, "category"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(_safe_value(row, "keywords"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(primary_text, mode=normalization_mode, resources=resources),
    ]
    normalized_alternate = normalize_text_for_dense(alternate_text, mode=normalization_mode, resources=resources)
    if normalized_alternate:
        dense_parts.append(normalized_alternate)
    return _join_non_empty(dense_parts)


def _adaptive_preview_length(text: str, requested_length: int | None = None) -> int:
    if requested_length is not None:
        return max(40, int(requested_length))
    normalized = normalize_text_for_search(text)
    if not normalized:
        return 80
    sentence_lengths = [len(part) for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    avg_sentence = sum(sentence_lengths) / max(1, len(sentence_lengths)) if sentence_lengths else len(normalized)
    target = avg_sentence + max(40.0, len(normalized) / max(1.0, len(sentence_lengths) or 1.0))
    return int(max(80, round(target)))


def build_preview_text(row: pd.Series, text_source: str = DEFAULT_TEXT_SOURCE, length: int | None = None) -> str:
    preview = resolve_primary_text(row, text_source=text_source)
    resolved_length = _adaptive_preview_length(preview, requested_length=length)
    return preview[:resolved_length] + ("..." if len(preview) > resolved_length else "")


def split_line_into_sentences(text: str) -> list[str]:
    normalized = normalize_text_for_search(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [normalize_text_for_search(part) for part in parts if normalize_text_for_search(part)]
    return sentences or [normalized]


def _logical_line_limits(sentences: list[str]) -> tuple[int, int]:
    if not sentences:
        return 120, 2
    avg_sentence = sum(len(sentence) for sentence in sentences) / max(1, len(sentences))
    max_chars = int(max(120, round(avg_sentence * max(1.5, len(sentences) / max(1.0, math.log1p(len(sentences) + 1.0))))))
    max_sentences = max(2, int(round(len(sentences) / max(1.0, math.sqrt(len(sentences))))))
    return max_chars, max_sentences


def _chunk_sentences_into_logical_lines(sentences: list[str]) -> list[str]:
    if not sentences:
        return []
    max_chars, max_sentences = _logical_line_limits(sentences)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        projected_length = current_length + sentence_length + (1 if current_sentences else 0)
        if current_sentences and (len(current_sentences) >= max_sentences or projected_length > max_chars):
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_length = 0
        current_sentences.append(sentence)
        current_length += sentence_length + (1 if len(current_sentences) > 1 else 0)
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def split_text_into_lines(text: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    raw_lines = [
        (source_line_number, normalize_text_for_search(raw_line))
        for source_line_number, raw_line in enumerate(str(text).splitlines(), start=1)
    ]
    for source_line_number, line_text in raw_lines:
        if not line_text:
            continue
        logical_lines = _chunk_sentences_into_logical_lines(split_line_into_sentences(line_text)) or [line_text]
        for logical_line in logical_lines:
            rows.append(
                {
                    "line_number": len(rows) + 1,
                    "line_text": logical_line,
                    "source_line_number": source_line_number,
                }
            )
    return rows


def build_sentence_segments(text: str) -> list[dict[str, str | int]]:
    segments: list[dict[str, str | int]] = []
    for line in split_text_into_lines(text):
        sentences = split_line_into_sentences(str(line["line_text"]))
        for sentence_number, sentence_text in enumerate(sentences, start=1):
            segments.append(
                {
                    "line_number": int(line["line_number"]),
                    "source_line_number": int(line["source_line_number"]),
                    "sentence_number": sentence_number,
                    "line_text": str(line["line_text"]),
                    "sentence_text": sentence_text,
                }
            )
    return segments


def text_source_suffix(text_source: str) -> str:
    validate_text_source(text_source)
    return text_source.replace("/", "_")
