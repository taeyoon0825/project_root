from __future__ import annotations

import re

import pandas as pd


SUPPORTED_TEXT_SOURCES = ("stt_transcript", "original_transcript", "combined")
DEFAULT_TEXT_SOURCE = "stt_transcript"
LOGICAL_LINE_MAX_CHARS = 180
LOGICAL_LINE_MAX_SENTENCES = 3


def validate_text_source(text_source: str) -> str:
    if text_source not in SUPPORTED_TEXT_SOURCES:
        raise ValueError(f"Unsupported text source: {text_source}")
    return text_source


def normalize_text_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


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
        combined = " ".join(part for part in [stt_text, original_text] if part)
        return normalize_text_for_search(combined)
    return stt_text or original_text


def build_search_text(row: pd.Series, text_source: str = DEFAULT_TEXT_SOURCE) -> str:
    primary_text = resolve_primary_text(row, text_source=text_source)
    return " ".join(
        [
            normalize_text_for_search(_safe_value(row, "title")),
            normalize_text_for_search(_safe_value(row, "description")),
            normalize_text_for_search(_safe_value(row, "tags")),
            normalize_text_for_search(_safe_value(row, "category")),
            normalize_text_for_search(_safe_value(row, "keywords")),
            primary_text,
        ]
    ).strip()


def build_preview_text(row: pd.Series, text_source: str = DEFAULT_TEXT_SOURCE, length: int = 180) -> str:
    preview = resolve_primary_text(row, text_source=text_source)
    return preview[:length] + ("..." if len(preview) > length else "")


def split_line_into_sentences(text: str) -> list[str]:
    normalized = normalize_text_for_search(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [normalize_text_for_search(part) for part in parts if normalize_text_for_search(part)]
    return sentences or [normalized]


def _chunk_sentences_into_logical_lines(
    sentences: list[str],
    max_chars: int = LOGICAL_LINE_MAX_CHARS,
    max_sentences: int = LOGICAL_LINE_MAX_SENTENCES,
) -> list[str]:
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        projected_length = current_length + sentence_length + (1 if current_sentences else 0)
        should_flush = current_sentences and (
            len(current_sentences) >= max_sentences or projected_length > max_chars
        )
        if should_flush:
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
