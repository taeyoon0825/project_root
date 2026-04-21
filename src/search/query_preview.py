from __future__ import annotations

import re
from typing import Any, Mapping

from src.search.match_locator import locate_best_dense_match, locate_best_keyword_match, simple_tokenize
from src.search.text_source import normalize_text_for_search


PREVIEW_LENGTH_RANGE = (80, 240)


def _payload_value(payload: Mapping[str, Any] | None, key: str) -> str:
    if not payload:
        return ""
    value = payload.get(key, "")
    if value is None:
        return ""
    return normalize_text_for_search(str(value))


def _resolve_preview_length(text: str, query: str, requested_length: int | None = None, search_mode: str = "generic") -> int:
    if requested_length is not None:
        return max(PREVIEW_LENGTH_RANGE[0], min(PREVIEW_LENGTH_RANGE[1], int(requested_length)))
    normalized = normalize_text_for_search(text)
    query_tokens = simple_tokenize(query)
    sentence_lengths = [len(part) for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    avg_sentence = sum(sentence_lengths) / max(1, len(sentence_lengths)) if sentence_lengths else len(normalized)
    base = 90 + (0.35 * min(avg_sentence, 180)) + (7 * len(query_tokens))
    if search_mode == "dense":
        base += 20
    return int(max(PREVIEW_LENGTH_RANGE[0], min(PREVIEW_LENGTH_RANGE[1], round(base))))


def _build_snippet(text: str, max_length: int, anchor_start: int | None = None, anchor_end: int | None = None) -> str:
    normalized = normalize_text_for_search(text)
    if not normalized:
        return ""
    if len(normalized) <= max_length:
        return normalized

    if anchor_start is None or anchor_end is None or anchor_start < 0 or anchor_end <= anchor_start:
        truncated = normalized[: max_length - 3].rstrip()
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        return truncated + "..."

    body_length = max(30, max_length - 8)
    anchor_size = anchor_end - anchor_start
    if anchor_size >= body_length:
        body = normalized[anchor_start : anchor_start + body_length].strip()
        return body + "..."

    remaining = body_length - anchor_size
    left_extra = remaining // 2
    right_extra = remaining - left_extra
    start = max(0, anchor_start - left_extra)
    end = min(len(normalized), anchor_end + right_extra)

    if start > 0:
        prev_space = normalized.rfind(" ", max(0, start - 20), start)
        if prev_space >= 0:
            start = prev_space + 1
    if end < len(normalized):
        next_space = normalized.find(" ", end, min(len(normalized), end + 20))
        if next_space >= 0:
            end = next_space

    body = normalized[start:end].strip()
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(normalized) else ""
    snippet = f"{prefix}{body}{suffix}".strip()
    if len(snippet) <= max_length:
        return snippet
    truncated = snippet[: max_length - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.rstrip(".") + "..."


def _find_anchor(text: str, anchor_text: str) -> tuple[int, int] | None:
    normalized_text = normalize_text_for_search(text)
    normalized_anchor = normalize_text_for_search(anchor_text)
    if not normalized_text or not normalized_anchor:
        return None

    start = normalized_text.lower().find(normalized_anchor.lower())
    if start < 0:
        return None
    return start, start + len(normalized_anchor)


def _best_keyword_window(text: str, query: str, max_length: int) -> str:
    normalized = normalize_text_for_search(text)
    if not normalized:
        return ""

    query_tokens = list(dict.fromkeys(simple_tokenize(query)))
    if not query_tokens:
        return ""

    lower_text = normalized.lower()
    occurrences: list[tuple[int, int, str]] = []
    for token in query_tokens:
        for match in re.finditer(re.escape(token.lower()), lower_text):
            occurrences.append((match.start(), match.end(), token))

    if not occurrences:
        return ""

    occurrences.sort(key=lambda item: (item[0], item[1]))
    window_size = max(max_length, max(100, max_length // 2))
    best_score: tuple[int, int, int] | None = None
    best_anchor = (occurrences[0][0], occurrences[0][1])

    for index, (start, _, _) in enumerate(occurrences):
        window_limit = start + window_size
        unique_tokens: set[str] = set()
        match_count = 0
        anchor_end = start
        for occ_start, occ_end, token in occurrences[index:]:
            if occ_start > window_limit:
                break
            unique_tokens.add(token)
            match_count += 1
            anchor_end = max(anchor_end, occ_end)
        score = (len(unique_tokens), match_count, -start)
        if best_score is None or score > best_score:
            best_score = score
            best_anchor = (start, anchor_end)

    return _build_snippet(normalized, max_length=max_length, anchor_start=best_anchor[0], anchor_end=best_anchor[1])


def _best_overlap_chunk(text: str, query: str, max_length: int) -> str:
    normalized = normalize_text_for_search(text)
    if not normalized:
        return ""

    query_tokens = set(simple_tokenize(query))
    if not query_tokens:
        return _build_snippet(normalized, max_length=max_length)

    stride = max(40, max_length // 2)
    best_chunk = normalized[:max_length]
    best_score = (-1, -1.0)
    for start in range(0, len(normalized), stride):
        chunk = normalized[start : start + max_length]
        if not chunk:
            continue
        chunk_tokens = set(simple_tokenize(chunk))
        overlap = len(query_tokens & chunk_tokens)
        density = overlap / max(1, len(chunk_tokens))
        score = (overlap, density)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    return _build_snippet(best_chunk, max_length=max_length)


def extract_keyword_preview(
    text: str,
    query: str,
    *,
    method: str = "bm25",
    length: int | None = None,
    match_payload: Mapping[str, Any] | None = None,
) -> str:
    normalized_text = normalize_text_for_search(text)
    preview_length = _resolve_preview_length(normalized_text, query, requested_length=length, search_mode="keyword")

    line_text = _payload_value(match_payload, "best_match_line_text")
    sentence_text = _payload_value(match_payload, "best_match_text")
    if not line_text and normalized_text and query:
        try:
            match_payload = locate_best_keyword_match(normalized_text, query, method=method)
            line_text = _payload_value(match_payload, "best_match_line_text")
            sentence_text = _payload_value(match_payload, "best_match_text")
        except Exception:
            line_text = ""
            sentence_text = ""

    candidate_text = line_text or sentence_text
    if candidate_text:
        keyword_window = _best_keyword_window(candidate_text, query, max_length=preview_length)
        if keyword_window:
            return keyword_window
        anchor = _find_anchor(candidate_text, sentence_text) if sentence_text else None
        if anchor is not None:
            return _build_snippet(candidate_text, max_length=preview_length, anchor_start=anchor[0], anchor_end=anchor[1])
        return _build_snippet(candidate_text, max_length=preview_length)

    if normalized_text:
        keyword_window = _best_keyword_window(normalized_text, query, max_length=preview_length)
        if keyword_window:
            return keyword_window
        return _best_overlap_chunk(normalized_text, query, max_length=preview_length)
    return ""


def extract_dense_preview(
    text: str,
    query: str,
    model: Any | None = None,
    *,
    length: int | None = None,
    match_payload: Mapping[str, Any] | None = None,
) -> str:
    normalized_text = normalize_text_for_search(text)
    preview_length = _resolve_preview_length(normalized_text, query, requested_length=length, search_mode="dense")

    line_text = _payload_value(match_payload, "best_match_line_text")
    sentence_text = _payload_value(match_payload, "best_match_text")
    if not line_text and normalized_text and query and model is not None:
        try:
            match_payload = locate_best_dense_match(normalized_text, query, model)
            line_text = _payload_value(match_payload, "best_match_line_text")
            sentence_text = _payload_value(match_payload, "best_match_text")
        except Exception:
            line_text = ""
            sentence_text = ""

    candidate_text = line_text or sentence_text
    if candidate_text:
        anchor = _find_anchor(candidate_text, sentence_text) if sentence_text else None
        if anchor is not None:
            return _build_snippet(candidate_text, max_length=preview_length, anchor_start=anchor[0], anchor_end=anchor[1])
        return _build_snippet(candidate_text, max_length=preview_length)

    if normalized_text:
        keyword_window = _best_keyword_window(normalized_text, query, max_length=preview_length)
        if keyword_window:
            return keyword_window
        return _best_overlap_chunk(normalized_text, query, max_length=preview_length)
    return ""
