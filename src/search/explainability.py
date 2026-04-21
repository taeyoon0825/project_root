from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.search.match_locator import simple_tokenize
from src.search.text_source import DEFAULT_TEXT_SOURCE, resolve_primary_text


DEFAULT_FIELD_WEIGHTS = {
    "title": 1.0,
    "tags": 1.0,
    "description": 1.0,
    "transcript": 1.0,
}


def unique_query_tokens(query: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in simple_tokenize(query):
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _safe_value(row: Mapping[str, Any] | pd.Series, column: str) -> str:
    value = row.get(column, "") if isinstance(row, pd.Series) else row.get(column, "")
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def metadata_field_texts(
    row: Mapping[str, Any] | pd.Series,
    text_source: str = DEFAULT_TEXT_SOURCE,
) -> dict[str, str]:
    tags = _safe_value(row, "tags") or _safe_value(row, "keywords")
    transcript = _safe_value(row, "primary_text")
    if not transcript:
        transcript = resolve_primary_text(pd.Series(dict(row)), text_source=text_source)
    return {
        "title": _safe_value(row, "title"),
        "description": _safe_value(row, "description"),
        "tags": tags,
        "transcript": transcript,
    }


def _matched_tokens(tokens: list[str], text: str) -> list[str]:
    field_tokens = set(simple_tokenize(text))
    return [token for token in tokens if token in field_tokens]


def _reason(
    counts: dict[str, int],
    ranker_score: float,
    semantic_score: float,
    score_kind: str,
    matched_tokens: list[str],
) -> str:
    matched_fields = [field for field, count in counts.items() if count > 0]
    parts: list[str] = []
    if matched_fields:
        parts.append(f"matched_fields={','.join(matched_fields)}")
    if matched_tokens:
        parts.append(f"matched_tokens={','.join(matched_tokens[:8])}")
    if ranker_score > 0:
        parts.append(f"{score_kind}_score={ranker_score:.3f}")
    if semantic_score > 0:
        parts.append(f"semantic_score={semantic_score:.3f}")
    if not parts:
        parts.append("retrieved_by_supporting_score")
    return " / ".join(parts)


def explain_match(
    row: Mapping[str, Any] | pd.Series,
    query: str,
    *,
    text_source: str = DEFAULT_TEXT_SOURCE,
    ranker_score: float = 0.0,
    semantic_score: float = 0.0,
    score_kind: str = "bm25",
    field_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    query_tokens = unique_query_tokens(query)
    fields = metadata_field_texts(row, text_source=text_source)
    matches = {field: _matched_tokens(query_tokens, text) for field, text in fields.items()}
    counts = {field: len(tokens) for field, tokens in matches.items()}
    matched_tokens = list(dict.fromkeys(token for tokens in matches.values() for token in tokens))

    resolved_weights = {**DEFAULT_FIELD_WEIGHTS, **(field_weights or {})}
    field_weight_score = sum(resolved_weights[field] * counts[field] for field in DEFAULT_FIELD_WEIGHTS)
    lexical_score = float(field_weight_score) + float(ranker_score)
    final_score = lexical_score + float(semantic_score)
    return {
        "matched_tokens": ", ".join(matched_tokens),
        "title_match_count": counts["title"],
        "description_match_count": counts["description"],
        "tags_match_count": counts["tags"],
        "transcript_match_count": counts["transcript"],
        "field_weight_score": float(field_weight_score),
        "ranker_score": float(ranker_score),
        "lexical_score": float(lexical_score),
        "semantic_score": float(semantic_score),
        "final_score": float(final_score),
        "reason": _reason(counts, float(ranker_score), float(semantic_score), score_kind, matched_tokens),
    }


def explain_frame(
    frame: pd.DataFrame,
    query: str,
    *,
    text_source: str = DEFAULT_TEXT_SOURCE,
    ranker_scores: list[float] | pd.Series | None = None,
    semantic_scores: list[float] | pd.Series | None = None,
    score_kind: str = "bm25",
    field_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    ranker_values = list(ranker_scores) if ranker_scores is not None else [0.0] * len(frame)
    semantic_values = list(semantic_scores) if semantic_scores is not None else [0.0] * len(frame)
    rows = []
    for index, (_, row) in enumerate(frame.iterrows()):
        rows.append(
            explain_match(
                row,
                query,
                text_source=text_source,
                ranker_score=float(ranker_values[index]),
                semantic_score=float(semantic_values[index]),
                score_kind=score_kind,
                field_weights=field_weights,
            )
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
