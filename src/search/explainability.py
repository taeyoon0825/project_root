from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.search.match_locator import simple_tokenize
from src.search.text_source import DEFAULT_TEXT_SOURCE, resolve_primary_text


FIELD_WEIGHTS = {
    "title": 5.0,
    "tags": 4.0,
    "description": 3.0,
    "transcript": 2.0,
}
RANKER_SCORE_WEIGHT = 3.0
SEMANTIC_SCORE_WEIGHT = 5.0


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
    field_labels = {
        "title": "제목",
        "tags": "태그",
        "description": "설명",
        "transcript": "자막",
    }
    matched_fields = [field_labels[field] for field, count in counts.items() if count > 0]

    parts: list[str] = []
    if matched_fields:
        parts.append(f"{'·'.join(matched_fields)} metadata에서 검색어 토큰이 확인됨")
    if counts.get("title", 0) or counts.get("tags", 0):
        parts.append("제목/태그처럼 가중치가 높은 영역이 점수에 크게 반영됨")
    if ranker_score >= RANKER_SCORE_WEIGHT * 0.7:
        parts.append(f"{score_kind.upper()} 보정 점수가 높음")
    elif ranker_score > 0:
        parts.append(f"{score_kind.upper()} 보정 점수가 일부 반영됨")
    if semantic_score >= SEMANTIC_SCORE_WEIGHT * 0.7:
        parts.append("임베딩 의미 유사도가 높음")
    elif semantic_score > 0:
        parts.append("임베딩 의미 유사도가 일부 반영됨")
    if not parts and matched_tokens:
        parts.append("낮은 빈도의 metadata 토큰 매칭으로 후보에 포함됨")
    if not parts:
        parts.append("직접 토큰 매칭은 약하지만 보조 점수로 후보에 포함됨")
    return " / ".join(parts)


def explain_match(
    row: Mapping[str, Any] | pd.Series,
    query: str,
    *,
    text_source: str = DEFAULT_TEXT_SOURCE,
    ranker_score: float = 0.0,
    semantic_score: float = 0.0,
    score_kind: str = "bm25",
) -> dict[str, Any]:
    query_tokens = unique_query_tokens(query)
    fields = metadata_field_texts(row, text_source=text_source)
    matches = {field: _matched_tokens(query_tokens, text) for field, text in fields.items()}
    counts = {field: len(tokens) for field, tokens in matches.items()}
    matched_tokens = list(dict.fromkeys(token for tokens in matches.values() for token in tokens))

    field_weight_score = sum(FIELD_WEIGHTS[field] * counts[field] for field in FIELD_WEIGHTS)
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
            )
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
