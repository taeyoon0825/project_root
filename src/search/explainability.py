from __future__ import annotations

"""각 후보 행의 검색 점수가 어떻게 조립되었는지 설명 정보를 만든다.

검색 스택은 어휘 기반 카운트, 랭커 출력, 밀집 점수를 함께 섞는다.
이 모듈은 그 내부 신호를 UI와 디버그 도구가 공통으로 볼 수 있는
안정적인 설명 컬럼 집합으로 변환한다.
"""

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
    """설명 문자열이 읽기 쉬우도록 순서를 유지한 채 질의 토큰을 중복 제거한다."""
    seen: set[str] = set()
    tokens: list[str] = []
    for token in simple_tokenize(query):
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _safe_value(row: Mapping[str, Any] | pd.Series, column: str) -> str:
    """메타데이터 행이 NaN이나 object 타입을 섞어 가질 수 있으므로 안전하게 값을 읽는다."""
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
    """설명과 점수화에 참여하는 텍스트 필드를 한곳에서 수집한다.

    필드 추출 규칙을 중앙화해야 검색, UI, 디버그 내보내기가
    각 필드의 의미를 서로 다르게 해석하지 않는다.
    """
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
    """특정 필드 안에 실제로 존재하는 질의 토큰 부분집합만 반환한다."""
    field_tokens = set(simple_tokenize(text))
    return [token for token in tokens if token in field_tokens]


def _reason(
    counts: dict[str, int],
    ranker_score: float,
    semantic_score: float,
    score_kind: str,
    matched_tokens: list[str],
) -> str:
    """한 행에 대한 사람이 읽을 수 있는 짧은 설명 문자열을 만든다."""
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
    """질의와 필드의 겹침 정보를 바탕으로 행 단위 설명 피처를 계산한다.

    검색 엔진은 원시 랭커 점수나 의미 점수를 만든 뒤 이 함수를 호출한다.
    그래야 시스템 전체가 하나의 통일된 설명 스키마를 볼 수 있다.
    """
    query_tokens = unique_query_tokens(query)
    fields = metadata_field_texts(row, text_source=text_source)
    matches = {field: _matched_tokens(query_tokens, text) for field, text in fields.items()}
    counts = {field: len(tokens) for field, tokens in matches.items()}
    matched_tokens = list(dict.fromkeys(token for tokens in matches.values() for token in tokens))

    # 필드 가중치 병합을 여기서 수행해야 적응형 튜닝이 기본값을 덮더라도
    # 모든 호출부가 동일한 어휘 점수 계산 규칙을 공유할 수 있다.
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
    """행 순서를 유지한 채 DataFrame 전체에 explain_match를 적용한다."""
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
