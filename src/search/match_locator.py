from __future__ import annotations

"""문서 내부에서 가장 근거가 되는 문장 또는 논리 줄을 찾는다.

검색 엔진은 먼저 문서 단위로 순위를 매긴다. 그 다음 UI는 왜 이 문서가 뽑혔는지
설명할 실제 위치가 필요하므로, 이 모듈에서 문장 조각을 다시 점수화해
어휘 기반 또는 밀집 기반 검색 근거와 연결되는 anchor를 고른다.
"""

import math
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.search.text_source import build_sentence_segments


def simple_tokenize(text: str) -> list[str]:
    """어휘 검색과 설명 로직이 같은 토큰 규칙을 쓰도록 공통 토크나이저를 둔다."""

    return [token for token in re.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", str(text).lower()) if token]


def _empty_match() -> dict[str, Any]:
    """선택 가능한 문장 조각이 없을 때 쓰는 표준 빈 payload를 반환한다."""

    return {
        "best_match_location": "",
        "best_match_line_number": 0,
        "best_match_source_line_number": 0,
        "best_match_sentence_number": 0,
        "best_match_similarity": 0.0,
        "best_match_text": "",
        "best_match_line_text": "",
    }


def _normalize_scores(scores: list[float] | np.ndarray) -> list[float]:
    """문장 조각 점수를 로컬 0..1 범위로 맞춰 UI 표시용 신뢰도로 사용한다."""

    values = [float(score) for score in scores]
    if not values:
        return []
    min_score = min(values)
    max_score = max(values)
    if math.isclose(min_score, max_score):
        return [1.0 for _ in values]
    return [(score - min_score) / (max_score - min_score) for score in values]


def _format_location(line_number: int, sentence_number: int) -> str:
    """대시보드가 바로 표시할 수 있는 짧은 위치 문자열을 만든다."""

    return f"{line_number}번째 줄 / {sentence_number}번째 문장"


def _build_match_payload(segment: dict[str, Any], normalized_score: float) -> dict[str, Any]:
    """선택된 문장 조각을 UI가 소비하는 표준 payload 형태로 투영한다."""

    line_number = int(segment["line_number"])
    sentence_number = int(segment["sentence_number"])
    return {
        "best_match_location": _format_location(line_number, sentence_number),
        "best_match_line_number": line_number,
        "best_match_source_line_number": int(segment["source_line_number"]),
        "best_match_sentence_number": sentence_number,
        "best_match_similarity": float(normalized_score),
        "best_match_text": str(segment["sentence_text"]),
        "best_match_line_text": str(segment["line_text"]),
    }


def locate_best_keyword_match(text: str, query: str, method: str = "bm25") -> dict[str, Any]:
    """문장별 어휘 점수화를 통해 키워드 검색 근거가 되는 위치를 찾는다.

    문장 수준에서도 BM25나 TF-IDF를 다시 쓰는 이유는, 미리보기가 단순히
    아무 문장이나 보여주는 것이 아니라 실제로 키워드 엔진이 문서를 높게 본
    이유와 연결된 문장을 보여줘야 하기 때문이다.
    """

    segments = build_sentence_segments(text)
    if not segments:
        return _empty_match()

    segment_texts = [str(segment["sentence_text"]) for segment in segments]
    if method.lower() == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=simple_tokenize, lowercase=True)
        matrix = vectorizer.fit_transform(segment_texts)
        query_vector = vectorizer.transform([query])
        scores = (matrix @ query_vector.T).toarray().ravel().tolist()
    else:
        tokenized = [simple_tokenize(segment_text) for segment_text in segment_texts]
        scores = list(BM25Okapi(tokenized).get_scores(simple_tokenize(query)))

    best_index = int(np.argmax(scores))
    normalized_scores = _normalize_scores(scores)
    return _build_match_payload(segments[best_index], normalized_scores[best_index])


def locate_best_dense_match(text: str, query: str, wrapper: Any) -> dict[str, Any]:
    """밀집 유사도로 가장 의미적으로 가까운 문장을 찾는다.

    밀집 검색은 문자 그대로의 토큰 겹침이 약해도 문서를 끌어올릴 수 있으므로,
    미리보기 anchor도 어휘 겹침만 보지 말고 문서 내부 문장을 의미 기반으로
    한 번 더 점수화해야 한다.
    """

    segments = build_sentence_segments(text)
    if not segments:
        return _empty_match()

    segment_texts = [str(segment["sentence_text"]) for segment in segments]
    segment_embeddings = wrapper.encode_documents(segment_texts, batch_size=min(16, len(segment_texts)))
    query_embedding = wrapper.encode_queries([query])[0]
    scores = (segment_embeddings @ query_embedding).tolist()

    best_index = int(np.argmax(scores))
    normalized_scores = _normalize_scores(scores)
    return _build_match_payload(segments[best_index], normalized_scores[best_index])
