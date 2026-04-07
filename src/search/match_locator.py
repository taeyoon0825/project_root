from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.search.text_source import build_sentence_segments


def simple_tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[가-힣A-Za-z0-9]+", str(text).lower()) if token]


def _empty_match() -> dict[str, Any]:
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
    values = [float(score) for score in scores]
    if not values:
        return []
    min_score = min(values)
    max_score = max(values)
    if math.isclose(min_score, max_score):
        return [1.0 for _ in values]
    return [(score - min_score) / (max_score - min_score) for score in values]


def _format_location(line_number: int, sentence_number: int) -> str:
    return f"{line_number}번째 줄 / {sentence_number}번째 문장"


def _build_match_payload(segment: dict[str, Any], normalized_score: float) -> dict[str, Any]:
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
