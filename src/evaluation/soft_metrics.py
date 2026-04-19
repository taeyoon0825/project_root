from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd


SOFT_ACCURACY_FORMULA = (
    "soft_accuracy=0.50*exact_match+0.30*search_score+0.15*text_overlap+0.05*rank_weight"
)
SOFT_PRECISION_FORMULA = "soft_precision=0.70*exact_precision+0.30*mean_soft_accuracy"
SOFT_RECALL_FORMULA = "soft_recall=0.70*exact_recall+0.30*soft_recall_evidence"
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def _get(row: Mapping[str, Any] | pd.Series, key: str, default: Any = "") -> Any:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def f1_score(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)


def score_to_unit(row: Mapping[str, Any] | pd.Series) -> float:
    normalized = _get(row, "normalized_score", None)
    if normalized not in (None, ""):
        return clamp01(normalized)

    for column in ("display_score", "similarity_score"):
        value = _get(row, column, None)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 1.0:
            return clamp01(number / 100.0)
        return clamp01(number)

    return 0.0


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _tokens(value: Any) -> list[str]:
    return TOKEN_RE.findall(str(value or "").lower())


def text_overlap_score(query: Any, row: Mapping[str, Any] | pd.Series) -> float:
    query_text = _normalize_text(query)
    if not query_text:
        return 0.0

    candidate_text = " ".join(
        _normalize_text(_get(row, column, ""))
        for column in ("best_match_text", "best_match_line_text", "preview", "transcript_preview", "title", "file_name")
    ).strip()
    if not candidate_text:
        return 0.0

    exact_phrase = 1.0 if query_text in candidate_text or candidate_text in query_text else 0.0
    query_tokens = _tokens(query_text)
    candidate_tokens = set(_tokens(candidate_text))
    token_overlap = 0.0
    if query_tokens:
        token_overlap = sum(1 for token in query_tokens if token in candidate_tokens) / max(1, len(query_tokens))

    best_match_similarity = clamp01(_get(row, "best_match_similarity", 0.0))
    return clamp01(max(exact_phrase, token_overlap, best_match_similarity * 0.5))


def soft_match_components(
    row: Mapping[str, Any] | pd.Series,
    query: Any,
    *,
    is_relevant: bool,
    rank: int,
) -> dict[str, Any]:
    exact_match = 1.0 if is_relevant else 0.0
    search_score = score_to_unit(row)
    overlap = text_overlap_score(query, row)
    rank_weight = 1.0 / max(1, int(rank or 1))
    soft_accuracy = clamp01(
        (0.50 * exact_match)
        + (0.30 * search_score)
        + (0.15 * overlap)
        + (0.05 * rank_weight)
    )
    reason = (
        f"exact_match={exact_match:.3f}, search_score={search_score:.3f}, "
        f"text_overlap={overlap:.3f}, rank_weight={rank_weight:.3f}, "
        f"{SOFT_ACCURACY_FORMULA} => {soft_accuracy:.3f}"
    )
    return {
        "exact_match": exact_match,
        "search_score_unit": search_score,
        "text_overlap": overlap,
        "rank_weight": rank_weight,
        "soft_accuracy": soft_accuracy,
        "confidence_reason": reason,
    }


def combine_soft_precision(exact_precision: float, mean_soft_accuracy: float) -> float:
    return clamp01((0.70 * exact_precision) + (0.30 * mean_soft_accuracy))


def combine_soft_recall(exact_recall: float, soft_recall_evidence: float) -> float:
    return clamp01((0.70 * exact_recall) + (0.30 * soft_recall_evidence))


def summarize_soft_ranked_results(
    results: pd.DataFrame,
    query: Any,
    relevant_ids: set[str],
    top_k: int,
) -> dict[str, Any]:
    topk = results.head(top_k).reset_index(drop=True)
    if topk.empty:
        return {
            "soft_precision_at_k": 0.0,
            "soft_recall_at_k": 0.0,
            "soft_accuracy_at_1": 0.0,
            "soft_f1_at_k": 0.0,
            "mean_soft_accuracy_at_k": 0.0,
            "soft_score_reason": SOFT_ACCURACY_FORMULA,
            "topk_soft_accuracy_scores": "",
        }

    soft_scores: list[float] = []
    exact_hits = 0
    top1_reason = ""
    for rank, row in enumerate(topk.to_dict(orient="records"), start=1):
        is_relevant = str(row.get("id", "")) in relevant_ids
        exact_hits += int(is_relevant)
        components = soft_match_components(row, query, is_relevant=is_relevant, rank=rank)
        soft_scores.append(float(components["soft_accuracy"]))
        if rank == 1:
            top1_reason = str(components["confidence_reason"])

    predicted_count = len(topk)
    exact_precision = exact_hits / max(1, predicted_count)
    exact_recall = exact_hits / max(1, len(relevant_ids))
    mean_soft_accuracy = sum(soft_scores) / max(1, predicted_count)
    soft_recall_evidence = min(1.0, sum(soft_scores) / max(1, len(relevant_ids)))
    soft_precision = combine_soft_precision(exact_precision, mean_soft_accuracy)
    soft_recall = combine_soft_recall(exact_recall, soft_recall_evidence)
    soft_f1 = f1_score(soft_precision, soft_recall)

    return {
        "soft_precision_at_k": soft_precision,
        "soft_recall_at_k": soft_recall,
        "soft_accuracy_at_1": soft_scores[0],
        "soft_f1_at_k": soft_f1,
        "mean_soft_accuracy_at_k": mean_soft_accuracy,
        "soft_score_reason": f"{top1_reason}; {SOFT_PRECISION_FORMULA}; {SOFT_RECALL_FORMULA}",
        "topk_soft_accuracy_scores": ", ".join(f"{score:.3f}" for score in soft_scores),
    }
