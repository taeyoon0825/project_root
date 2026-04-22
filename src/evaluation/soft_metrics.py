from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.adaptive.tuning_config import load_tuning_config


TOKEN_RE = re.compile(r"[0-9A-Za-z\uac00-\ud7a3]+")
_TUNE = load_tuning_config().get("soft_metrics", {})
FALLBACK_SOFT_ACCURACY_WEIGHTS = _TUNE.get(
    "fallback_soft_accuracy_weights",
    {"exact_match": 0.25, "search_score": 0.25, "text_overlap": 0.25, "rank_weight": 0.25},
)
FALLBACK_SOFT_PRECISION_EXACT_WEIGHT = float(_TUNE.get("fallback_soft_precision_exact_weight", 0.5) or 0.5)
FALLBACK_SOFT_RECALL_EXACT_WEIGHT = float(_TUNE.get("fallback_soft_recall_exact_weight", 0.5) or 0.5)


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


def _metric_weights(metric_config: Any | None) -> tuple[dict[str, float], float, float]:
    weights = FALLBACK_SOFT_ACCURACY_WEIGHTS.copy()
    precision_exact_weight = FALLBACK_SOFT_PRECISION_EXACT_WEIGHT
    recall_exact_weight = FALLBACK_SOFT_RECALL_EXACT_WEIGHT

    if metric_config is None:
        return weights, precision_exact_weight, recall_exact_weight

    candidate_weights = getattr(metric_config, "soft_accuracy_weights", None)
    if candidate_weights is None and isinstance(metric_config, dict):
        candidate_weights = metric_config.get("soft_accuracy_weights")
    if isinstance(candidate_weights, Mapping):
        normalized = {
            key: max(0.0, float(candidate_weights.get(key, 0.0)))
            for key in FALLBACK_SOFT_ACCURACY_WEIGHTS
        }
        total = sum(normalized.values())
        if total > 0:
            weights = {key: value / total for key, value in normalized.items()}

    precision_value = getattr(metric_config, "soft_precision_exact_weight", None)
    recall_value = getattr(metric_config, "soft_recall_exact_weight", None)
    if isinstance(metric_config, dict):
        precision_value = metric_config.get("soft_precision_exact_weight", precision_value)
        recall_value = metric_config.get("soft_recall_exact_weight", recall_value)

    try:
        precision_exact_weight = float(precision_value)
    except (TypeError, ValueError):
        precision_exact_weight = FALLBACK_SOFT_PRECISION_EXACT_WEIGHT
    try:
        recall_exact_weight = float(recall_value)
    except (TypeError, ValueError):
        recall_exact_weight = FALLBACK_SOFT_RECALL_EXACT_WEIGHT

    return weights, clamp01(precision_exact_weight), clamp01(recall_exact_weight)


def _soft_accuracy_formula(weights: dict[str, float]) -> str:
    return (
        "soft_accuracy="
        f"{weights['exact_match']:.2f}*exact_match+"
        f"{weights['search_score']:.2f}*search_score+"
        f"{weights['text_overlap']:.2f}*text_overlap+"
        f"{weights['rank_weight']:.2f}*rank_weight"
    )


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
    metric_config: Any | None = None,
) -> dict[str, Any]:
    weights, _, _ = _metric_weights(metric_config)
    exact_match = 1.0 if is_relevant else 0.0
    search_score = score_to_unit(row)
    overlap = text_overlap_score(query, row)
    rank_weight = 1.0 / max(1, int(rank or 1))
    soft_accuracy = clamp01(
        (weights["exact_match"] * exact_match)
        + (weights["search_score"] * search_score)
        + (weights["text_overlap"] * overlap)
        + (weights["rank_weight"] * rank_weight)
    )
    reason = (
        f"exact_match={exact_match:.3f}, search_score={search_score:.3f}, "
        f"text_overlap={overlap:.3f}, rank_weight={rank_weight:.3f}, "
        f"{_soft_accuracy_formula(weights)} => {soft_accuracy:.3f}"
    )
    return {
        "exact_match": exact_match,
        "search_score_unit": search_score,
        "text_overlap": overlap,
        "rank_weight": rank_weight,
        "soft_accuracy": soft_accuracy,
        "confidence_reason": reason,
    }


def combine_soft_precision(
    exact_precision: float,
    mean_soft_accuracy: float,
    *,
    metric_config: Any | None = None,
) -> float:
    _, precision_exact_weight, _ = _metric_weights(metric_config)
    return clamp01((precision_exact_weight * exact_precision) + ((1.0 - precision_exact_weight) * mean_soft_accuracy))


def combine_soft_recall(
    exact_recall: float,
    soft_recall_evidence: float,
    *,
    metric_config: Any | None = None,
) -> float:
    _, _, recall_exact_weight = _metric_weights(metric_config)
    return clamp01((recall_exact_weight * exact_recall) + ((1.0 - recall_exact_weight) * soft_recall_evidence))


def summarize_soft_ranked_results(
    results: pd.DataFrame,
    query: Any,
    relevant_ids: set[str],
    top_k: int,
    *,
    metric_config: Any | None = None,
) -> dict[str, Any]:
    topk = results.head(top_k).reset_index(drop=True)
    weights, precision_exact_weight, recall_exact_weight = _metric_weights(metric_config)
    accuracy_formula = _soft_accuracy_formula(weights)
    precision_formula = (
        f"soft_precision={precision_exact_weight:.2f}*exact_precision+"
        f"{(1.0 - precision_exact_weight):.2f}*mean_soft_accuracy"
    )
    recall_formula = (
        f"soft_recall={recall_exact_weight:.2f}*exact_recall+"
        f"{(1.0 - recall_exact_weight):.2f}*soft_recall_evidence"
    )

    if topk.empty:
        return {
            "soft_precision_at_k": 0.0,
            "soft_recall_at_k": 0.0,
            "soft_accuracy_at_1": 0.0,
            "soft_f1_at_k": 0.0,
            "mean_soft_accuracy_at_k": 0.0,
            "soft_score_reason": f"{accuracy_formula}; {precision_formula}; {recall_formula}",
            "topk_soft_accuracy_scores": "",
        }

    soft_scores: list[float] = []
    exact_hits = 0
    top1_reason = ""
    for rank, row in enumerate(topk.to_dict(orient="records"), start=1):
        is_relevant = str(row.get("id", "")) in relevant_ids
        exact_hits += int(is_relevant)
        components = soft_match_components(
            row,
            query,
            is_relevant=is_relevant,
            rank=rank,
            metric_config=metric_config,
        )
        soft_scores.append(float(components["soft_accuracy"]))
        if rank == 1:
            top1_reason = str(components["confidence_reason"])

    predicted_count = len(topk)
    exact_precision = exact_hits / max(1, predicted_count)
    exact_recall = exact_hits / max(1, len(relevant_ids))
    mean_soft_accuracy = sum(soft_scores) / max(1, predicted_count)
    soft_recall_evidence = min(1.0, sum(soft_scores) / max(1, len(relevant_ids)))
    soft_precision = combine_soft_precision(
        exact_precision,
        mean_soft_accuracy,
        metric_config=metric_config,
    )
    soft_recall = combine_soft_recall(
        exact_recall,
        soft_recall_evidence,
        metric_config=metric_config,
    )
    soft_f1 = f1_score(soft_precision, soft_recall)

    return {
        "soft_precision_at_k": soft_precision,
        "soft_recall_at_k": soft_recall,
        "soft_accuracy_at_1": soft_scores[0],
        "soft_f1_at_k": soft_f1,
        "mean_soft_accuracy_at_k": mean_soft_accuracy,
        "soft_score_reason": f"{top1_reason}; {precision_formula}; {recall_formula}",
        "topk_soft_accuracy_scores": ", ".join(f"{score:.3f}" for score in soft_scores),
    }
