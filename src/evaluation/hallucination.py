from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.ground_truth import _split_values


FALLBACK_HALLUCINATION_THRESHOLD = 75.0
DEFAULT_HALLUCINATION_THRESHOLD = FALLBACK_HALLUCINATION_THRESHOLD


def _display_score(row: pd.Series) -> float:
    if "display_score" in row:
        return float(row.get("display_score", 0.0) or 0.0)
    if "similarity_score" in row:
        return float(row.get("similarity_score", 0.0) or 0.0)
    return 0.0


def _metric_threshold(metric_config: Any | None) -> float | None:
    if metric_config is None:
        return None
    value = getattr(metric_config, "hallucination_threshold", None)
    if value is None and isinstance(metric_config, dict):
        value = metric_config.get("hallucination_threshold")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_hallucination_threshold(
    results: pd.DataFrame,
    top_k: int,
    *,
    threshold: float | None = None,
    metric_config: Any | None = None,
) -> float:
    if threshold is not None:
        return float(threshold)

    config_threshold = _metric_threshold(metric_config)
    if config_threshold is not None:
        return config_threshold

    topk = results.head(top_k)
    if topk.empty:
        return FALLBACK_HALLUCINATION_THRESHOLD

    score_values = topk.apply(_display_score, axis=1).astype(float).to_numpy()
    score_values = score_values[np.isfinite(score_values)]
    if not len(score_values):
        return FALLBACK_HALLUCINATION_THRESHOLD

    median_score = float(np.median(score_values))
    q3 = float(np.quantile(score_values, 0.75))
    mad = float(np.median(np.abs(score_values - median_score)))
    dynamic_threshold = max(q3, median_score + (1.25 * mad))
    return float(max(55.0, min(95.0, dynamic_threshold)))


def detect_retrieval_hallucination(
    results: pd.DataFrame,
    query_row: pd.Series,
    top_k: int,
    threshold: float | None = None,
    metric_config: Any | None = None,
) -> dict[str, Any]:
    topk = results.head(top_k).reset_index(drop=True)
    threshold_used = resolve_hallucination_threshold(
        results,
        top_k,
        threshold=threshold,
        metric_config=metric_config,
    )
    if topk.empty:
        return {
            "hallucination": "NO",
            "hallucination_flag": 0,
            "hallucination_reason": "",
            "hallucination_threshold_used": threshold_used,
        }

    relevant_ids = set(_split_values(query_row.get("relevant_ids", "")))
    target_category = str(query_row.get("target_category", "")).strip()
    target_source_type = str(query_row.get("target_source_type", "")).strip()

    top1 = topk.iloc[0]
    top1_id = str(top1.get("id", "")).strip()
    top1_category = str(top1.get("category", "")).strip()
    top1_source_type = str(top1.get("source_type", "")).strip()
    top1_display_score = _display_score(top1)

    reasons: list[str] = []
    if top1_id and relevant_ids and top1_id not in relevant_ids and top1_display_score >= threshold_used:
        reasons.append("irrelevant_top1_high_score")
    if target_category and top1_category and top1_category != target_category and top1_display_score >= threshold_used:
        reasons.append("wrong_category_high_rank")
    if target_source_type and top1_source_type and top1_source_type != target_source_type and top1_display_score >= threshold_used:
        reasons.append("wrong_source_type_high_rank")

    predicted_ids = set(topk["id"].fillna("").astype(str))
    has_relevant_in_topk = bool(predicted_ids & relevant_ids) if relevant_ids else False
    if relevant_ids and not has_relevant_in_topk and top1_display_score >= threshold_used:
        reasons.append("no_relevant_doc_in_topk")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "hallucination": "YES" if unique_reasons else "NO",
        "hallucination_flag": int(bool(unique_reasons)),
        "hallucination_reason": ", ".join(unique_reasons),
        "hallucination_threshold_used": threshold_used,
    }
