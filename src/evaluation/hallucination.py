from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.ground_truth import _split_values


DEFAULT_HALLUCINATION_THRESHOLD = 75.0


def _display_score(row: pd.Series) -> float:
    if "display_score" in row:
        return float(row.get("display_score", 0.0) or 0.0)
    if "similarity_score" in row:
        return float(row.get("similarity_score", 0.0) or 0.0)
    return 0.0


def detect_retrieval_hallucination(
    results: pd.DataFrame,
    query_row: pd.Series,
    top_k: int,
    threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
) -> dict[str, Any]:
    topk = results.head(top_k).reset_index(drop=True)
    if topk.empty:
        return {
            "hallucination": "NO",
            "hallucination_flag": 0,
            "hallucination_reason": "",
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
    if top1_id and relevant_ids and top1_id not in relevant_ids and top1_display_score >= threshold:
        reasons.append("irrelevant_top1_high_score")
    if target_category and top1_category and top1_category != target_category and top1_display_score >= threshold:
        reasons.append("wrong_category_high_rank")
    if target_source_type and top1_source_type and top1_source_type != target_source_type and top1_display_score >= threshold:
        reasons.append("wrong_source_type_high_rank")

    predicted_ids = set(topk["id"].fillna("").astype(str))
    has_relevant_in_topk = bool(predicted_ids & relevant_ids) if relevant_ids else False
    if relevant_ids and not has_relevant_in_topk and top1_display_score >= threshold:
        reasons.append("no_relevant_doc_in_topk")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "hallucination": "YES" if unique_reasons else "NO",
        "hallucination_flag": int(bool(unique_reasons)),
        "hallucination_reason": ", ".join(unique_reasons),
    }
