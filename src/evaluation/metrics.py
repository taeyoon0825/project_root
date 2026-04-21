from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.evaluation.ground_truth import _split_int_values, _split_values
from src.evaluation.soft_metrics import summarize_soft_ranked_results


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _display_score(row: pd.Series) -> float:
    if "display_score" in row:
        return float(row.get("display_score", 0.0) or 0.0)
    if "similarity_score" in row:
        return float(row.get("similarity_score", 0.0) or 0.0)
    return 0.0


def _raw_score(row: pd.Series) -> float:
    return float(row.get("raw_score", 0.0) or 0.0)


def _dcg(relevances: list[int]) -> float:
    score = 0.0
    for rank, relevance in enumerate(relevances, start=1):
        score += float(relevance) / math.log2(rank + 1.0)
    return score


def ground_truth_payload(query_row: pd.Series) -> dict[str, Any]:
    return {
        "relevant_ids": set(_split_values(query_row.get("relevant_ids", ""))),
        "relevant_file_names": set(_split_values(query_row.get("relevant_file_names", ""))),
        "relevant_line_numbers": set(_split_int_values(query_row.get("relevant_line_numbers", ""))),
        "relevant_segment_indexes": set(_split_int_values(query_row.get("relevant_segment_indexes", ""))),
        "relevant_segment_texts": [_normalize_text(value) for value in _split_values(query_row.get("relevant_segment_texts", ""))],
        "target_category": str(query_row.get("target_category", "")).strip(),
        "target_source_type": str(query_row.get("target_source_type", "")).strip(),
        "evaluation_level": str(query_row.get("evaluation_level", "file")).strip() or "file",
        "ground_truth_rule": str(query_row.get("ground_truth_rule", "")).strip() or "derived_from_metadata",
    }


def _segment_match(result_row: pd.Series, ground_truth: dict[str, Any]) -> tuple[str, str]:
    if ground_truth["evaluation_level"] != "segment":
        return "not_scored", ""

    if str(result_row.get("id", "")) not in ground_truth["relevant_ids"]:
        return "none", "file_mismatch"

    line_number = int(result_row.get("best_match_source_line_number", 0) or 0)
    candidate_text = _normalize_text(result_row.get("best_match_text", ""))
    if ground_truth["relevant_line_numbers"] and line_number in ground_truth["relevant_line_numbers"]:
        return "exact", "matched_line_number"

    if candidate_text and ground_truth["relevant_segment_texts"]:
        for segment_text in ground_truth["relevant_segment_texts"]:
            if segment_text and (segment_text in candidate_text or candidate_text in segment_text):
                return "exact", "matched_segment_text"

    return "partial", "file_match_but_segment_miss"


def evaluate_ranked_results(
    results: pd.DataFrame,
    query_row: pd.Series,
    top_k: int,
    *,
    metric_config: Any | None = None,
) -> dict[str, Any]:
    topk = results.head(top_k).reset_index(drop=True)
    ground_truth = ground_truth_payload(query_row)
    relevant_ids = ground_truth["relevant_ids"]

    predicted_ids = topk["id"].fillna("").astype(str).tolist() if not topk.empty else []
    predicted_categories = topk["category"].fillna("").astype(str).tolist() if "category" in topk.columns else []
    predicted_file_names = topk["file_name"].fillna("").astype(str).tolist() if "file_name" in topk.columns else []

    tp = sum(1 for doc_id in predicted_ids if doc_id in relevant_ids)
    fp = max(0, len(predicted_ids) - tp)
    fn = max(0, len(relevant_ids) - tp)
    precision = tp / max(1, len(predicted_ids))
    recall = tp / max(1, len(relevant_ids))
    accuracy = float(bool(predicted_ids and predicted_ids[0] in relevant_ids))
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    soft_metrics = summarize_soft_ranked_results(
        topk,
        query_row.get("query", ""),
        relevant_ids,
        top_k=top_k,
        metric_config=metric_config,
    )

    reciprocal_rank = 0.0
    binary_relevance: list[int] = []
    for rank, doc_id in enumerate(predicted_ids, start=1):
        is_relevant = int(doc_id in relevant_ids)
        binary_relevance.append(is_relevant)
        if not reciprocal_rank and is_relevant:
            reciprocal_rank = 1.0 / rank
    ideal_relevance = [1] * min(len(relevant_ids), len(binary_relevance))
    ndcg = 0.0
    if binary_relevance:
        ideal_dcg = _dcg(ideal_relevance)
        ndcg = 0.0 if ideal_dcg == 0 else _dcg(binary_relevance) / ideal_dcg

    top1_row = topk.iloc[0] if not topk.empty else pd.Series(dtype=object)
    top1_display_score = _display_score(top1_row) if not topk.empty else 0.0
    top1_raw_score = _raw_score(top1_row) if not topk.empty else 0.0
    top1_final_score = float(top1_row.get("final_score", top1_raw_score) or 0.0) if not topk.empty else 0.0

    top1_segment_match, top1_segment_reason = _segment_match(top1_row, ground_truth) if not topk.empty else ("none", "")
    topk_segment_matches = []
    for _, result_row in topk.iterrows():
        match_type, _ = _segment_match(result_row, ground_truth)
        topk_segment_matches.append(match_type)

    return {
        "relevant_count": len(relevant_ids),
        "tp_at_k": tp,
        "fp_at_k": fp,
        "fn_at_k": fn,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "accuracy_at_1": accuracy,
        "f1_at_k": f1,
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": ndcg,
        **soft_metrics,
        "topk_hit_rate": float(tp > 0),
        "top1_id": str(top1_row.get("id", "")) if not topk.empty else "",
        "top1_file_name": str(top1_row.get("file_name", "")) if not topk.empty else "",
        "top1_category": str(top1_row.get("category", "")) if not topk.empty else "",
        "top1_source_type": str(top1_row.get("source_type", "")) if not topk.empty else "",
        "top1_raw_score": top1_raw_score,
        "top1_display_score": top1_display_score,
        "top1_final_score": top1_final_score,
        "top1_matched_tokens": str(top1_row.get("matched_tokens", "")) if not topk.empty else "",
        "top1_title_match_count": int(top1_row.get("title_match_count", 0) or 0) if not topk.empty else 0,
        "top1_description_match_count": int(top1_row.get("description_match_count", 0) or 0) if not topk.empty else 0,
        "top1_tags_match_count": int(top1_row.get("tags_match_count", 0) or 0) if not topk.empty else 0,
        "top1_transcript_match_count": int(top1_row.get("transcript_match_count", 0) or 0) if not topk.empty else 0,
        "top1_lexical_score": float(top1_row.get("lexical_score", 0.0) or 0.0) if not topk.empty else 0.0,
        "top1_semantic_score": float(top1_row.get("semantic_score", 0.0) or 0.0) if not topk.empty else 0.0,
        "top1_reason": str(top1_row.get("reason", "")) if not topk.empty else "",
        "adaptive_field_weights": str(top1_row.get("adaptive_field_weights", "")) if not topk.empty else "",
        "adaptive_keyword_alpha": float(top1_row.get("adaptive_keyword_alpha", 0.0) or 0.0) if not topk.empty else 0.0,
        "adaptive_dense_alpha": float(top1_row.get("adaptive_dense_alpha", 0.0) or 0.0) if not topk.empty else 0.0,
        "adaptive_reason": str(top1_row.get("adaptive_reason", "")) if not topk.empty else "",
        "mean_topk_raw_score": float(topk["raw_score"].astype(float).mean()) if "raw_score" in topk.columns and not topk.empty else 0.0,
        "mean_topk_final_score": float(topk["final_score"].astype(float).mean()) if "final_score" in topk.columns and not topk.empty else 0.0,
        "mean_topk_display_score": float(topk["display_score"].astype(float).mean()) if "display_score" in topk.columns and not topk.empty else float(topk["similarity_score"].astype(float).mean()) if "similarity_score" in topk.columns and not topk.empty else 0.0,
        "topk_ids": ", ".join(predicted_ids),
        "topk_file_names": ", ".join(predicted_file_names),
        "topk_categories": ", ".join(predicted_categories),
        "topk_raw_scores": ", ".join(f"{float(score):.4f}" for score in topk["raw_score"].tolist()) if "raw_score" in topk.columns else "",
        "topk_final_scores": ", ".join(f"{float(score):.4f}" for score in topk["final_score"].tolist()) if "final_score" in topk.columns else "",
        "topk_display_scores": ", ".join(f"{float(score):.2f}" for score in (topk["display_score"] if "display_score" in topk.columns else topk.get("similarity_score", pd.Series(dtype=float))).tolist()) if not topk.empty else "",
        "topk_soft_accuracy_scores": soft_metrics["topk_soft_accuracy_scores"],
        "ground_truth_ids": ", ".join(sorted(relevant_ids)),
        "ground_truth_file_names": ", ".join(sorted(ground_truth["relevant_file_names"])),
        "ground_truth_line_numbers": ", ".join(map(str, sorted(ground_truth["relevant_line_numbers"]))),
        "ground_truth_segment_indexes": ", ".join(map(str, sorted(ground_truth["relevant_segment_indexes"]))),
        "ground_truth_segment_texts": " | ".join(ground_truth["relevant_segment_texts"]),
        "ground_truth_rule": ground_truth["ground_truth_rule"],
        "evaluation_level": ground_truth["evaluation_level"],
        "top1_file_match": "YES" if accuracy else "NO",
        "top1_segment_match": top1_segment_match,
        "top1_segment_reason": top1_segment_reason,
        "topk_segment_exact_hit": int(any(match == "exact" for match in topk_segment_matches)),
        "topk_segment_partial_hit": int(any(match in {"exact", "partial"} for match in topk_segment_matches)),
        "score_kind": str(top1_row.get("score_kind", "")) if not topk.empty else "",
        "raw_score_explanation": str(top1_row.get("raw_score_explanation", "")) if not topk.empty else "",
    }


def aggregate_metric_rows(detail: pd.DataFrame, top_k: int) -> dict[str, Any]:
    if detail.empty:
        return {
            "macro_precision_at_k": 0.0,
            "macro_recall_at_k": 0.0,
            "macro_accuracy_at_1": 0.0,
            "macro_f1_at_k": 0.0,
            "macro_mrr_at_k": 0.0,
            "macro_ndcg_at_k": 0.0,
            "macro_soft_precision_at_k": 0.0,
            "macro_soft_recall_at_k": 0.0,
            "macro_soft_accuracy_at_1": 0.0,
            "macro_soft_f1_at_k": 0.0,
            "macro_mean_soft_accuracy_at_k": 0.0,
            "micro_precision_at_k": 0.0,
            "micro_recall_at_k": 0.0,
            "micro_f1_at_k": 0.0,
            "hallucination_rate": 0.0,
            "topk_hit_rate": 0.0,
            f"top{top_k}_accuracy": 0.0,
            "segment_exact_hit_rate": 0.0,
            "segment_partial_hit_rate": 0.0,
            "mean_top1_display_score": 0.0,
            "mean_hallucination_threshold": 0.0,
        }

    tp = float(detail["tp_at_k"].sum())
    fp = float(detail["fp_at_k"].sum())
    fn = float(detail["fn_at_k"].sum())
    micro_precision = tp / max(1.0, tp + fp)
    micro_recall = tp / max(1.0, tp + fn)
    micro_f1 = 0.0 if micro_precision + micro_recall == 0 else (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    return {
        "macro_precision_at_k": float(detail["precision_at_k"].mean()),
        "macro_recall_at_k": float(detail["recall_at_k"].mean()),
        "macro_accuracy_at_1": float(detail["accuracy_at_1"].mean()),
        "macro_f1_at_k": float(detail["f1_at_k"].mean()),
        "macro_mrr_at_k": float(detail["mrr_at_k"].mean()) if "mrr_at_k" in detail.columns else 0.0,
        "macro_ndcg_at_k": float(detail["ndcg_at_k"].mean()) if "ndcg_at_k" in detail.columns else 0.0,
        "macro_soft_precision_at_k": float(detail["soft_precision_at_k"].mean()) if "soft_precision_at_k" in detail.columns else 0.0,
        "macro_soft_recall_at_k": float(detail["soft_recall_at_k"].mean()) if "soft_recall_at_k" in detail.columns else 0.0,
        "macro_soft_accuracy_at_1": float(detail["soft_accuracy_at_1"].mean()) if "soft_accuracy_at_1" in detail.columns else 0.0,
        "macro_soft_f1_at_k": float(detail["soft_f1_at_k"].mean()) if "soft_f1_at_k" in detail.columns else 0.0,
        "macro_mean_soft_accuracy_at_k": float(detail["mean_soft_accuracy_at_k"].mean()) if "mean_soft_accuracy_at_k" in detail.columns else 0.0,
        "micro_precision_at_k": micro_precision,
        "micro_recall_at_k": micro_recall,
        "micro_f1_at_k": micro_f1,
        "hallucination_rate": float(detail["hallucination_flag"].mean()) if "hallucination_flag" in detail.columns else 0.0,
        "topk_hit_rate": float(detail["topk_hit_rate"].mean()),
        f"top{top_k}_accuracy": float(detail["topk_hit_rate"].mean()),
        "segment_exact_hit_rate": float(detail["topk_segment_exact_hit"].mean()) if "topk_segment_exact_hit" in detail.columns else 0.0,
        "segment_partial_hit_rate": float(detail["topk_segment_partial_hit"].mean()) if "topk_segment_partial_hit" in detail.columns else 0.0,
        "mean_top1_display_score": float(detail["top1_display_score"].mean()) if "top1_display_score" in detail.columns else 0.0,
        "mean_hallucination_threshold": float(detail["hallucination_threshold_used"].mean()) if "hallucination_threshold_used" in detail.columns else 0.0,
    }
