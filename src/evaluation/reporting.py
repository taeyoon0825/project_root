from __future__ import annotations

import pandas as pd

from src.embedding.vector_models import list_available_models


def build_model_weight_frame(include_optional: bool = False) -> pd.DataFrame:
    rows = [
        {
            "component": "fused-retrieval",
            "weight": "adaptive",
            "status": "active",
            "note": "bm25 + MiniLM + E5 are normalized and fused with dataset/query-aware adaptive weights",
        },
        {
            "component": "keyword",
            "weight": "adaptive",
            "status": "active",
            "note": "field weights and keyword-vs-dense balance are resolved from dataset and query profiles at runtime",
        }
    ]
    for model_alias in list_available_models(include_optional=include_optional).keys():
        rows.append(
            {
                "component": model_alias,
                "weight": "adaptive",
                "status": "active",
                "note": "semantic weighting, preview length, clustering, and visualization parameters are resolved from profile statistics",
            }
        )
    rows.append(
        {
            "component": "static_reference",
            "weight": "reference-only",
            "status": "optional",
            "note": "used only when evaluation generates adaptive-vs-static comparison artifacts",
        }
    )
    return pd.DataFrame(rows)


def format_model_weight_lines(include_optional: bool = False) -> list[str]:
    frame = build_model_weight_frame(include_optional=include_optional)
    return [f"[MODEL] {row.component}: {row.weight} ({row.note})" for row in frame.itertuples(index=False)]


def format_query_console_report(detail_row: pd.Series, top_k: int) -> list[str]:
    return [
        f"[EVAL] Query: {detail_row.get('query', '')}",
        f"[EVAL] Parameter mode: {detail_row.get('parameter_mode', 'adaptive')}",
        f"[EVAL] Ground truth rule: {detail_row.get('ground_truth_rule', '')}",
        f"[EVAL] Ground truth IDs: [{detail_row.get('ground_truth_ids', '')}]",
        f"[EVAL] Top-1: {detail_row.get('top1_file_name', '')} ({detail_row.get('top1_id', '')})",
        f"[EVAL] Top-{top_k} IDs: [{detail_row.get('topk_ids', '')}]",
        f"[EVAL] Precision@{top_k}: {float(detail_row.get('precision_at_k', 0.0)):.4f}",
        f"[EVAL] Recall@{top_k}: {float(detail_row.get('recall_at_k', 0.0)):.4f}",
        f"[EVAL] Accuracy@1: {float(detail_row.get('accuracy_at_1', 0.0)):.4f}",
        f"[EVAL] F1@{top_k}: {float(detail_row.get('f1_at_k', 0.0)):.4f}",
        f"[EVAL] MRR@{top_k}: {float(detail_row.get('mrr_at_k', 0.0)):.4f}",
        f"[EVAL] nDCG@{top_k}: {float(detail_row.get('ndcg_at_k', 0.0)):.4f}",
        f"[EVAL] Soft Precision@{top_k}: {float(detail_row.get('soft_precision_at_k', 0.0)):.4f}",
        f"[EVAL] Soft Accuracy@1: {float(detail_row.get('soft_accuracy_at_1', 0.0)):.4f}",
        f"[EVAL] Hallucination: {detail_row.get('hallucination', 'NO')} "
        f"(threshold={float(detail_row.get('hallucination_threshold_used', 0.0)):.2f})",
        f"[EVAL] Adaptive search reason: {detail_row.get('adaptive_search_reason', '')}",
        f"[EVAL] Adaptive metric reason: {detail_row.get('adaptive_metric_reason', '')}",
    ]


def format_summary_console_report(summary: pd.DataFrame, top_k: int) -> list[str]:
    if summary.empty:
        return ["[EVAL] no evaluation rows were produced."]

    lines: list[str] = []
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                f"[EVAL] System: {row.system_name} / text_source={row.text_source} / mode={getattr(row, 'parameter_mode', 'adaptive')}",
                f"[EVAL] Query count: {int(getattr(row, 'query_count', 0))}",
                f"[EVAL] Resolved top_k: {int(getattr(row, 'top_k', 0))}",
                f"[EVAL] Macro Precision@{top_k}: {float(row.precision_at_k):.4f}",
                f"[EVAL] Macro Recall@{top_k}: {float(row.recall_at_k):.4f}",
                f"[EVAL] Accuracy@1: {float(row.accuracy_at_1):.4f}",
                f"[EVAL] Macro F1@{top_k}: {float(row.f1_at_k):.4f}",
                f"[EVAL] Macro MRR@{top_k}: {float(getattr(row, 'mrr_at_k', 0.0)):.4f}",
                f"[EVAL] Macro nDCG@{top_k}: {float(getattr(row, 'ndcg_at_k', 0.0)):.4f}",
                f"[EVAL] Soft F1@{top_k}: {float(getattr(row, 'soft_f1_at_k', 0.0)):.4f}",
                f"[EVAL] Hallucination Rate: {float(getattr(row, 'hallucination_rate', 0.0)):.4f}",
                f"[EVAL] Dataset profile: {getattr(row, 'dataset_profile_summary', '')}",
            ]
        )
    return lines


def compare_summary_frames(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before.empty or after.empty:
        return pd.DataFrame()

    join_columns = ["system_name", "system_type", "text_source"]
    metric_columns = [
        "top_k",
        "precision_at_k",
        "recall_at_k",
        "accuracy_at_1",
        "f1_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "soft_precision_at_k",
        "soft_recall_at_k",
        "soft_accuracy_at_1",
        "soft_f1_at_k",
        "hallucination_rate",
        "micro_precision_at_k",
        "micro_recall_at_k",
        "micro_f1_at_k",
    ]
    for frame in (before, after):
        for column in metric_columns:
            if column not in frame.columns:
                frame[column] = 0.0
    before_frame = before[join_columns + metric_columns].copy()
    before_frame = before_frame.rename(columns={column: f"{column}_before" for column in metric_columns})
    after_frame = after[join_columns + metric_columns].copy()
    after_frame = after_frame.rename(columns={column: f"{column}_after" for column in metric_columns})

    merged = before_frame.merge(after_frame, on=join_columns, how="outer")
    for column in metric_columns:
        if column == "top_k":
            continue
        merged[f"{column}_delta"] = merged[f"{column}_after"].fillna(0.0) - merged[f"{column}_before"].fillna(0.0)
    return merged.sort_values(join_columns, kind="stable").reset_index(drop=True)
