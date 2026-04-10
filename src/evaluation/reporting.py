from __future__ import annotations

from typing import Any

import pandas as pd

from src.embedding.vector_models import list_available_models


def build_model_weight_frame(include_optional: bool = False) -> pd.DataFrame:
    rows = [
        {
            "component": "keyword",
            "weight": "none",
            "status": "independent",
            "note": "BM25와 TF-IDF를 각각 독립 시스템으로 평가하며 hybrid 가중치는 사용하지 않음",
        }
    ]
    for model_alias in list_available_models(include_optional=include_optional).keys():
        rows.append(
            {
                "component": model_alias,
                "weight": "none",
                "status": "independent",
                "note": "단일 dense 모델로 개별 평가, ensemble 또는 weighted fusion은 현재 비활성",
            }
        )
    rows.append(
        {
            "component": "rerank",
            "weight": "not enabled",
            "status": "disabled",
            "note": "reranker와 score fusion 단계는 현재 연결되어 있지 않음",
        }
    )
    return pd.DataFrame(rows)


def format_model_weight_lines(include_optional: bool = False) -> list[str]:
    frame = build_model_weight_frame(include_optional=include_optional)
    return [f"[MODEL] {row.component} weight: {row.weight} ({row.note})" for row in frame.itertuples(index=False)]


def format_query_console_report(detail_row: pd.Series, top_k: int) -> list[str]:
    return [
        f"[EVAL] Query: {detail_row.get('query', '')}",
        f"[EVAL] Ground Truth Rule: {detail_row.get('ground_truth_rule', '')}",
        f"[EVAL] Ground Truth IDs: [{detail_row.get('ground_truth_ids', '')}]",
        f"[EVAL] Ground Truth Files: [{detail_row.get('ground_truth_file_names', '')}]",
        f"[EVAL] Top-1: {detail_row.get('top1_file_name', '')} ({detail_row.get('top1_id', '')})",
        f"[EVAL] Top-1 Raw Score: {float(detail_row.get('top1_raw_score', 0.0)):.4f}",
        f"[EVAL] Top-1 Display Score: {float(detail_row.get('top1_display_score', 0.0)):.2f}",
        f"[EVAL] Top-{top_k} IDs: [{detail_row.get('topk_ids', '')}]",
        f"[EVAL] Top-{top_k} Raw Scores: [{detail_row.get('topk_raw_scores', '')}]",
        f"[EVAL] Top-{top_k} Display Scores: [{detail_row.get('topk_display_scores', '')}]",
        f"[EVAL] Precision@{top_k}: {float(detail_row.get('precision_at_k', 0.0)):.4f}",
        f"[EVAL] Recall@{top_k}: {float(detail_row.get('recall_at_k', 0.0)):.4f}",
        f"[EVAL] Accuracy@1: {float(detail_row.get('accuracy_at_1', 0.0)):.4f}",
        f"[EVAL] F1@{top_k}: {float(detail_row.get('f1_at_k', 0.0)):.4f}",
        f"[EVAL] File Match Top-1: {detail_row.get('top1_file_match', 'NO')}",
        f"[EVAL] Segment Match Top-1: {detail_row.get('top1_segment_match', 'not_scored')}",
        f"[EVAL] Hallucination: {detail_row.get('hallucination', 'NO')}",
        f"[EVAL] Hallucination Reason: {detail_row.get('hallucination_reason', '')}",
    ]


def format_summary_console_report(summary: pd.DataFrame, top_k: int) -> list[str]:
    if summary.empty:
        return ["[EVAL] 평가 결과가 비어 있습니다."]

    lines: list[str] = []
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                f"[EVAL] 시스템: {row.system_name} / text_source={row.text_source}",
                f"[EVAL] 기준: {getattr(row, 'evaluation_definition', '')}",
                f"[EVAL] Score kind: {getattr(row, 'score_kind', '')}",
                f"[EVAL] Macro Precision@{top_k}: {float(row.precision_at_k):.4f}",
                f"[EVAL] Macro Recall@{top_k}: {float(row.recall_at_k):.4f}",
                f"[EVAL] Accuracy@1: {float(row.accuracy_at_1):.4f}",
                f"[EVAL] Macro F1@{top_k}: {float(row.f1_at_k):.4f}",
                f"[EVAL] Micro Precision@{top_k}: {float(getattr(row, 'micro_precision_at_k', 0.0)):.4f}",
                f"[EVAL] Micro Recall@{top_k}: {float(getattr(row, 'micro_recall_at_k', 0.0)):.4f}",
                f"[EVAL] Micro F1@{top_k}: {float(getattr(row, 'micro_f1_at_k', 0.0)):.4f}",
                f"[EVAL] Hallucination Rate: {float(getattr(row, 'hallucination_rate', 0.0)):.4f}",
            ]
        )
    return lines


def compare_summary_frames(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before.empty or after.empty:
        return pd.DataFrame()

    join_columns = ["system_name", "system_type", "text_source"]
    metric_columns = [
        "precision_at_k",
        "recall_at_k",
        "accuracy_at_1",
        "f1_at_k",
        "hallucination_rate",
        "micro_precision_at_k",
        "micro_recall_at_k",
        "micro_f1_at_k",
    ]
    before_frame = before[join_columns + metric_columns].copy()
    before_frame = before_frame.rename(columns={column: f"{column}_before" for column in metric_columns})
    after_frame = after[join_columns + metric_columns].copy()
    after_frame = after_frame.rename(columns={column: f"{column}_after" for column in metric_columns})

    merged = before_frame.merge(after_frame, on=join_columns, how="outer")
    for column in metric_columns:
        merged[f"{column}_delta"] = merged[f"{column}_after"].fillna(0.0) - merged[f"{column}_before"].fillna(0.0)
    return merged.sort_values(join_columns, kind="stable").reset_index(drop=True)
