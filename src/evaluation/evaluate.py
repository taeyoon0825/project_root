from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from src.adaptive.parameter_resolver import (
    AdaptiveContext,
    build_adaptive_context,
    build_static_reference_context,
    resolve_metric_config,
    resolve_top_k,
)
from src.config import DEFAULT_METADATA_CSV, DEFAULT_QUERYSET_CSV, EVALUATION_DIR, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame
from src.embedding.build_indices import DenseSearchEngine
from src.embedding.vector_models import list_available_models
from src.evaluation.ground_truth import (
    build_incremental_probe_queryset,
    evaluation_definition_text,
    normalize_ground_truth_queryset,
)
from src.evaluation.hallucination import detect_retrieval_hallucination
from src.evaluation.metrics import aggregate_metric_rows, evaluate_ranked_results
from src.evaluation.reporting import (
    build_model_weight_frame,
    compare_summary_frames,
    format_model_weight_lines,
    format_query_console_report,
    format_summary_console_report,
)
from src.retrieval.fused_search import FusedSearchEngine
from src.search.keyword_search import KeywordSearchEngine
from src.search.text_source import DEFAULT_TEXT_SOURCE
from src.utils.io_utils import save_dataframe, save_json


def evaluation_artifact_path(filename: str, artifact_namespace: str | None = None) -> Path:
    if not artifact_namespace:
        return EVALUATION_DIR / filename
    safe_namespace = re.sub(r"[^0-9A-Za-z._-]+", "_", artifact_namespace).strip("._")
    prefix = f"{safe_namespace}__" if safe_namespace else ""
    return EVALUATION_DIR / f"{prefix}{filename}"


def _console_print(message: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_message = str(message).encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_message)


def _profile_summary(context: AdaptiveContext) -> str:
    profile = context.profile
    return (
        f"docs={profile.document_count}, categories={profile.category_count}, "
        f"lang={profile.dominant_language}, avg_doc_len={profile.avg_document_length:.1f}, "
        f"exact_match={profile.exact_match_ratio:.3f}, semantic_need={profile.semantic_need_score:.3f}, "
        f"stt_quality={profile.stt_quality_score:.3f}"
    )


def _detail_columns() -> list[str]:
    return [
        "system_name",
        "system_type",
        "parameter_mode",
        "text_source",
        "score_kind",
        "query_id",
        "query",
        "query_preview",
        "ground_truth_rule",
        "evaluation_level",
        "ground_truth_ids",
        "ground_truth_file_names",
        "ground_truth_line_numbers",
        "ground_truth_segment_indexes",
        "ground_truth_segment_texts",
        "relevant_count",
        "tp_at_k",
        "fp_at_k",
        "fn_at_k",
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
        "mean_soft_accuracy_at_k",
        "topk_hit_rate",
        "top1_id",
        "top1_file_name",
        "top1_category",
        "top1_source_type",
        "top1_raw_score",
        "top1_display_score",
        "top1_final_score",
        "top1_matched_tokens",
        "top1_title_match_count",
        "top1_description_match_count",
        "top1_tags_match_count",
        "top1_transcript_match_count",
        "top1_lexical_score",
        "top1_semantic_score",
        "top1_reason",
        "mean_topk_raw_score",
        "mean_topk_final_score",
        "mean_topk_display_score",
        "topk_ids",
        "topk_file_names",
        "topk_categories",
        "topk_raw_scores",
        "topk_final_scores",
        "topk_display_scores",
        "topk_soft_accuracy_scores",
        "top1_file_match",
        "top1_segment_match",
        "top1_segment_reason",
        "topk_segment_exact_hit",
        "topk_segment_partial_hit",
        "hallucination",
        "hallucination_flag",
        "hallucination_reason",
        "hallucination_threshold_used",
        "soft_score_reason",
        "raw_score_explanation",
        "adaptive_field_weights",
        "adaptive_keyword_alpha",
        "adaptive_dense_alpha",
        "adaptive_reason",
        "adaptive_search_reason",
        "adaptive_metric_reason",
        "dataset_profile_summary",
    ]


def _summary_columns(top_k: int) -> list[str]:
    return [
        "system_name",
        "system_type",
        "parameter_mode",
        "text_source",
        "score_kind",
        "evaluation_definition",
        "query_count",
        "top_k",
        "similarity_score",
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
        "mean_soft_accuracy_at_k",
        "micro_precision_at_k",
        "micro_recall_at_k",
        "micro_f1_at_k",
        "hallucination_rate",
        "topk_hit_rate",
        "top1_accuracy",
        f"top{top_k}_accuracy",
        "segment_exact_hit_rate",
        "segment_partial_hit_rate",
        "mean_hallucination_threshold",
        "dataset_profile_summary",
        "search_parameter_reasoning",
        "metric_parameter_reasoning",
        "score_definition",
    ]


def _empty_eval_outputs(artifact_namespace: str | None = None, top_k: int = 3) -> dict[str, pd.DataFrame]:
    detail = pd.DataFrame(columns=_detail_columns())
    summary = pd.DataFrame(columns=_summary_columns(top_k))
    comparison = pd.DataFrame()
    ground_truth = pd.DataFrame()
    mode_comparison = pd.DataFrame()

    save_dataframe(evaluation_artifact_path("retrieval_eval_detail.csv", artifact_namespace), detail)
    save_dataframe(evaluation_artifact_path("retrieval_eval_summary.csv", artifact_namespace), summary)
    save_dataframe(evaluation_artifact_path("retrieval_eval_source_comparison.csv", artifact_namespace), comparison)
    save_dataframe(evaluation_artifact_path("retrieval_eval_mode_comparison.csv", artifact_namespace), mode_comparison)
    save_dataframe(evaluation_artifact_path("ground_truth_mapping.csv", artifact_namespace), ground_truth)
    save_json(evaluation_artifact_path("retrieval_eval_summary.json", artifact_namespace), [])
    save_json(evaluation_artifact_path("ground_truth_mapping.json", artifact_namespace), [])
    return {
        "detail": detail,
        "summary": summary,
        "comparison": comparison,
        "mode_comparison": mode_comparison,
        "ground_truth": ground_truth,
    }


def _build_source_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base = summary[
        [
            "system_name",
            "system_type",
            "parameter_mode",
            "text_source",
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
            "micro_precision_at_k",
            "micro_recall_at_k",
            "micro_f1_at_k",
            "hallucination_rate",
        ]
    ].copy()
    pivot = base.pivot_table(
        index=["system_name", "system_type", "parameter_mode"],
        columns="text_source",
        values=[
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
            "micro_precision_at_k",
            "micro_recall_at_k",
            "micro_f1_at_k",
            "hallucination_rate",
        ],
    )
    pivot.columns = ["__".join(column).strip() for column in pivot.columns.to_flat_index()]
    return pivot.reset_index()


def _summary_row(
    detail: pd.DataFrame,
    *,
    system_name: str,
    system_type: str,
    parameter_mode: str,
    text_source: str,
    score_kind: str,
    definition_text: str,
    top_k: int,
    context: AdaptiveContext,
) -> dict:
    summary = aggregate_metric_rows(detail, top_k=top_k)
    score_definition = detail["raw_score_explanation"].dropna().astype(str).iloc[0] if not detail.empty else ""
    metric_reason = detail["adaptive_metric_reason"].dropna().astype(str).iloc[0] if not detail.empty else ""
    return {
        "system_name": system_name,
        "system_type": system_type,
        "parameter_mode": parameter_mode,
        "text_source": text_source,
        "score_kind": score_kind,
        "evaluation_definition": definition_text,
        "query_count": int(len(detail)),
        "top_k": top_k,
        "similarity_score": summary["mean_top1_display_score"],
        "precision_at_k": summary["macro_precision_at_k"],
        "recall_at_k": summary["macro_recall_at_k"],
        "accuracy_at_1": summary["macro_accuracy_at_1"],
        "f1_at_k": summary["macro_f1_at_k"],
        "mrr_at_k": summary["macro_mrr_at_k"],
        "ndcg_at_k": summary["macro_ndcg_at_k"],
        "soft_precision_at_k": summary["macro_soft_precision_at_k"],
        "soft_recall_at_k": summary["macro_soft_recall_at_k"],
        "soft_accuracy_at_1": summary["macro_soft_accuracy_at_1"],
        "soft_f1_at_k": summary["macro_soft_f1_at_k"],
        "mean_soft_accuracy_at_k": summary["macro_mean_soft_accuracy_at_k"],
        "micro_precision_at_k": summary["micro_precision_at_k"],
        "micro_recall_at_k": summary["micro_recall_at_k"],
        "micro_f1_at_k": summary["micro_f1_at_k"],
        "hallucination_rate": summary["hallucination_rate"],
        "topk_hit_rate": summary["topk_hit_rate"],
        "top1_accuracy": summary["macro_accuracy_at_1"],
        f"top{top_k}_accuracy": summary[f"top{top_k}_accuracy"],
        "segment_exact_hit_rate": summary["segment_exact_hit_rate"],
        "segment_partial_hit_rate": summary["segment_partial_hit_rate"],
        "mean_hallucination_threshold": summary["mean_hallucination_threshold"],
        "dataset_profile_summary": _profile_summary(context),
        "search_parameter_reasoning": context.search.reasoning,
        "metric_parameter_reasoning": metric_reason or context.metric.reasoning,
        "score_definition": score_definition,
    }


def _evaluate_system(
    *,
    system_name: str,
    system_type: str,
    parameter_mode: str,
    score_kind: str,
    queryset: pd.DataFrame,
    search_fn: Callable[[str, int], pd.DataFrame],
    top_k: int,
    hallucination_threshold: float | None,
    text_source: str,
    context: AdaptiveContext,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    definition_text = evaluation_definition_text(queryset)

    for _, query_row in queryset.iterrows():
        results = search_fn(str(query_row["query"]), top_k)
        row_metric_config = resolve_metric_config(
            context.profile,
            score_values=results["display_score"].astype(float).tolist() if "display_score" in results.columns else None,
        )
        metrics = evaluate_ranked_results(
            results,
            query_row,
            top_k=top_k,
            metric_config=row_metric_config,
        )
        hallucination = detect_retrieval_hallucination(
            results,
            query_row,
            top_k=top_k,
            threshold=hallucination_threshold,
            metric_config=row_metric_config,
        )
        rows.append(
            {
                "system_name": system_name,
                "system_type": system_type,
                "parameter_mode": parameter_mode,
                "text_source": text_source,
                "score_kind": score_kind,
                "query_id": query_row["query_id"],
                "query": query_row["query"],
                "query_preview": query_row.get("query_preview", ""),
                **metrics,
                **hallucination,
                "adaptive_search_reason": context.search.reasoning,
                "adaptive_metric_reason": row_metric_config.reasoning,
                "dataset_profile_summary": _profile_summary(context),
            }
        )

    detail = pd.DataFrame(rows, columns=_detail_columns())
    summary = _summary_row(
        detail,
        system_name=system_name,
        system_type=system_type,
        parameter_mode=parameter_mode,
        text_source=text_source,
        score_kind=score_kind,
        definition_text=definition_text,
        top_k=top_k,
        context=context,
    )
    return detail, summary


def evaluate_keyword_engine(
    queryset: pd.DataFrame,
    metadata: pd.DataFrame,
    method: str,
    text_source: str,
    top_k: int,
    hallucination_threshold: float | None,
    adaptive_context: AdaptiveContext,
    parameter_mode: str,
) -> tuple[pd.DataFrame, dict]:
    engine = KeywordSearchEngine(
        metadata,
        text_source=text_source,
        adaptive_context=adaptive_context,
    )
    score_kind = "tfidf_dot" if method.lower() == "tfidf" else "bm25"
    return _evaluate_system(
        system_name=f"keyword-{method}",
        system_type="keyword",
        parameter_mode=parameter_mode,
        score_kind=score_kind,
        queryset=queryset,
        search_fn=lambda query, resolved_top_k: engine.search(query, top_k=resolved_top_k, method=method),
        top_k=top_k,
        hallucination_threshold=hallucination_threshold,
        text_source=text_source,
        context=adaptive_context,
    )


def evaluate_dense_engine(
    queryset: pd.DataFrame,
    metadata: pd.DataFrame,
    model_alias: str,
    text_source: str,
    top_k: int,
    hallucination_threshold: float | None,
    adaptive_context: AdaptiveContext,
    parameter_mode: str,
    artifact_namespace: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    engine = DenseSearchEngine(
        metadata,
        model_alias,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
        adaptive_context=adaptive_context,
    )
    engine.load()
    return _evaluate_system(
        system_name=model_alias,
        system_type="dense",
        parameter_mode=parameter_mode,
        score_kind="cosine_similarity",
        queryset=queryset,
        search_fn=lambda query, resolved_top_k: engine.search(query, top_k=resolved_top_k),
        top_k=top_k,
        hallucination_threshold=hallucination_threshold,
        text_source=text_source,
        context=adaptive_context,
    )


def evaluate_fused_engine(
    queryset: pd.DataFrame,
    metadata: pd.DataFrame,
    text_source: str,
    top_k: int,
    hallucination_threshold: float | None,
    adaptive_context: AdaptiveContext,
    parameter_mode: str,
    artifact_namespace: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    engine = FusedSearchEngine(
        metadata,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
        adaptive_context=adaptive_context,
    )
    return _evaluate_system(
        system_name="fused-retrieval",
        system_type="fused",
        parameter_mode=parameter_mode,
        score_kind="fused_weighted_score",
        queryset=queryset,
        search_fn=lambda query, resolved_top_k: engine.search(query, top_k=resolved_top_k, keyword_method="bm25"),
        top_k=top_k,
        hallucination_threshold=hallucination_threshold,
        text_source=text_source,
        context=adaptive_context,
    )


def evaluate_all(
    metadata_path: Path = DEFAULT_METADATA_CSV,
    queryset_path: Path = DEFAULT_QUERYSET_CSV,
    text_sources: tuple[str, ...] = (DEFAULT_TEXT_SOURCE, "original_transcript"),
    include_optional: bool = False,
    artifact_namespace: str | None = None,
    metadata: pd.DataFrame | None = None,
    queryset: pd.DataFrame | None = None,
    top_k: int | None = None,
    hallucination_threshold: float | None = None,
    print_report: bool = False,
    show_weights: bool = False,
    include_static_reference: bool = True,
) -> dict[str, pd.DataFrame]:
    ensure_project_dirs()
    metadata = metadata.copy() if metadata is not None else load_metadata_frame(metadata_path)
    raw_queryset = queryset.copy() if queryset is not None else pd.read_csv(queryset_path) if queryset_path.exists() else pd.DataFrame()

    fallback_top_k = top_k if top_k is not None else 3
    if metadata.empty:
        return _empty_eval_outputs(artifact_namespace, top_k=fallback_top_k)

    if raw_queryset.empty:
        raw_queryset = build_incremental_probe_queryset(metadata, set(metadata["id"].fillna("").astype(str)))

    ground_truth = normalize_ground_truth_queryset(raw_queryset, metadata)
    ground_truth = ground_truth.loc[ground_truth["relevant_count"].fillna(0).astype(int) > 0].reset_index(drop=True)
    if ground_truth.empty:
        raw_queryset = build_incremental_probe_queryset(metadata, set(metadata["id"].fillna("").astype(str)))
        ground_truth = normalize_ground_truth_queryset(raw_queryset, metadata).reset_index(drop=True)
    if ground_truth.empty:
        return _empty_eval_outputs(artifact_namespace, top_k=fallback_top_k)

    save_dataframe(evaluation_artifact_path("ground_truth_mapping.csv", artifact_namespace), ground_truth)
    save_json(
        evaluation_artifact_path("ground_truth_mapping.json", artifact_namespace),
        ground_truth.to_dict(orient="records"),
    )

    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    model_aliases = list(list_available_models(include_optional=include_optional).keys())

    for text_source in text_sources:
        adaptive_context = build_adaptive_context(
            metadata,
            ground_truth,
            text_source=text_source,
            artifact_namespace=artifact_namespace,
        )
        adaptive_top_k = resolve_top_k(adaptive_context.profile, top_k)

        keyword_bm25_detail, keyword_bm25_summary = evaluate_keyword_engine(
            queryset=ground_truth,
            metadata=metadata,
            method="bm25",
            text_source=text_source,
            top_k=adaptive_top_k,
            hallucination_threshold=hallucination_threshold,
            adaptive_context=adaptive_context,
            parameter_mode="adaptive",
        )
        detail_frames.append(keyword_bm25_detail)
        summary_rows.append(keyword_bm25_summary)

        keyword_tfidf_detail, keyword_tfidf_summary = evaluate_keyword_engine(
            queryset=ground_truth,
            metadata=metadata,
            method="tfidf",
            text_source=text_source,
            top_k=adaptive_top_k,
            hallucination_threshold=hallucination_threshold,
            adaptive_context=adaptive_context,
            parameter_mode="adaptive",
        )
        detail_frames.append(keyword_tfidf_detail)
        summary_rows.append(keyword_tfidf_summary)

        for model_alias in model_aliases:
            dense_context = build_adaptive_context(
                metadata,
                ground_truth,
                text_source=text_source,
                embedding_model_alias=model_alias,
                artifact_namespace=artifact_namespace,
            )
            dense_detail, dense_summary = evaluate_dense_engine(
                queryset=ground_truth,
                metadata=metadata,
                model_alias=model_alias,
                text_source=text_source,
                top_k=adaptive_top_k,
                hallucination_threshold=hallucination_threshold,
                adaptive_context=dense_context,
                parameter_mode="adaptive",
                artifact_namespace=artifact_namespace,
            )
            detail_frames.append(dense_detail)
            summary_rows.append(dense_summary)

        fused_detail, fused_summary = evaluate_fused_engine(
            queryset=ground_truth,
            metadata=metadata,
            text_source=text_source,
            top_k=adaptive_top_k,
            hallucination_threshold=hallucination_threshold,
            adaptive_context=adaptive_context,
            parameter_mode="adaptive",
            artifact_namespace=artifact_namespace,
        )
        detail_frames.append(fused_detail)
        summary_rows.append(fused_summary)

        if include_static_reference:
            static_context = build_static_reference_context(
                metadata,
                ground_truth,
                text_source=text_source,
                artifact_namespace=artifact_namespace,
            )
            static_top_k = top_k if top_k is not None else adaptive_top_k

            keyword_bm25_detail, keyword_bm25_summary = evaluate_keyword_engine(
                queryset=ground_truth,
                metadata=metadata,
                method="bm25",
                text_source=text_source,
                top_k=static_top_k,
                hallucination_threshold=hallucination_threshold,
                adaptive_context=static_context,
                parameter_mode="static_reference",
            )
            detail_frames.append(keyword_bm25_detail)
            summary_rows.append(keyword_bm25_summary)

            keyword_tfidf_detail, keyword_tfidf_summary = evaluate_keyword_engine(
                queryset=ground_truth,
                metadata=metadata,
                method="tfidf",
                text_source=text_source,
                top_k=static_top_k,
                hallucination_threshold=hallucination_threshold,
                adaptive_context=static_context,
                parameter_mode="static_reference",
            )
            detail_frames.append(keyword_tfidf_detail)
            summary_rows.append(keyword_tfidf_summary)

            for model_alias in model_aliases:
                dense_context = build_static_reference_context(
                    metadata,
                    ground_truth,
                    text_source=text_source,
                    embedding_model_alias=model_alias,
                    artifact_namespace=artifact_namespace,
                )
                dense_detail, dense_summary = evaluate_dense_engine(
                    queryset=ground_truth,
                    metadata=metadata,
                    model_alias=model_alias,
                    text_source=text_source,
                    top_k=static_top_k,
                    hallucination_threshold=hallucination_threshold,
                    adaptive_context=dense_context,
                    parameter_mode="static_reference",
                    artifact_namespace=artifact_namespace,
                )
                detail_frames.append(dense_detail)
                summary_rows.append(dense_summary)

            fused_detail, fused_summary = evaluate_fused_engine(
                queryset=ground_truth,
                metadata=metadata,
                text_source=text_source,
                top_k=static_top_k,
                hallucination_threshold=hallucination_threshold,
                adaptive_context=static_context,
                parameter_mode="static_reference",
                artifact_namespace=artifact_namespace,
            )
            detail_frames.append(fused_detail)
            summary_rows.append(fused_summary)

    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(columns=_detail_columns())
    summary_top_k = int(max(summary_rows[0]["top_k"], 1)) if summary_rows else fallback_top_k
    summary = pd.DataFrame(summary_rows, columns=_summary_columns(summary_top_k))
    comparison = _build_source_comparison(summary)

    adaptive_summary = summary.loc[summary["parameter_mode"] == "adaptive"].reset_index(drop=True)
    static_summary = summary.loc[summary["parameter_mode"] == "static_reference"].reset_index(drop=True)
    mode_comparison = compare_summary_frames(static_summary, adaptive_summary)

    save_dataframe(evaluation_artifact_path("retrieval_eval_detail.csv", artifact_namespace), detail)
    save_dataframe(evaluation_artifact_path("retrieval_eval_summary.csv", artifact_namespace), summary)
    save_dataframe(evaluation_artifact_path("retrieval_eval_source_comparison.csv", artifact_namespace), comparison)
    save_dataframe(evaluation_artifact_path("retrieval_eval_mode_comparison.csv", artifact_namespace), mode_comparison)
    save_json(
        evaluation_artifact_path("retrieval_eval_summary.json", artifact_namespace),
        summary.to_dict(orient="records"),
    )

    if show_weights:
        for line in format_model_weight_lines(include_optional=include_optional):
            _console_print(line)

    if print_report:
        _console_print(f"[EVAL] definition: {evaluation_definition_text(raw_queryset)}")
        _console_print("[EVAL] accuracy@1 is 1 when the top result is relevant, otherwise 0.")
        report_top_k = int(summary["top_k"].iloc[0]) if not summary.empty else fallback_top_k
        for system_name, system_frame in detail.groupby(["system_name", "text_source", "parameter_mode"], sort=False):
            _console_print(f"[EVAL] --- {system_name[0]} / {system_name[1]} / {system_name[2]} ---")
            for _, row in system_frame.iterrows():
                for line in format_query_console_report(row, top_k=report_top_k):
                    _console_print(line)
        for line in format_summary_console_report(summary, top_k=report_top_k):
            _console_print(line)

    return {
        "detail": detail,
        "summary": summary,
        "comparison": comparison,
        "mode_comparison": mode_comparison,
        "ground_truth": ground_truth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval systems on query ground truth.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--queryset-path", type=Path, default=DEFAULT_QUERYSET_CSV)
    parser.add_argument("--artifact-namespace", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--show-metrics", action="store_true")
    parser.add_argument("--show-weights", action="store_true")
    parser.add_argument("--hallucination-threshold", type=float, default=None)
    parser.add_argument("--no-static-reference", dest="include_static_reference", action="store_false")
    parser.set_defaults(include_static_reference=True)
    parser.add_argument(
        "--text-sources",
        nargs="+",
        default=[DEFAULT_TEXT_SOURCE, "original_transcript"],
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    outputs = evaluate_all(
        metadata_path=args.metadata_path,
        queryset_path=args.queryset_path,
        text_sources=tuple(args.text_sources),
        include_optional=args.include_optional,
        artifact_namespace=args.artifact_namespace,
        top_k=args.top_k,
        hallucination_threshold=args.hallucination_threshold,
        print_report=args.show_metrics,
        show_weights=args.show_weights,
        include_static_reference=args.include_static_reference,
    )
    _console_print(outputs["summary"].to_string(index=False))
    if args.show_weights:
        _console_print(build_model_weight_frame(include_optional=args.include_optional).to_string(index=False))


__all__ = [
    "compare_summary_frames",
    "evaluate_all",
    "evaluation_artifact_path",
]


if __name__ == "__main__":
    main()
