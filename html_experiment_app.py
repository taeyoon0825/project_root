from __future__ import annotations

import argparse
import cgi
import hashlib
import json
import math
import mimetypes
import os
import re
import sys
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import plotly
import plotly.io as pio

from src.adaptive.parameter_resolver import (
    AdaptiveContext,
    build_adaptive_context,
    resolve_metric_config,
    resolve_top_k,
)
from src.config import (
    CLUSTERS_DIR,
    DEFAULT_QUERYSET_CSV,
    EMBEDDINGS_DIR,
    HTML_UPLOAD_MEDIA_DIR,
    HTML_UPLOAD_TRANSCRIPTS_DIR,
    HTML_UPLOAD_WAV_DIR,
    INCREMENTAL_RUN_SUMMARY_JSON,
    REALDATA_METADATA_CSV,
    ensure_project_dirs,
)
from src.audio.audio_utils import convert_media_to_wav
from src.data.metadata_schema import ensure_metadata_columns, load_metadata_frame, save_metadata_frame
from src.embedding.build_indices import DenseSearchEngine, artifact_stem
from src.embedding.vector_models import list_available_models
from src.evaluation.evaluate import evaluate_all, evaluation_artifact_path
from src.evaluation.ground_truth import (
    build_incremental_probe_queryset,
    build_metadata_token_query_row,
    normalize_ground_truth_queryset,
    resolve_relevant_ids,
)
from src.evaluation.hallucination import DEFAULT_HALLUCINATION_THRESHOLD
from src.evaluation.reporting import build_model_weight_frame
from src.evaluation.soft_metrics import f1_score
from src.ingest.incremental_registry import load_incremental_run_summary
from src.search.keyword_search import KeywordSearchEngine
from src.search.query_preview import extract_dense_preview, extract_keyword_preview
from src.retrieval.fused_search import FusedSearchEngine
from src.search.load_realdata_dataset import dataset_artifact_namespace, load_search_metadata
from src.search.text_source import DEFAULT_TEXT_SOURCE, resolve_primary_text, split_text_into_lines
from src.stt.batch_transcribe import transcribe_audio_batch
from src.utils.io_utils import load_json, save_json
from src.utils.device import device_payload
from src.visualize.clustering import (
    cluster_embeddings,
    load_cluster_frame,
    load_cluster_summary,
    load_representatives,
)
from src.visualize.interactive_plot import (
    build_projection_figure,
    load_projection_frame,
    project_query_vector,
    projection_artifact_path,
)
from src.visualize.pca_plot import build_projection_artifacts

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


ROOT_DIR = Path(__file__).resolve().parent
HTML_PATH = ROOT_DIR / "web" / "experiment_dashboard.html"
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
PLOT_MODE_OPTIONS = {
    "PCA 3D": ("pca", 3),
    "PCA 2D": ("pca", 2),
    "UMAP 3D": ("umap", 3),
    "UMAP 2D": ("umap", 2),
    "t-SNE 3D": ("tsne", 3),
    "t-SNE 2D": ("tsne", 2),
}
SEARCH_TABLE_COLUMNS = [
    "rank",
    "file_name",
    "relative_match_score",
    "normalized_relevance_score",
    "ranking_confidence_score",
    "matched_tokens",
    "title_match_count",
    "description_match_count",
    "tags_match_count",
    "transcript_match_count",
    "lexical_score",
    "semantic_score",
    "final_score",
    "reason",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "hallucination_flag",
    "search_location",
    "query_preview",
]
SUPPORTED_UPLOAD_EXTENSIONS = {".mp4", ".wav"}
UPLOAD_SOURCE_TYPE = "local_media"


class SearchPreparationError(ValueError):
    pass


class ModelLoadError(RuntimeError):
    pass


def _manual_query_row(query_text: str) -> pd.Series:
    normalized = str(query_text or "").strip()
    return pd.Series(
        {
            "query_id": "manual_query",
            "query": normalized,
            "query_preview": truncate_text(normalized, 80),
            "target_category": "",
            "target_source_type": "",
            "relevant_ids": "",
            "relevant_file_names": "",
            "relevant_line_numbers": "",
            "relevant_segment_indexes": "",
            "relevant_segment_texts": "",
            "relevant_count": 0,
            "evaluation_level": "file",
            "ground_truth_rule": "manual_query_no_ground_truth",
        }
    )


def _is_auto_probe_query(query_row: pd.Series) -> bool:
    query_id = str(query_row.get("query_id", "")).strip().lower()
    rule = str(query_row.get("ground_truth_rule", "")).strip().lower()
    if query_id.startswith("probe_"):
        return True
    auto_tokens = [
        "derived_from_metadata_token_match",
        "derived_from_exact_phrase_match",
        "derived_from_all_query_tokens",
        "manual_query_no_ground_truth",
    ]
    return any(token in rule for token in auto_tokens)


def _evaluation_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    return catalog.loc[~catalog.apply(_is_auto_probe_query, axis=1)].reset_index(drop=True)


def truncate_text(text: Any, length: int = 96) -> str:
    value = str(text or "").strip()
    if len(value) <= length:
        return value
    return value[: length - 3] + "..."


def format_seconds(value: Any) -> str:
    try:
        total = max(0, int(float(value)))
    except (TypeError, ValueError):
        return "-"
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _frame_records(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    selected = frame.copy()
    if columns is not None:
        selected = selected[[column for column in columns if column in selected.columns]]
    if limit is not None:
        selected = selected.head(limit)
    return [_json_safe(row) for row in selected.to_dict(orient="records")]


def _table_payload(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    selected_columns = columns or frame.columns.tolist()
    selected_columns = [column for column in selected_columns if column in frame.columns]
    return {
        "columns": selected_columns,
        "rows": _frame_records(frame, selected_columns, limit=limit),
    }


def _parse_requested_top_k(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _adaptive_payload(context: AdaptiveContext | None) -> dict[str, Any]:
    if context is None:
        return {}
    return {
        "profile": _json_safe(context.profile.to_dict()),
        "search": _json_safe(context.search.to_dict()),
        "language": _json_safe(context.language.to_dict()),
        "metric": _json_safe(context.metric.to_dict()),
        "cluster": _json_safe(context.cluster.to_dict()),
        "visualization": _json_safe(context.visualization.to_dict()),
    }


def _load_base_queryset() -> pd.DataFrame:
    if DEFAULT_QUERYSET_CSV.exists():
        return pd.read_csv(DEFAULT_QUERYSET_CSV)
    return pd.DataFrame(columns=["query_id", "query", "target_category", "relevant_id", "target_source_type"])


def _catalog_with_relevant_docs(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    relevant_counts = pd.to_numeric(catalog["relevant_count"], errors="coerce").fillna(0)
    relevant_ids = catalog["relevant_ids"].fillna("").astype(str).str.strip()
    return catalog.loc[relevant_counts.gt(0) & relevant_ids.ne("")].reset_index(drop=True)


def build_query_catalog(metadata: pd.DataFrame, artifact_namespace: str) -> tuple[pd.DataFrame, Path]:
    target_ids = set(metadata["id"].fillna("").astype(str))
    probe_queryset = build_incremental_probe_queryset(metadata, target_ids)
    probe_catalog = _catalog_with_relevant_docs(normalize_ground_truth_queryset(probe_queryset, metadata))

    base_catalog = _catalog_with_relevant_docs(normalize_ground_truth_queryset(_load_base_queryset(), metadata))
    catalog = pd.concat([probe_catalog, base_catalog], ignore_index=True)
    if not catalog.empty:
        catalog = catalog.drop_duplicates(subset=["query_id"], keep="first").reset_index(drop=True)
    if catalog.empty:
        catalog = normalize_ground_truth_queryset(probe_queryset, metadata).reset_index(drop=True)

    catalog_path = evaluation_artifact_path("query_catalog.json", artifact_namespace)
    save_json(catalog_path, catalog.to_dict(orient="records"))
    return catalog, catalog_path


def _normalize_for_match(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _segment_payload_from_row(row: pd.Series) -> dict[str, Any] | None:
    for candidate in [str(row.get("stt_txt_path", "")).strip(), str(row.get("processed_txt_path", "")).strip()]:
        if not candidate:
            continue
        segment_path = Path(candidate).with_suffix(".segments.json")
        if segment_path.exists():
            try:
                return load_json(segment_path)
            except Exception:
                return None
    return None


def resolve_location_display(row: pd.Series) -> str:
    best_match_text = _normalize_for_match(row.get("best_match_text", ""))
    payload = _segment_payload_from_row(row)
    if payload and payload.get("segments"):
        for index, segment in enumerate(payload["segments"], start=1):
            segment_text = _normalize_for_match(segment.get("text", ""))
            if best_match_text and (best_match_text in segment_text or segment_text in best_match_text):
                return (
                    f"{format_seconds(segment.get('start', 0))} ~ "
                    f"{format_seconds(segment.get('end', 0))} / segment {index}"
                )

    source_line = int(row.get("best_match_source_line_number", 0) or 0)
    line_number = int(row.get("best_match_line_number", 0) or 0)
    sentence_number = int(row.get("best_match_sentence_number", 0) or 0)
    parts: list[str] = []
    if source_line:
        parts.append(f"line {source_line}")
    elif line_number:
        parts.append(f"line {line_number}")
    if sentence_number:
        parts.append(f"sentence {sentence_number}")
    return " / ".join(parts) if parts else "-"


def annotate_search_results(
    results: pd.DataFrame,
    query_row: pd.Series,
    metadata: pd.DataFrame,
    top_k: int,
    *,
    preview_mode: str,
    keyword_method: str = "bm25",
    dense_wrapper: Any | None = None,
    hallucination_threshold: float | None = DEFAULT_HALLUCINATION_THRESHOLD,
) -> tuple[pd.DataFrame, set[str]]:
    relevant_ids = resolve_relevant_ids(query_row, metadata)
    has_ground_truth = bool(relevant_ids)
    query_text = str(query_row.get("query", "")).strip()
    metadata_lookup = (
        metadata.copy()
        .assign(id=metadata["id"].fillna("").astype(str))
        .drop_duplicates(subset=["id"], keep="first")
        .set_index("id")
    )
    hits = 0
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(results.head(top_k).itertuples(index=False), start=1):
        row_map = pd.Series(row._asdict())
        row_id = str(row_map.get("id", "")).strip()
        source_text = ""
        if row_id and row_id in metadata_lookup.index:
            source_text = resolve_primary_text(
                metadata_lookup.loc[row_id],
                text_source=str(row_map.get("search_source", DEFAULT_TEXT_SOURCE) or DEFAULT_TEXT_SOURCE),
            )
        if preview_mode == "keyword":
            query_preview = extract_keyword_preview(
                source_text,
                query_text,
                method=keyword_method,
                length=int(row_map.get("adaptive_preview_length", 0) or 0) or None,
                match_payload=row_map.to_dict(),
            )
        else:
            query_preview = extract_dense_preview(
                source_text,
                query_text,
                model=dense_wrapper,
                length=int(row_map.get("adaptive_preview_length", 0) or 0) or None,
                match_payload=row_map.to_dict(),
            )
        is_relevant = str(getattr(row, "id", "")) in relevant_ids
        hits += int(is_relevant)
        precision = (hits / rank) if has_ground_truth else 0.0
        recall = (hits / max(1, len(relevant_ids))) if has_ground_truth else 0.0
        accuracy = (1.0 if rank == 1 and is_relevant else 0.0) if has_ground_truth else 0.0
        f1 = f1_score(precision, recall) if has_ground_truth else 0.0
        similarity = float(getattr(row, "similarity_score", 0.0) or 0.0)
        threshold_used = hallucination_threshold if hallucination_threshold is not None else DEFAULT_HALLUCINATION_THRESHOLD
        hallucination = bool((not is_relevant) and similarity >= threshold_used)
        rows.append(
            {
                **row._asdict(),
                "rank": rank,
                "is_relevant": is_relevant,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "f1_score": f1,
                "hallucination_flag": "YES" if hallucination else "NO",
                "hallucination_threshold_used": threshold_used,
                "has_ground_truth": has_ground_truth,
                "search_location": resolve_location_display(row_map),
                "query_preview": query_preview,
            }
        )
    return pd.DataFrame(rows), relevant_ids


def build_query_summary(system_name: str, annotated: pd.DataFrame, relevant_ids: set[str], top_k: int) -> dict[str, Any]:
    hits = int(annotated["is_relevant"].sum()) if not annotated.empty else 0
    predicted_count = len(annotated)
    has_ground_truth = bool(relevant_ids)
    precision = (hits / max(1, predicted_count)) if has_ground_truth else 0.0
    recall = (hits / max(1, len(relevant_ids))) if has_ground_truth else 0.0
    accuracy = float(bool(not annotated.empty and bool(annotated.iloc[0]["is_relevant"]))) if has_ground_truth else 0.0
    f1 = f1_score(precision, recall) if has_ground_truth else 0.0
    hallucination_count = int((annotated["hallucination_flag"] == "YES").sum()) if not annotated.empty else 0
    reciprocal_rank = 0.0
    binary_relevance: list[int] = []
    for rank, is_relevant in enumerate(annotated.get("is_relevant", pd.Series(dtype=bool)).tolist(), start=1):
        relevance = int(bool(is_relevant))
        binary_relevance.append(relevance)
        if reciprocal_rank == 0.0 and relevance:
            reciprocal_rank = 1.0 / rank

    def _dcg(relevances: list[int]) -> float:
        total = 0.0
        for rank, relevance in enumerate(relevances, start=1):
            total += float(relevance) / math.log2(rank + 1.0)
        return total

    ideal_relevance = [1] * min(len(relevant_ids), len(binary_relevance))
    ndcg = 0.0
    if binary_relevance:
        ideal_dcg = _dcg(ideal_relevance)
        ndcg = 0.0 if ideal_dcg == 0 else _dcg(binary_relevance) / ideal_dcg
    topk_hit_rate = float(hits > 0) if has_ground_truth else 0.0

    return {
        "system_name": system_name,
        "relevant_count": len(relevant_ids),
        f"top_{top_k}_hit_count": hits,
        f"top_{top_k}_result_count": predicted_count,
        f"precision@{top_k}": precision,
        f"recall@{top_k}": recall,
        f"f1@{top_k}": f1,
        "accuracy@1_reference": accuracy,
        "metric_definition": (
            "precision/recall/f1 are calculated from retrieved ids and ground truth relevant ids; accuracy is only a top-1 reference."
            if has_ground_truth
            else "no explicit ground truth mapped for this query; quality metrics are not reliable."
        ),
        "has_ground_truth": has_ground_truth,
        "hallucination_count": hallucination_count,
        "hallucination_rate": hallucination_count / max(1, predicted_count),
        "mean_final_score": float(annotated["final_score"].mean()) if "final_score" in annotated.columns and not annotated.empty else 0.0,
        "mean_similarity_score": float(annotated["similarity_score"].mean()) if not annotated.empty else 0.0,
        "accuracy_at_1": accuracy,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": ndcg,
        "topk_hit_rate": topk_hit_rate,
    }


def build_search_table(annotated: pd.DataFrame) -> pd.DataFrame:
    display = annotated.copy()
    if "similarity_score" in display.columns:
        # query-internal relative ranking score (top row can be 100)
        display["relative_match_score"] = display["similarity_score"].astype(float)
    if "final_score" in display.columns:
        # absolute-ish fused/engine score for this query (not min-max top1 fixed 100)
        display["normalized_relevance_score"] = display["final_score"].astype(float) * 100.0
    if "reranker_score" in display.columns:
        display["ranking_confidence_score"] = display["reranker_score"].astype(float) * 100.0
    else:
        display["ranking_confidence_score"] = 0.0
    for column in [
        "relative_match_score",
        "normalized_relevance_score",
        "ranking_confidence_score",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "title_match_count",
        "description_match_count",
        "tags_match_count",
        "transcript_match_count",
        "field_weight_score",
        "ranker_score",
        "lexical_score",
        "semantic_score",
        "final_score",
    ]:
        if column in display.columns:
            display[column] = display[column].astype(float).round(4)
    available = [column for column in SEARCH_TABLE_COLUMNS if column in display.columns]
    return display[available]


def artifact_ids_match(csv_path: Path, metadata: pd.DataFrame) -> bool:
    if not csv_path.exists():
        return False
    try:
        artifact_ids = pd.read_csv(csv_path, usecols=["id"])["id"].fillna("").astype(str).tolist()
    except Exception:
        return False
    expected_ids = metadata["id"].fillna("").astype(str).tolist()
    return artifact_ids == expected_ids


def ensure_artifacts(
    metadata: pd.DataFrame,
    artifact_namespace: str,
    text_source: str,
    model_aliases: list[str],
    cluster_method: str,
    n_clusters: int | None = None,
    optional_methods: tuple[str, ...] = ("tsne",),
    adaptive_context: AdaptiveContext | None = None,
) -> None:
    keyword_context = adaptive_context or build_adaptive_context(
        metadata,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
    )
    KeywordSearchEngine(
        metadata,
        text_source=text_source,
        adaptive_context=keyword_context,
        artifact_namespace=artifact_namespace,
    ).export_index_metadata(artifact_namespace=artifact_namespace)
    for model_alias in model_aliases:
        dense_context = build_adaptive_context(
            metadata,
            text_source=text_source,
            embedding_model_alias=model_alias,
            artifact_namespace=artifact_namespace,
        )
        dense_engine = DenseSearchEngine(
            metadata,
            model_alias,
            text_source=text_source,
            artifact_namespace=artifact_namespace,
            adaptive_context=dense_context,
        )
        dense_engine.load()

        projection_paths = [
            projection_artifact_path(model_alias, text_source, "pca", 3, artifact_namespace=artifact_namespace),
        ]
        for method in optional_methods:
            if method in {"tsne", "umap"}:
                projection_paths.extend(
                    [
                        projection_artifact_path(model_alias, text_source, method, 2, artifact_namespace=artifact_namespace),
                        projection_artifact_path(model_alias, text_source, method, 3, artifact_namespace=artifact_namespace),
                    ]
                )
        projection_is_stale = any(not artifact_ids_match(path, metadata) for path in projection_paths)
        if projection_is_stale:
            build_projection_artifacts(
                metadata,
                model_alias,
                text_source=text_source,
                optional_methods=optional_methods,
                artifact_namespace=artifact_namespace,
                adaptive_context=dense_context,
            )

        cluster_csv = CLUSTERS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_{cluster_method}_clusters.csv"
        if not artifact_ids_match(cluster_csv, metadata):
            try:
                cluster_embeddings(
                    metadata,
                    model_alias,
                    method=cluster_method,
                    n_clusters=n_clusters,
                    text_source=text_source,
                    artifact_namespace=artifact_namespace,
                    adaptive_context=dense_context,
                )
            except Exception:
                if cluster_method != "kmeans":
                    pass


def _validate_fused_search_artifacts(
    metadata: pd.DataFrame,
    artifact_namespace: str,
    text_source: str,
    model_aliases: list[str],
) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "metadata_rows": int(len(metadata)),
        "searchable_primary_rows": int(
            metadata.apply(lambda row: bool(str(resolve_primary_text(row, text_source=text_source)).strip()), axis=1).sum()
        ),
    }
    metadata_ids = metadata["id"].fillna("").astype(str).tolist()
    if counts["metadata_rows"] == 0:
        raise SearchPreparationError("no metadata rows available for fused search")

    if counts["searchable_primary_rows"] == 0:
        raise SearchPreparationError("missing searchable transcript for uploaded files")

    for model_alias in model_aliases:
        stem = artifact_stem(model_alias, text_source, artifact_namespace)
        emb_path = EMBEDDINGS_DIR / f"{stem}_embeddings.npy"
        dense_meta_path = EMBEDDINGS_DIR / f"{stem}_metadata.csv"
        if not emb_path.exists():
            raise SearchPreparationError(f"missing embedding artifact: {emb_path}")
        if not dense_meta_path.exists():
            raise SearchPreparationError(f"missing embedding metadata artifact: {dense_meta_path}")
        emb_rows = int(np.load(emb_path).shape[0])
        dense_meta = load_metadata_frame(dense_meta_path)
        dense_ids = dense_meta["id"].fillna("").astype(str).tolist()
        counts[f"{model_alias}_embedding_rows"] = emb_rows
        counts[f"{model_alias}_dense_metadata_rows"] = int(len(dense_meta))
        if emb_rows != len(metadata):
            raise SearchPreparationError(
                f"embedding/metadata row mismatch after rebuild: model={model_alias}, "
                f"embedding rows={emb_rows}, metadata rows={len(metadata)}"
            )
        if len(dense_meta) != len(metadata):
            raise SearchPreparationError(
                f"dense metadata row mismatch after rebuild: model={model_alias}, "
                f"dense metadata rows={len(dense_meta)}, metadata rows={len(metadata)}"
            )
        if dense_ids != metadata_ids:
            raise SearchPreparationError(f"doc_id alignment failed during fused search preparation: model={model_alias}")
    return counts


def evaluation_text_sources(metadata: pd.DataFrame) -> tuple[str, ...]:
    sources: list[str] = []
    if metadata["stt_transcript"].fillna("").astype(str).str.strip().str.len().gt(0).any():
        sources.append("stt_transcript")
    if metadata["original_transcript"].fillna("").astype(str).str.strip().str.len().gt(0).any():
        sources.append("original_transcript")
    return tuple(sources or ["stt_transcript"])


def evaluation_content_token(
    metadata: pd.DataFrame,
    query_catalog: pd.DataFrame,
    text_sources: tuple[str, ...],
    top_k: int,
) -> str:
    import hashlib

    digest = hashlib.sha1()
    digest.update(b"html-evaluation-explainable-search-v3")
    digest.update(str(top_k).encode("utf-8"))
    digest.update("|".join(text_sources).encode("utf-8"))
    for row in metadata.fillna("").itertuples(index=False):
        for field in ["id", "title", "description", "tags", "keywords", "stt_transcript", "original_transcript"]:
            digest.update(str(getattr(row, field, "")).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        digest.update(b"\n")
    for row in query_catalog.fillna("").itertuples(index=False):
        for field in ["query_id", "query", "relevant_ids", "relevant_count"]:
            digest.update(str(getattr(row, field, "")).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_evaluation_artifacts(metadata: pd.DataFrame, query_catalog: pd.DataFrame, artifact_namespace: str, top_k: int) -> None:
    text_sources = evaluation_text_sources(metadata)
    current_token = evaluation_content_token(metadata, query_catalog, text_sources, top_k)
    manifest_path = evaluation_artifact_path("retrieval_eval_manifest.json", artifact_namespace)
    required_paths = [
        evaluation_artifact_path("retrieval_eval_summary.csv", artifact_namespace),
        evaluation_artifact_path("retrieval_eval_detail.csv", artifact_namespace),
        evaluation_artifact_path("retrieval_eval_source_comparison.csv", artifact_namespace),
        evaluation_artifact_path("retrieval_eval_mode_comparison.csv", artifact_namespace),
    ]
    if all(path.exists() for path in required_paths) and manifest_path.exists():
        try:
            manifest = load_json(manifest_path)
        except Exception:
            manifest = {}
        if manifest.get("token") == current_token:
            return

    outputs = evaluate_all(
        metadata=metadata,
        queryset=query_catalog,
        text_sources=text_sources,
        include_optional=False,
        artifact_namespace=artifact_namespace,
        top_k=top_k,
        print_report=False,
        show_weights=False,
    )
    save_json(
        manifest_path,
        {
            "token": current_token,
            "top_k": top_k,
            "text_sources": list(text_sources),
            "query_count": int(len(query_catalog)),
            "summary_rows": int(len(outputs["summary"])),
        },
    )


def relevant_file_summary(query_row: pd.Series, metadata: pd.DataFrame) -> str:
    relevant_ids = resolve_relevant_ids(query_row, metadata)
    if not relevant_ids:
        return "수동 질의 또는 정답 매핑 미지정 상태입니다. 검색 결과는 표시되지만 정답 기반 평가는 수행되지 않습니다."
    matched = metadata.loc[metadata["id"].fillna("").astype(str).isin(relevant_ids)]
    names = matched["file_name"].fillna("").astype(str).tolist()
    if names:
        return f"정답 문서 {len(relevant_ids)}개: {', '.join(names[:3])}" + (" ..." if len(names) > 3 else "")
    return f"정답 문서 {len(relevant_ids)}개: {', '.join(sorted(relevant_ids)[:3])}"


def _score_semantics_payload() -> dict[str, str]:
    return {
        "relative_match_score": "질의 내부 상대 점수(0~100). 같은 질의의 후보 간 상대값이며 top1이 100일 수 있습니다.",
        "normalized_relevance_score": "최종 rank에 쓰인 점수(final_score)의 직접 스케일(0~100). 상대점수와 별도로 해석하세요.",
        "ranking_confidence_score": "reranker confidence score(0~100). reranker 미적용 시 0으로 표시됩니다.",
        "is_relevant": "ground-truth 기준 판정값입니다. 수동 질의에서 정답 매핑이 없으면 false로 표시될 수 있습니다.",
    }


def build_plot_frame(
    model_alias: str,
    text_source: str,
    method: str,
    dimensions: int,
    score_frame: pd.DataFrame,
    cluster_method: str,
    artifact_namespace: str,
) -> pd.DataFrame:
    projection = load_projection_frame(model_alias, text_source, method, dimensions, artifact_namespace=artifact_namespace)
    cluster_frame = load_cluster_frame(model_alias, text_source, cluster_method, artifact_namespace)[["id", "cluster_id"]]
    merged = projection.merge(cluster_frame, on="id", how="left")
    merged = merged.merge(score_frame[["id", "raw_score", "normalized_score"]], on="id", how="left")
    merged["preview"] = merged["stt_transcript"].where(
        merged["stt_transcript"].astype(str).str.len() > 0,
        merged["original_transcript"],
    )
    merged["preview"] = merged["preview"].astype(str).apply(lambda value: truncate_text(value, 160))
    return merged


def _load_filtered_metadata(source_types: list[str] | tuple[str, ...] | None) -> tuple[pd.DataFrame, Path]:
    _ = source_types
    return load_search_metadata(REALDATA_METADATA_CSV, None)


def _artifact_namespace(metadata_path: Path, source_types: list[str] | tuple[str, ...] | None) -> str:
    _ = source_types
    return dataset_artifact_namespace(metadata_path, None)


def _safe_upload_name(value: str) -> str:
    path_name = Path(str(value or "upload")).name
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", path_name).strip("._")
    return cleaned or "upload"


def _infer_upload_keywords(title: str, tags: str, transcript: str, limit: int = 12) -> str:
    candidates = []
    for token in re.findall(r"[0-9A-Za-z가-힣]+", " ".join([title, tags, transcript]).lower()):
        if len(token) < 2:
            continue
        candidates.append(token)
    counts: dict[str, int] = {}
    for token in candidates:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(token for token, _ in ranked[:limit])


def _update_upload_error(metadata_path: Path, doc_id: str, message: str) -> None:
    metadata = load_metadata_frame(metadata_path)
    matched = metadata["id"].astype(str) == doc_id
    if matched.any():
        metadata.loc[matched, "processing_status"] = "stt_error"
        metadata.loc[matched, "error_message"] = message
        save_metadata_frame(metadata, metadata_path)


def _append_upload_metadata_row(
    *,
    doc_id: str,
    source_path: Path,
    wav_path: Path,
    transcript_path: Path,
    title: str,
    description: str,
    tags: str,
) -> None:
    metadata = load_metadata_frame(REALDATA_METADATA_CSV) if REALDATA_METADATA_CSV.exists() else ensure_metadata_columns(pd.DataFrame())
    metadata = metadata.loc[metadata["id"].astype(str) != doc_id].copy()
    row = {
        "id": doc_id,
        "source_type": UPLOAD_SOURCE_TYPE,
        "category": UPLOAD_SOURCE_TYPE,
        "title": title or source_path.stem,
        "description": description,
        "file_name": source_path.name,
        "file_path": str(source_path.resolve()),
        "audio_path": str(wav_path.resolve()),
        "processed_txt_path": str(transcript_path.resolve()),
        "original_transcript": "",
        "stt_transcript": "",
        "tags": tags,
        "keywords": tags,
        "tts_text": "",
        "audio_file_name": wav_path.name,
        "audio_file_path": str(wav_path.resolve()),
        "stt_txt_path": str(transcript_path.resolve()),
        "tts_provider": "",
        "stt_model_name": "",
        "processing_status": "uploaded",
        "error_message": "",
        "input_kind": source_path.suffix.lower().lstrip("."),
        "source_mtime": "",
        "source_size": str(source_path.stat().st_size if source_path.exists() else ""),
        "source_hash": "",
        "last_ingested_at": "",
    }
    metadata = pd.concat([metadata, ensure_metadata_columns(pd.DataFrame([row]))], ignore_index=True)
    save_metadata_frame(metadata, REALDATA_METADATA_CSV)


def _finalize_upload_metadata(doc_id: str, fallback_title: str, tags: str) -> pd.Series:
    metadata = load_metadata_frame(REALDATA_METADATA_CSV)
    matched = metadata["id"].astype(str) == doc_id
    if not matched.any():
        raise RuntimeError(f"Uploaded metadata row was not found: {doc_id}")
    row = metadata.loc[matched].iloc[0].copy()
    transcript = str(row.get("stt_transcript", "")).strip()
    inferred_keywords = _infer_upload_keywords(str(row.get("title", fallback_title)), tags, transcript)
    keywords = tags.strip() or inferred_keywords
    metadata.loc[matched, "keywords"] = keywords
    metadata.loc[matched, "tags"] = keywords
    metadata.loc[matched, "processed_txt_path"] = str(row.get("stt_txt_path", ""))
    metadata.loc[matched, "processing_status"] = "transcribed" if transcript else str(row.get("processing_status", ""))
    save_metadata_frame(metadata, REALDATA_METADATA_CSV)
    return metadata.loc[matched].iloc[0]


def handle_upload(form: cgi.FieldStorage) -> dict[str, Any]:
    if "file" not in form:
        return {"ok": False, "error": "업로드할 MP4 또는 WAV 파일을 선택하세요."}

    file_item = form["file"]
    if isinstance(file_item, list):
        file_item = file_item[0]
    original_name = _safe_upload_name(file_item.filename or "upload")
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        return {"ok": False, "error": "MP4 또는 WAV 파일만 업로드할 수 있습니다."}

    ensure_project_dirs()
    HTML_UPLOAD_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    HTML_UPLOAD_WAV_DIR.mkdir(parents=True, exist_ok=True)
    HTML_UPLOAD_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    file_bytes = file_item.file.read()
    if not file_bytes:
        return {"ok": False, "error": "업로드된 파일이 비어 있습니다."}

    digest = hashlib.sha1(file_bytes).hexdigest()[:12].upper()
    doc_id = f"HTMLUP-{digest}-{uuid.uuid4().hex[:8].upper()}"
    source_path = HTML_UPLOAD_MEDIA_DIR / f"{doc_id}_{original_name}"
    wav_path = HTML_UPLOAD_WAV_DIR / f"{doc_id}.wav"
    transcript_path = HTML_UPLOAD_TRANSCRIPTS_DIR / f"{doc_id}.txt"
    source_path.write_bytes(file_bytes)

    title = str(form.getfirst("title", "") or "").strip() or Path(original_name).stem
    description = str(form.getfirst("description", "") or "").strip()
    tags = str(form.getfirst("tags", "") or "").strip()
    whisper_model = str(form.getfirst("whisperModel", "") or "").strip() or None
    language = str(form.getfirst("language", "") or "").strip() or None

    try:
        convert_media_to_wav(source_path, wav_path, overwrite=True)
        _append_upload_metadata_row(
            doc_id=doc_id,
            source_path=source_path,
            wav_path=wav_path,
            transcript_path=transcript_path,
            title=title,
            description=description,
            tags=tags,
        )
        transcribe_audio_batch(
            metadata_path=REALDATA_METADATA_CSV,
            model_name=whisper_model,
            language=language,
            overwrite=True,
            target_ids={doc_id},
            skip_errors=False,
        )
        row = _finalize_upload_metadata(doc_id, title, tags)
    except Exception as exc:
        _update_upload_error(REALDATA_METADATA_CSV, doc_id, str(exc))
        return {
            "ok": False,
            "error": str(exc),
            "device": device_payload(),
            "docId": doc_id,
        }

    transcript = str(row.get("stt_transcript", "")).strip()
    search_query = " ".join(part for part in [str(row.get("title", "")), str(row.get("keywords", "")), " ".join(transcript.split()[:8])] if part).strip()
    return {
        "ok": True,
        "message": "업로드와 STT 변환이 완료되었습니다.",
        "docId": doc_id,
        "sourceType": UPLOAD_SOURCE_TYPE,
        "sourcePath": str(source_path),
        "wavPath": str(wav_path),
        "transcriptPath": str(transcript_path),
        "sttCsvPath": str(row.get("stt_csv_path", "")),
        "title": str(row.get("title", "")),
        "searchQuery": search_query,
        "transcriptPreview": transcript[:240],
        "metadataPath": str(REALDATA_METADATA_CSV),
        "uploadRoot": str(HTML_UPLOAD_MEDIA_DIR.parent),
        "device": device_payload(),
    }


def _selected_query_row(
    catalog: pd.DataFrame,
    query_id: str | None,
    query: str | None,
    metadata: pd.DataFrame,
    text_source: str,
) -> pd.Series:
    query_text = str(query or "").strip()
    if catalog.empty:
        row = _manual_query_row(query_text or "유튜브 전사 검색")
    else:
        exact_query = (
            catalog.loc[catalog["query"].fillna("").astype(str).str.strip() == query_text]
            if query_text
            else pd.DataFrame()
        )
        matched = catalog.loc[catalog["query_id"].astype(str) == str(query_id)]
        if not exact_query.empty:
            row = exact_query.iloc[0].copy()
        elif not matched.empty and (not query_text or query_text == str(matched.iloc[0].get("query", "")).strip()):
            row = matched.iloc[0].copy()
        elif query_text:
            row = _manual_query_row(query_text)
        else:
            row = matched.iloc[0].copy() if not matched.empty else catalog.iloc[0].copy()
    row["query_preview"] = truncate_text(row.get("query", ""), 80)
    return row


def _load_eval_frame(filename: str, artifact_namespace: str) -> pd.DataFrame:
    path = evaluation_artifact_path(filename, artifact_namespace)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def get_options() -> dict[str, Any]:
    metadata, metadata_path = _load_filtered_metadata(None)
    source_types = sorted(metadata["source_type"].dropna().astype(str).unique().tolist()) if not metadata.empty else []
    default_source_types = source_types or [UPLOAD_SOURCE_TYPE]
    namespace = _artifact_namespace(metadata_path, None)
    catalog, catalog_path = build_query_catalog(metadata, namespace) if not metadata.empty else (pd.DataFrame(), Path(""))
    model_aliases = list(list_available_models(include_optional=False).keys())
    adaptive_context = (
        build_adaptive_context(metadata, catalog, text_source=DEFAULT_TEXT_SOURCE, artifact_namespace=namespace)
        if not metadata.empty
        else None
    )
    return {
        "sourceTypes": source_types,
        "defaultSourceTypes": default_source_types,
        "textSources": ["stt_transcript", "original_transcript", "combined"],
        "keywordMethods": ["bm25", "tfidf"],
        "modelAliases": model_aliases,
        "clusterMethods": ["kmeans", "hdbscan"],
        "plotModes": list(PLOT_MODE_OPTIONS.keys()),
        "colorFields": ["category", "cluster_id"],
        "queryCatalog": _table_payload(catalog),
        "queryCatalogPath": str(catalog_path),
        "metadataPath": str(metadata_path),
        "artifactNamespace": namespace,
        "documentCount": int(len(metadata)),
        "adaptive": _adaptive_payload(adaptive_context),
        "recommendedTopK": resolve_top_k(adaptive_context.profile, None) if adaptive_context is not None else 5,
        "recommendedClusters": adaptive_context.cluster.n_clusters if adaptive_context is not None else None,
        "device": device_payload(),
    }


def handle_search(payload: dict[str, Any]) -> dict[str, Any]:
    source_types = list(payload.get("sourceTypes") or [])
    text_source = "stt_transcript"
    keyword_method = "bm25"
    requested_top_k = None
    cluster_method = "kmeans"
    query_input = str(payload.get("query") or "").strip()
    debug_mode = bool(payload.get("debugMode", False))
    include_details = bool(payload.get("includeDetails", False)) or debug_mode
    model_aliases = list(list_available_models(include_optional=False).keys())
    model_a = "paraphrase-multilingual-MiniLM-L12-v2" if "paraphrase-multilingual-MiniLM-L12-v2" in model_aliases else model_aliases[0]
    model_b = "multilingual-e5-base" if "multilingual-e5-base" in model_aliases else model_aliases[min(1, len(model_aliases) - 1)]

    if not query_input:
        return {"ok": False, "error": "검색 문장을 입력해 주세요.", "device": device_payload()}

    metadata, metadata_path = _load_filtered_metadata(source_types)
    if metadata.empty:
        return {"ok": False, "error": "선택한 source_type에 해당하는 데이터가 없습니다.", "device": device_payload()}

    artifact_namespace = _artifact_namespace(metadata_path, source_types)
    catalog, catalog_path = build_query_catalog(metadata, artifact_namespace)
    adaptive_context = build_adaptive_context(
        metadata,
        catalog,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
    )
    top_k = resolve_top_k(adaptive_context.profile, requested_top_k)
    query_row = _selected_query_row(catalog, payload.get("queryId"), query_input, metadata, text_source)
    query = str(query_row.get("query", ""))
    ui_catalog = catalog.copy()
    if query:
        selected_query_frame = pd.DataFrame([query_row.to_dict()])
        if ui_catalog.empty:
            ui_catalog = selected_query_frame
        elif not ui_catalog["query_id"].fillna("").astype(str).eq(str(query_row.get("query_id", ""))).any():
            ui_catalog = pd.concat([selected_query_frame, ui_catalog], ignore_index=True)

    try:
        ensure_artifacts(
            metadata,
            artifact_namespace,
            text_source,
            [model_a, model_b],
            cluster_method,
            n_clusters=None,
            adaptive_context=adaptive_context,
        )
        artifact_checks = _validate_fused_search_artifacts(
            metadata,
            artifact_namespace=artifact_namespace,
            text_source=text_source,
            model_aliases=[model_a, model_b],
        )
    except SearchPreparationError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "errorCode": "FUSED_SEARCH_PREPARATION_FAILED",
            "artifactNamespace": artifact_namespace,
            "device": device_payload(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": "Dense model loading failed during progress/stderr handling",
            "errorCode": "MODEL_LOAD_FAILED",
            "detail": str(exc),
            "artifactNamespace": artifact_namespace,
            "device": device_payload(),
        }
    dense_context_a = build_adaptive_context(
        metadata,
        catalog,
        text_source=text_source,
        embedding_model_alias=model_a,
        artifact_namespace=artifact_namespace,
    )
    dense_context_b = build_adaptive_context(
        metadata,
        catalog,
        text_source=text_source,
        embedding_model_alias=model_b,
        artifact_namespace=artifact_namespace,
    )
    fused_engine = FusedSearchEngine(
        metadata,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
        adaptive_context=adaptive_context,
        minilm_alias=model_a,
        e5_alias=model_b,
    )
    fused_results = fused_engine.search(query, top_k=top_k, keyword_method=keyword_method)
    fused_metric = resolve_metric_config(
        adaptive_context.profile,
        score_values=fused_results["display_score"].astype(float).tolist(),
    )
    fused_table, fused_relevant = annotate_search_results(
        fused_results,
        query_row,
        metadata,
        top_k,
        preview_mode="keyword",
        keyword_method=keyword_method,
        hallucination_threshold=fused_metric.hallucination_threshold,
    )
    summary_rows = [build_query_summary("fused-retrieval", fused_table, fused_relevant, top_k)]
    debug_tables: dict[str, dict[str, Any]] = {}
    if debug_mode:
        keyword_engine = KeywordSearchEngine(
            metadata,
            text_source=text_source,
            adaptive_context=adaptive_context,
            artifact_namespace=artifact_namespace,
        )
        dense_engine_a = DenseSearchEngine(
            metadata,
            model_a,
            text_source=text_source,
            artifact_namespace=artifact_namespace,
            adaptive_context=dense_context_a,
        )
        dense_engine_b = DenseSearchEngine(
            metadata,
            model_b,
            text_source=text_source,
            artifact_namespace=artifact_namespace,
            adaptive_context=dense_context_b,
        )
        dense_engine_a.load()
        dense_engine_b.load()
        keyword_results = keyword_engine.search(query, top_k=top_k, method=keyword_method)
        dense_results_a = dense_engine_a.search(query, top_k=top_k)
        dense_results_b = dense_engine_b.search(query, top_k=top_k)
        keyword_metric = resolve_metric_config(
            adaptive_context.profile,
            score_values=keyword_results["display_score"].astype(float).tolist(),
        )
        dense_metric_a = resolve_metric_config(
            dense_context_a.profile,
            score_values=dense_results_a["display_score"].astype(float).tolist(),
        )
        dense_metric_b = resolve_metric_config(
            dense_context_b.profile,
            score_values=dense_results_b["display_score"].astype(float).tolist(),
        )
        keyword_table, keyword_relevant = annotate_search_results(
            keyword_results,
            query_row,
            metadata,
            top_k,
            preview_mode="keyword",
            keyword_method=keyword_method,
            hallucination_threshold=keyword_metric.hallucination_threshold,
        )
        dense_table_a, dense_relevant_a = annotate_search_results(
            dense_results_a,
            query_row,
            metadata,
            top_k,
            preview_mode="dense",
            dense_wrapper=dense_engine_a.wrapper,
            hallucination_threshold=dense_metric_a.hallucination_threshold,
        )
        dense_table_b, dense_relevant_b = annotate_search_results(
            dense_results_b,
            query_row,
            metadata,
            top_k,
            preview_mode="dense",
            dense_wrapper=dense_engine_b.wrapper,
            hallucination_threshold=dense_metric_b.hallucination_threshold,
        )
        summary_rows.extend(
            [
                build_query_summary(f"keyword-{keyword_method}", keyword_table, keyword_relevant, top_k),
                build_query_summary(model_a, dense_table_a, dense_relevant_a, top_k),
                build_query_summary(model_b, dense_table_b, dense_relevant_b, top_k),
            ]
        )
        debug_tables = {
            "keyword": _table_payload(build_search_table(keyword_table)),
            "denseA": _table_payload(build_search_table(dense_table_a)),
            "denseB": _table_payload(build_search_table(dense_table_b)),
        }
    summary = pd.DataFrame(summary_rows)
    fused_current = summary.loc[summary["system_name"] == "fused-retrieval"].head(1)
    fused_current_metrics = _json_safe(fused_current.iloc[0].to_dict()) if not fused_current.empty else {}
    has_ground_truth = bool(fused_current_metrics.get("has_ground_truth", False))
    fused_top1_row = fused_table.iloc[0] if not fused_table.empty else pd.Series(dtype=object)
    final_selection = {
        "doc_id": str(fused_top1_row.get("id", "")),
        "file_name": str(fused_top1_row.get("file_name", "")),
        "relative_match_score": float(fused_top1_row.get("similarity_score", 0.0) or 0.0),
        "normalized_relevance_score": float(fused_top1_row.get("final_score", 0.0) or 0.0) * 100.0,
        "final_score": float(fused_top1_row.get("final_score", 0.0) or 0.0),
        "reranker_score": float(fused_top1_row.get("reranker_score", 0.0) or 0.0),
        "ranking_confidence_score": float(fused_top1_row.get("reranker_score", 0.0) or 0.0) * 100.0,
        "ranking_reason": str(fused_top1_row.get("ranking_reason", "")),
        "chosen_preview_reason": str(fused_top1_row.get("chosen_preview_reason", "")),
        "preview": str(fused_top1_row.get("query_preview", "")),
    }

    eval_catalog = _evaluation_catalog(catalog)
    eval_summary_path = evaluation_artifact_path("retrieval_eval_summary.csv", artifact_namespace)
    should_refresh_eval = include_details or (not eval_summary_path.exists())
    if should_refresh_eval:
        ensure_evaluation_artifacts(metadata, eval_catalog, artifact_namespace, top_k=top_k)
    eval_summary = _load_eval_frame("retrieval_eval_summary.csv", artifact_namespace)
    fused_eval = eval_summary.loc[eval_summary["system_name"] == "fused-retrieval"].head(1).copy() if not eval_summary.empty else pd.DataFrame()
    eval_detail = _load_eval_frame("retrieval_eval_detail.csv", artifact_namespace) if include_details else pd.DataFrame()
    eval_comparison = _load_eval_frame("retrieval_eval_source_comparison.csv", artifact_namespace) if include_details else pd.DataFrame()
    eval_mode_comparison = _load_eval_frame("retrieval_eval_mode_comparison.csv", artifact_namespace) if include_details else pd.DataFrame()
    candidate_ids = fused_results["id"].astype(str).tolist()

    run_summary = load_incremental_run_summary(INCREMENTAL_RUN_SUMMARY_JSON)
    category_counts = metadata["category"].fillna("unknown").value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    return {
        "ok": True,
        "artifactNamespace": artifact_namespace,
        "metadataPath": str(metadata_path),
        "queryCatalogPath": str(catalog_path),
        "queryCatalog": _table_payload(ui_catalog) if include_details else {"columns": [], "rows": []},
        "selectedQuery": _json_safe(query_row.to_dict()),
        "relevantSummary": relevant_file_summary(query_row, metadata),
        "querySummary": _table_payload(summary),
        "fusedCurrentMetrics": fused_current_metrics,
        "evaluationStatus": {
            "has_ground_truth": has_ground_truth,
            "message": (
                "정답 매핑이 없어 평가 불가 상태입니다. 검색 결과 순위만 해석하세요."
                if not has_ground_truth
                else "정답 매핑 기반 평가가 가능합니다."
            ),
        },
        "finalFusedSelection": _json_safe(final_selection),
        "scoreSemantics": _score_semantics_payload(),
        "tables": {
            "fused": _table_payload(build_search_table(fused_table)),
            **debug_tables,
        },
        "primarySystem": "fused",
        "fusedExplainability": _table_payload(
            fused_results[
                [
                    "rank",
                    "id",
                    "bm25_score",
                    "minilm_score",
                    "e5_score",
                    "normalized_bm25",
                    "normalized_minilm",
                    "normalized_e5",
                    "fused_score",
                    "reranker_score",
                    "reranker_alpha",
                    "rerank_anchor_overlap",
                    "reranker_enabled",
                    "dense_normalization_mode",
                    "query_normalization_mode",
                    "adaptive_candidate_pool_k",
                    "adaptive_rerank_top_n",
                    "adaptive_query_bucket",
                    "adaptive_semantic_need",
                    "adaptive_ambiguity",
                    "adaptive_reranker_value",
                    "fusion_weights",
                    "top_model_contribution",
                    "used_fallback_tuning",
                    "fallback_reason",
                    "chosen_preview_reason",
                    "ranking_reason",
                ]
            ]
        ),
        "candidateIds": candidate_ids,
        "evalSummary": _table_payload(eval_summary.loc[eval_summary["system_name"] == "fused-retrieval"].reset_index(drop=True)),
        "fusedGlobalMetrics": _json_safe(fused_eval.iloc[0].to_dict()) if not fused_eval.empty else {},
        "evalDetail": _table_payload(eval_detail, limit=200) if include_details else {"columns": [], "rows": []},
        "evalComparison": _table_payload(eval_comparison) if include_details else {"columns": [], "rows": []},
        "evalModeComparison": _table_payload(eval_mode_comparison) if include_details else {"columns": [], "rows": []},
        "dataset": {
            "documentCount": int(len(metadata)),
            "metadataPath": str(metadata_path),
            "artifactNamespace": artifact_namespace,
            "categoryCounts": _frame_records(category_counts),
            "queryCatalog": _table_payload(ui_catalog) if include_details else {"columns": [], "rows": []},
            "metadataPreview": _table_payload(metadata.head(30)) if include_details else {"columns": [], "rows": []},
            "modelWeights": _table_payload(build_model_weight_frame(include_optional=False)) if include_details else {"columns": [], "rows": []},
            "runSummary": _json_safe(run_summary) if include_details else {},
            "adaptive": _adaptive_payload(adaptive_context),
        },
        "adaptive": {
            "base": _adaptive_payload(adaptive_context),
            "denseA": _adaptive_payload(dense_context_a),
            "denseB": _adaptive_payload(dense_context_b),
            "resolvedTopK": top_k,
        },
        "device": device_payload(),
        "artifactChecks": artifact_checks,
    }


def handle_document(payload: dict[str, Any]) -> dict[str, Any]:
    source_types = list(payload.get("sourceTypes") or [])
    text_source = str(payload.get("textSource") or "stt_transcript")
    doc_id = str(payload.get("docId") or "")
    metadata, _ = _load_filtered_metadata(source_types)
    selected = metadata.loc[metadata["id"].fillna("").astype(str) == doc_id]
    if selected.empty:
        return {"ok": False, "error": "문서를 찾지 못했습니다."}
    row = selected.iloc[0]
    info_fields = [
        "id",
        "source_type",
        "category",
        "file_name",
        "file_path",
        "audio_path",
        "processed_txt_path",
        "keywords",
        "processing_status",
        "stt_model_name",
    ]
    info = [{"field": field, "value": _json_safe(row.get(field, ""))} for field in info_fields]
    line_frame = pd.DataFrame(split_text_into_lines(resolve_primary_text(row, text_source=text_source)))
    return {
        "ok": True,
        "info": info,
        "sttTranscript": str(row.get("stt_transcript", "")),
        "originalTranscript": str(row.get("original_transcript", "")),
        "lines": _table_payload(line_frame),
    }


def handle_plot(payload: dict[str, Any]) -> dict[str, Any]:
    source_types = list(payload.get("sourceTypes") or [])
    text_source = str(payload.get("textSource") or "stt_transcript")
    model_alias = str(payload.get("modelAlias") or "")
    plot_mode = str(payload.get("plotMode") or "PCA 3D")
    cluster_method = str(payload.get("clusterMethod") or "kmeans")
    color_by = str(payload.get("colorBy") or "category")
    requested_top_k = _parse_requested_top_k(payload.get("topK"))
    query = str(payload.get("query") or "")
    method, dimensions = PLOT_MODE_OPTIONS.get(plot_mode, ("pca", 3))
    optional_methods = tuple([method]) if method in {"tsne", "umap"} else ("tsne",)

    metadata, metadata_path = _load_filtered_metadata(source_types)
    artifact_namespace = _artifact_namespace(metadata_path, source_types)
    adaptive_context = build_adaptive_context(
        metadata,
        text_source=text_source,
        embedding_model_alias=model_alias,
        artifact_namespace=artifact_namespace,
    )
    top_k = resolve_top_k(adaptive_context.profile, requested_top_k)
    ensure_artifacts(
        metadata,
        artifact_namespace,
        text_source,
        [model_alias],
        cluster_method,
        n_clusters=None,
        optional_methods=optional_methods,
        adaptive_context=adaptive_context,
    )

    dense_engine = DenseSearchEngine(
        metadata,
        model_alias,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
        adaptive_context=adaptive_context,
    )
    dense_engine.load()
    score_frame = dense_engine.score_all(query)
    results = dense_engine.search(query, top_k=top_k)
    plot_frame = build_plot_frame(model_alias, text_source, method, dimensions, score_frame, cluster_method, artifact_namespace)
    query_coords = project_query_vector(
        model_alias,
        text_source,
        method,
        dimensions,
        dense_engine.encode_query(query),
        artifact_namespace=artifact_namespace,
    )
    fig = build_projection_figure(
        plot_frame,
        method=method,
        dimensions=dimensions,
        color_by=color_by,
        top_result_ids=results["id"].astype(str).tolist(),
        query_point=query_coords[:dimensions] if query_coords is not None else None,
        query_label=query,
        title=f"{model_alias} / {text_source} / {plot_mode}",
    )
    cluster_summary = load_cluster_summary(model_alias, text_source, cluster_method, artifact_namespace)
    representatives = load_representatives(model_alias, text_source=text_source, method=cluster_method, artifact_namespace=artifact_namespace)

    pca_variance = {}
    variance_path = projection_artifact_path(model_alias, text_source, "pca", 3, artifact_namespace=artifact_namespace)
    variance_json = variance_path.with_name(f"{artifact_stem(model_alias, text_source, artifact_namespace)}_pca_variance.json")
    if variance_json.exists():
        pca_variance = load_json(variance_json)

    return {
        "ok": True,
        "figure": json.loads(pio.to_json(fig)),
        "clusterSummary": _json_safe(cluster_summary),
        "representatives": _table_payload(representatives),
        "pcaVariance": _json_safe(pca_variance),
        "adaptive": _adaptive_payload(adaptive_context),
        "resolvedTopK": top_k,
    }


class ExperimentHtmlHandler(BaseHTTPRequestHandler):
    server_version = "ExperimentHtml/1.0"

    def _send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        content = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self._send_bytes(content, "application/json; charset=utf-8", status=status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in {"/", "/index.html", "/experiment_dashboard.html"}:
                self._send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/assets/plotly.min.js":
                plotly_js = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
                self._send_bytes(plotly_js.read_bytes(), "application/javascript; charset=utf-8")
                return
            if path == "/api/options":
                self._send_json({"ok": True, **get_options()})
                return
            if path.startswith("/web/"):
                target = (ROOT_DIR / path.lstrip("/")).resolve()
                if ROOT_DIR not in target.parents or not target.exists():
                    self._send_json({"ok": False, "error": "파일을 찾지 못했습니다."}, status=404)
                    return
                content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self._send_bytes(target.read_bytes(), content_type)
                return
            self._send_json({"ok": False, "error": "지원하지 않는 경로입니다."}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/upload":
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                self._send_json(handle_upload(form))
                return
            payload = self._read_json()
            if parsed.path == "/api/search":
                self._send_json(handle_search(payload))
                return
            if parsed.path == "/api/document":
                self._send_json(handle_document(payload))
                return
            if parsed.path == "/api/plot":
                self._send_json(handle_plot(payload))
                return
            self._send_json({"ok": False, "error": "지원하지 않는 API입니다."}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[html-app] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HTML experiment dashboard server.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not HTML_PATH.exists():
        raise FileNotFoundError(f"Missing HTML file: {HTML_PATH}")

    try:
        from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars

        hf_disable_progress_bars()
    except Exception:
        pass

    server = ThreadingHTTPServer((args.host, args.port), ExperimentHtmlHandler)
    print(f"HTML dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
