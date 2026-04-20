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

import pandas as pd
import plotly
import plotly.io as pio

from src.config import (
    CLUSTERS_DIR,
    DEFAULT_QUERYSET_CSV,
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
from src.search.load_realdata_dataset import dataset_artifact_namespace, load_search_metadata
from src.search.text_source import resolve_primary_text, split_text_into_lines
from src.stt.batch_transcribe import transcribe_audio_batch
from src.utils.io_utils import load_json, save_json
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
    "is_relevant",
    "similarity_score",
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
    hallucination_threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
) -> tuple[pd.DataFrame, set[str]]:
    relevant_ids = resolve_relevant_ids(query_row, metadata)
    query_preview = truncate_text(query_row.get("query", ""), 80)
    hits = 0
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(results.head(top_k).itertuples(index=False), start=1):
        row_map = pd.Series(row._asdict())
        is_relevant = str(getattr(row, "id", "")) in relevant_ids
        hits += int(is_relevant)
        precision = hits / rank
        recall = hits / max(1, len(relevant_ids))
        accuracy = 1.0 if rank == 1 and is_relevant else 0.0
        f1 = f1_score(precision, recall)
        similarity = float(getattr(row, "similarity_score", 0.0) or 0.0)
        hallucination = bool((not is_relevant) and similarity >= hallucination_threshold)
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
                "search_location": resolve_location_display(row_map),
                "query_preview": query_preview,
            }
        )
    return pd.DataFrame(rows), relevant_ids


def build_query_summary(system_name: str, annotated: pd.DataFrame, relevant_ids: set[str], top_k: int) -> dict[str, Any]:
    hits = int(annotated["is_relevant"].sum()) if not annotated.empty else 0
    predicted_count = len(annotated)
    precision = hits / max(1, predicted_count)
    recall = hits / max(1, len(relevant_ids))
    accuracy = float(bool(not annotated.empty and bool(annotated.iloc[0]["is_relevant"])))
    f1 = f1_score(precision, recall)
    hallucination_count = int((annotated["hallucination_flag"] == "YES").sum()) if not annotated.empty else 0
    return {
        "system_name": system_name,
        "relevant_count": len(relevant_ids),
        f"top_{top_k}_hit_count": hits,
        f"top_{top_k}_result_count": predicted_count,
        f"precision@{top_k}": precision,
        f"recall@{top_k}": recall,
        f"f1@{top_k}": f1,
        "accuracy@1_reference": accuracy,
        "metric_definition": "precision/recall/f1 are calculated from retrieved ids and ground truth relevant ids; accuracy is only a top-1 reference.",
        "hallucination_count": hallucination_count,
        "hallucination_rate": hallucination_count / max(1, predicted_count),
        "mean_final_score": float(annotated["final_score"].mean()) if "final_score" in annotated.columns and not annotated.empty else 0.0,
        "mean_similarity_score": float(annotated["similarity_score"].mean()) if not annotated.empty else 0.0,
    }


def build_search_table(annotated: pd.DataFrame) -> pd.DataFrame:
    display = annotated.copy()
    for column in [
        "similarity_score",
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
    n_clusters: int = 6,
    optional_methods: tuple[str, ...] = ("tsne",),
) -> None:
    KeywordSearchEngine(metadata, text_source=text_source).export_index_metadata(artifact_namespace=artifact_namespace)
    for model_alias in model_aliases:
        dense_engine = DenseSearchEngine(metadata, model_alias, text_source=text_source, artifact_namespace=artifact_namespace)
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
                )
            except Exception:
                if cluster_method != "kmeans":
                    pass


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
        return "정답 문서가 없습니다. 이 질의는 점수 계산에서 제외되어야 합니다."
    matched = metadata.loc[metadata["id"].fillna("").astype(str).isin(relevant_ids)]
    names = matched["file_name"].fillna("").astype(str).tolist()
    if names:
        return f"정답 문서 {len(relevant_ids)}개: {', '.join(names[:3])}" + (" ..." if len(names) > 3 else "")
    return f"정답 문서 {len(relevant_ids)}개: {', '.join(sorted(relevant_ids)[:3])}"


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
    whisper_model = str(form.getfirst("whisperModel", "base") or "base").strip() or "base"
    language = str(form.getfirst("language", "ko") or "ko").strip() or "ko"

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
        raise

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
        "title": str(row.get("title", "")),
        "searchQuery": search_query,
        "transcriptPreview": transcript[:240],
        "metadataPath": str(REALDATA_METADATA_CSV),
        "uploadRoot": str(HTML_UPLOAD_MEDIA_DIR.parent),
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
        row = build_metadata_token_query_row(query_text or "유튜브 전사 검색", metadata, text_source=text_source)
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
            row = build_metadata_token_query_row(query_text, metadata, text_source=text_source)
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
    }


def handle_search(payload: dict[str, Any]) -> dict[str, Any]:
    source_types = list(payload.get("sourceTypes") or [])
    text_source = str(payload.get("textSource") or "stt_transcript")
    keyword_method = str(payload.get("keywordMethod") or "bm25")
    top_k = max(1, min(50, int(payload.get("topK") or 5)))
    cluster_method = str(payload.get("clusterMethod") or "kmeans")
    query_input = str(payload.get("query") or "").strip()
    model_aliases = list(list_available_models(include_optional=False).keys())
    model_a = str(payload.get("modelA") or model_aliases[0])
    model_b = str(payload.get("modelB") or model_aliases[min(1, len(model_aliases) - 1)])

    if not query_input:
        return {"ok": False, "error": "검색 문장을 입력해 주세요."}

    metadata, metadata_path = _load_filtered_metadata(source_types)
    if metadata.empty:
        return {"ok": False, "error": "선택한 source_type에 해당하는 데이터가 없습니다."}

    artifact_namespace = _artifact_namespace(metadata_path, source_types)
    catalog, catalog_path = build_query_catalog(metadata, artifact_namespace)
    query_row = _selected_query_row(catalog, payload.get("queryId"), query_input, metadata, text_source)
    query = str(query_row.get("query", ""))
    ui_catalog = catalog.copy()
    if query:
        selected_query_frame = pd.DataFrame([query_row.to_dict()])
        if ui_catalog.empty:
            ui_catalog = selected_query_frame
        elif not ui_catalog["query_id"].fillna("").astype(str).eq(str(query_row.get("query_id", ""))).any():
            ui_catalog = pd.concat([selected_query_frame, ui_catalog], ignore_index=True)

    ensure_artifacts(metadata, artifact_namespace, text_source, [model_a, model_b], cluster_method, n_clusters=6)
    keyword_engine = KeywordSearchEngine(metadata, text_source=text_source)
    dense_engine_a = DenseSearchEngine(metadata, model_a, text_source=text_source, artifact_namespace=artifact_namespace)
    dense_engine_b = DenseSearchEngine(metadata, model_b, text_source=text_source, artifact_namespace=artifact_namespace)
    dense_engine_a.load()
    dense_engine_b.load()

    keyword_results = keyword_engine.search(query, top_k=top_k, method=keyword_method)
    dense_results_a = dense_engine_a.search(query, top_k=top_k)
    dense_results_b = dense_engine_b.search(query, top_k=top_k)
    keyword_table, keyword_relevant = annotate_search_results(keyword_results, query_row, metadata, top_k)
    dense_table_a, dense_relevant_a = annotate_search_results(dense_results_a, query_row, metadata, top_k)
    dense_table_b, dense_relevant_b = annotate_search_results(dense_results_b, query_row, metadata, top_k)
    summary = pd.DataFrame(
        [
            build_query_summary(f"keyword-{keyword_method}", keyword_table, keyword_relevant, top_k),
            build_query_summary(model_a, dense_table_a, dense_relevant_a, top_k),
            build_query_summary(model_b, dense_table_b, dense_relevant_b, top_k),
        ]
    )

    ensure_evaluation_artifacts(metadata, catalog, artifact_namespace, top_k=top_k)
    eval_summary = _load_eval_frame("retrieval_eval_summary.csv", artifact_namespace)
    eval_detail = _load_eval_frame("retrieval_eval_detail.csv", artifact_namespace)
    eval_comparison = _load_eval_frame("retrieval_eval_source_comparison.csv", artifact_namespace)
    candidate_ids = pd.unique(
        pd.concat([keyword_results["id"], dense_results_a["id"], dense_results_b["id"]], ignore_index=True)
    ).astype(str).tolist()

    run_summary = load_incremental_run_summary(INCREMENTAL_RUN_SUMMARY_JSON)
    category_counts = metadata["category"].fillna("unknown").value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    return {
        "ok": True,
        "artifactNamespace": artifact_namespace,
        "metadataPath": str(metadata_path),
        "queryCatalogPath": str(catalog_path),
        "queryCatalog": _table_payload(ui_catalog),
        "selectedQuery": _json_safe(query_row.to_dict()),
        "relevantSummary": relevant_file_summary(query_row, metadata),
        "querySummary": _table_payload(summary),
        "tables": {
            "keyword": _table_payload(build_search_table(keyword_table)),
            "denseA": _table_payload(build_search_table(dense_table_a)),
            "denseB": _table_payload(build_search_table(dense_table_b)),
        },
        "candidateIds": candidate_ids,
        "evalSummary": _table_payload(eval_summary),
        "evalDetail": _table_payload(eval_detail, limit=200),
        "evalComparison": _table_payload(eval_comparison),
        "dataset": {
            "documentCount": int(len(metadata)),
            "metadataPath": str(metadata_path),
            "artifactNamespace": artifact_namespace,
            "categoryCounts": _frame_records(category_counts),
            "queryCatalog": _table_payload(ui_catalog),
            "metadataPreview": _table_payload(metadata.head(30)),
            "modelWeights": _table_payload(build_model_weight_frame(include_optional=False)),
            "runSummary": _json_safe(run_summary),
        },
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
    top_k = max(1, min(50, int(payload.get("topK") or 5)))
    query = str(payload.get("query") or "")
    method, dimensions = PLOT_MODE_OPTIONS.get(plot_mode, ("pca", 3))
    optional_methods = tuple([method]) if method in {"tsne", "umap"} else ("tsne",)

    metadata, metadata_path = _load_filtered_metadata(source_types)
    artifact_namespace = _artifact_namespace(metadata_path, source_types)
    ensure_artifacts(metadata, artifact_namespace, text_source, [model_alias], cluster_method, optional_methods=optional_methods)

    dense_engine = DenseSearchEngine(metadata, model_alias, text_source=text_source, artifact_namespace=artifact_namespace)
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
    server = ThreadingHTTPServer((args.host, args.port), ExperimentHtmlHandler)
    print(f"HTML dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
