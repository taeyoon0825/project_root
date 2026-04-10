from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CLUSTERS_DIR, DEFAULT_QUERYSET_CSV, INCREMENTAL_RUN_SUMMARY_JSON, REALDATA_METADATA_CSV
from src.embedding.build_indices import DenseSearchEngine, artifact_stem
from src.embedding.vector_models import list_available_models
from src.evaluation.evaluate import evaluate_all, evaluation_artifact_path
from src.evaluation.metrics_report import (
    DEFAULT_HALLUCINATION_THRESHOLD,
    build_incremental_probe_queryset,
    build_model_weight_frame,
    normalize_ground_truth_queryset,
    resolve_relevant_ids,
)
from src.ingest.incremental_registry import load_incremental_run_summary
from src.search.keyword_search import KeywordSearchEngine
from src.search.load_realdata_dataset import dataset_artifact_namespace, load_search_metadata
from src.search.text_source import resolve_primary_text, split_text_into_lines
from src.utils.io_utils import load_json, save_json
from src.visualize.clustering import load_representatives
from src.visualize.interactive_plot import (
    build_projection_figure,
    load_projection_frame,
    project_query_vector,
    projection_artifact_path,
)
from src.visualize.pca_plot import build_projection_artifacts
from src.visualize.clustering import cluster_embeddings


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
    "similarity_score",
    "accuracy",
    "precision",
    "f1_score",
    "hallucination_flag",
    "search_location",
    "query_preview",
]

st.set_page_config(page_title="YouTube STT Retrieval Experiment", layout="wide")


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


def file_token(path: Path) -> str:
    if not path.exists():
        return str(path)
    stat = path.stat()
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def metadata_content_token(metadata: pd.DataFrame, text_source: str) -> str:
    digest = hashlib.sha1()
    digest.update(text_source.encode("utf-8"))
    if metadata.empty:
        digest.update(b"__empty__")
        return digest.hexdigest()
    primary_texts = metadata.apply(lambda row: resolve_primary_text(row, text_source=text_source), axis=1)
    for doc_id, primary_text in zip(metadata["id"].astype(str).tolist(), primary_texts.astype(str).tolist()):
        digest.update(doc_id.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(primary_text.encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def evaluation_content_token(metadata: pd.DataFrame, query_catalog: pd.DataFrame) -> str:
    digest = hashlib.sha1()
    digest.update(b"evaluation")
    for row in metadata.fillna("").itertuples(index=False):
        digest.update(str(getattr(row, "id", "")).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(str(getattr(row, "stt_transcript", "")).encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    for row in query_catalog.fillna("").itertuples(index=False):
        digest.update(str(getattr(row, "query_id", "")).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(str(getattr(row, "query", "")).encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_ids_match(csv_path: Path, metadata: pd.DataFrame) -> bool:
    if not csv_path.exists():
        return False
    try:
        artifact_ids = pd.read_csv(csv_path, usecols=["id"])["id"].fillna("").astype(str).tolist()
    except Exception:
        return False
    expected_ids = metadata["id"].fillna("").astype(str).tolist()
    return artifact_ids == expected_ids


@st.cache_data
def load_metadata_for_app(source_types: tuple[str, ...], refresh_token: str) -> tuple[pd.DataFrame, str]:
    _ = refresh_token
    metadata, metadata_path = load_search_metadata(REALDATA_METADATA_CSV, source_types or None)
    return metadata, str(metadata_path)


@st.cache_data
def load_base_queryset(query_token: str) -> pd.DataFrame:
    _ = query_token
    if DEFAULT_QUERYSET_CSV.exists():
        return pd.read_csv(DEFAULT_QUERYSET_CSV)
    return pd.DataFrame(columns=["query_id", "query", "target_category", "relevant_id", "target_source_type"])


@st.cache_data
def load_incremental_summary(summary_token: str) -> dict[str, Any]:
    _ = summary_token
    return load_incremental_run_summary(INCREMENTAL_RUN_SUMMARY_JSON)


def build_query_catalog(metadata: pd.DataFrame, artifact_namespace: str) -> tuple[pd.DataFrame, Path]:
    base_queryset = load_base_queryset(file_token(DEFAULT_QUERYSET_CSV))
    if base_queryset.empty:
        base_queryset = build_incremental_probe_queryset(metadata, set(metadata["id"].fillna("").astype(str)))
    catalog = normalize_ground_truth_queryset(base_queryset, metadata).reset_index(drop=True)
    if catalog.empty:
        probe_queryset = build_incremental_probe_queryset(metadata, set(metadata["id"].fillna("").astype(str)))
        catalog = normalize_ground_truth_queryset(probe_queryset, metadata).reset_index(drop=True)

    catalog_path = evaluation_artifact_path("query_catalog.json", artifact_namespace)
    save_json(catalog_path, catalog.to_dict(orient="records"))
    return catalog, catalog_path


def _normalize_for_match(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _segment_payload_from_row(row: pd.Series) -> dict[str, Any] | None:
    candidates = [
        str(row.get("stt_txt_path", "")).strip(),
        str(row.get("processed_txt_path", "")).strip(),
    ]
    for candidate in candidates:
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
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
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
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    hallucination_count = int((annotated["hallucination_flag"] == "YES").sum()) if not annotated.empty else 0
    return {
        "system_name": system_name,
        f"Recall@{top_k}": recall,
        f"Precision@{top_k}": precision,
        "Accuracy@1": accuracy,
        f"F1@{top_k}": f1,
        "Hallucination 수": hallucination_count,
        "Hallucination 비율": hallucination_count / max(1, predicted_count),
        "평균 Similarity": float(annotated["similarity_score"].mean()) if not annotated.empty else 0.0,
    }


def build_search_table(annotated: pd.DataFrame) -> pd.DataFrame:
    display = annotated.copy()
    for column in ["similarity_score", "accuracy", "precision", "recall", "f1_score"]:
        if column in display.columns:
            display[column] = display[column].astype(float).round(4)
    available = [column for column in SEARCH_TABLE_COLUMNS if column in display.columns]
    return display[available]


@st.cache_data
def load_pca_variance(model_alias: str, text_source: str, artifact_namespace: str) -> dict[str, Any]:
    from src.config import PLOTS_DIR

    return load_json(PLOTS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_pca_variance.json")


@st.cache_data
def load_cluster_frame(model_alias: str, text_source: str, method: str, artifact_namespace: str) -> pd.DataFrame:
    from src.visualize.clustering import load_cluster_frame as _load_cluster_frame

    return _load_cluster_frame(model_alias, text_source=text_source, method=method, artifact_namespace=artifact_namespace)


@st.cache_data
def load_cluster_summary(model_alias: str, text_source: str, method: str, artifact_namespace: str) -> dict[str, Any]:
    from src.visualize.clustering import load_cluster_summary as _load_cluster_summary

    return _load_cluster_summary(model_alias, text_source=text_source, method=method, artifact_namespace=artifact_namespace)


@st.cache_data
def load_eval_summary(artifact_namespace: str, eval_token: str) -> pd.DataFrame:
    _ = eval_token
    path = evaluation_artifact_path("retrieval_eval_summary.csv", artifact_namespace)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_eval_detail(artifact_namespace: str, eval_token: str) -> pd.DataFrame:
    _ = eval_token
    path = evaluation_artifact_path("retrieval_eval_detail.csv", artifact_namespace)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_eval_comparison(artifact_namespace: str, eval_token: str) -> pd.DataFrame:
    _ = eval_token
    path = evaluation_artifact_path("retrieval_eval_source_comparison.csv", artifact_namespace)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource
def get_keyword_engine(source_types: tuple[str, ...], text_source: str, metadata_token: str) -> KeywordSearchEngine:
    _ = metadata_token
    metadata, _ = load_search_metadata(REALDATA_METADATA_CSV, source_types or None)
    return KeywordSearchEngine(metadata, text_source=text_source)


@st.cache_resource
def get_dense_engine(
    source_types: tuple[str, ...],
    model_alias: str,
    text_source: str,
    artifact_namespace: str,
    metadata_token: str,
) -> DenseSearchEngine:
    _ = metadata_token
    metadata, _ = load_search_metadata(REALDATA_METADATA_CSV, source_types or None)
    engine = DenseSearchEngine(metadata, model_alias, text_source=text_source, artifact_namespace=artifact_namespace)
    engine.load()
    return engine


def ensure_artifacts(
    metadata: pd.DataFrame,
    artifact_namespace: str,
    text_source: str,
    model_aliases: list[str],
    cluster_method: str,
    n_clusters: int = 6,
) -> None:
    KeywordSearchEngine(metadata, text_source=text_source).export_index_metadata(artifact_namespace=artifact_namespace)
    for model_alias in model_aliases:
        dense_engine = DenseSearchEngine(metadata, model_alias, text_source=text_source, artifact_namespace=artifact_namespace)
        dense_engine.load()

        pca_path = projection_artifact_path(model_alias, text_source, "pca", 3, artifact_namespace=artifact_namespace)
        tsne_2d_path = projection_artifact_path(model_alias, text_source, "tsne", 2, artifact_namespace=artifact_namespace)
        tsne_3d_path = projection_artifact_path(model_alias, text_source, "tsne", 3, artifact_namespace=artifact_namespace)
        projection_is_stale = (
            not artifact_ids_match(pca_path, metadata)
            or (tsne_2d_path.exists() and not artifact_ids_match(tsne_2d_path, metadata))
            or (tsne_3d_path.exists() and not artifact_ids_match(tsne_3d_path, metadata))
        )
        if projection_is_stale:
            build_projection_artifacts(
                metadata,
                model_alias,
                text_source=text_source,
                optional_methods=("tsne",),
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


def ensure_evaluation_artifacts(metadata: pd.DataFrame, query_catalog: pd.DataFrame, artifact_namespace: str) -> None:
    required_paths = [
        evaluation_artifact_path("retrieval_eval_summary.csv", artifact_namespace),
        evaluation_artifact_path("retrieval_eval_detail.csv", artifact_namespace),
        evaluation_artifact_path("retrieval_eval_source_comparison.csv", artifact_namespace),
    ]
    if all(path.exists() for path in required_paths):
        return
    evaluate_all(
        metadata=metadata,
        queryset=query_catalog,
        text_sources=evaluation_text_sources(metadata),
        include_optional=False,
        artifact_namespace=artifact_namespace,
        print_report=False,
        show_weights=False,
    )


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


def render_projection_panel(
    model_alias: str,
    text_source: str,
    method: str,
    dimensions: int,
    dense_engine: DenseSearchEngine,
    score_frame: pd.DataFrame,
    top_k_ids: list[str],
    query: str,
    color_by: str,
    cluster_method: str,
    artifact_namespace: str,
) -> None:
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
        top_result_ids=top_k_ids,
        query_point=query_coords[:dimensions] if query_coords is not None else None,
        query_label=query,
        title=f"{model_alias} / {text_source} / {method.upper()} {dimensions}D",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pca_metrics(model_alias: str, text_source: str, artifact_namespace: str) -> None:
    variance = load_pca_variance(model_alias, text_source, artifact_namespace)
    ratios = variance["explained_variance_ratio"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("PC1", f"{ratios[0]:.4f}")
    metric_cols[1].metric("PC2", f"{ratios[1]:.4f}")
    metric_cols[2].metric("PC3", f"{ratios[2]:.4f}")
    metric_cols[3].metric("PC1+PC2+PC3", f"{variance['cumulative_first3']:.4f}")


def render_document_detail(metadata: pd.DataFrame, doc_id: str, text_source: str) -> None:
    selected = metadata.loc[metadata["id"] == doc_id]
    if selected.empty:
        st.info("선택한 문서를 찾지 못했습니다.")
        return

    row = selected.iloc[0]
    info = pd.DataFrame(
        [
            {"field": "id", "value": row["id"]},
            {"field": "source_type", "value": row["source_type"]},
            {"field": "category", "value": row["category"]},
            {"field": "file_name", "value": row["file_name"]},
            {"field": "file_path", "value": row["file_path"]},
            {"field": "audio_path", "value": row["audio_path"]},
            {"field": "processed_txt_path", "value": row["processed_txt_path"]},
            {"field": "keywords", "value": row["keywords"]},
            {"field": "processing_status", "value": row["processing_status"]},
            {"field": "stt_model_name", "value": row["stt_model_name"]},
        ]
    )
    st.dataframe(info, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**STT 전사**")
        st.text_area("stt_transcript", value=row["stt_transcript"], height=260, key=f"stt_{doc_id}")
    with right:
        st.markdown("**원문/대체 텍스트**")
        st.text_area("original_transcript", value=row["original_transcript"], height=260, key=f"orig_{doc_id}")

    line_frame = pd.DataFrame(split_text_into_lines(resolve_primary_text(row, text_source=text_source)))
    if not line_frame.empty:
        st.markdown(f"**{text_source} 라인 보기**")
        st.dataframe(line_frame, use_container_width=True, height=260, hide_index=True)


def main() -> None:
    st.title("유튜브 다운로드 데이터 검색 실험 앱")
    st.caption("검색 비교 화면은 단순화하고, 문서 비교/벡터 분포/클러스터/평가/데이터셋 탭은 실제 youtube_mp4 데이터 기준으로 유지합니다.")

    full_metadata, metadata_path_str = load_metadata_for_app(tuple(), file_token(REALDATA_METADATA_CSV))
    if full_metadata.empty:
        st.warning("실제 유튜브 metadata가 아직 없습니다. 먼저 `run_real_mp4_pipeline.py --real-only`를 실행해 주세요.")
        st.stop()
    source_type_options = sorted(full_metadata["source_type"].dropna().astype(str).unique().tolist())
    default_sources = source_type_options or ["youtube_mp4"]

    with st.sidebar:
        st.header("실험 설정")
        selected_source_types = st.multiselect("source_type 필터", options=source_type_options, default=default_sources)
        search_source = st.selectbox("검색 텍스트 기준", options=["stt_transcript", "original_transcript", "combined"], index=0)
        keyword_method = st.selectbox("키워드 방식", options=["bm25", "tfidf"], index=0)
        model_aliases = list(list_available_models(include_optional=False).keys())
        model_a = st.selectbox("Dense 모델 A", options=model_aliases, index=0)
        model_b = st.selectbox("Dense 모델 B", options=model_aliases, index=min(1, len(model_aliases) - 1))
        top_k = st.slider("Top-K", min_value=3, max_value=15, value=5)
        cluster_method = st.selectbox("클러스터 방식", options=["kmeans", "hdbscan"], index=0)

    metadata, metadata_path_str = load_metadata_for_app(tuple(selected_source_types), file_token(REALDATA_METADATA_CSV))
    if metadata.empty:
        st.warning("현재 실제 유튜브 데이터가 없습니다. 먼저 증분 파이프라인을 실행해 주세요.")
        st.stop()

    artifact_namespace = dataset_artifact_namespace(Path(metadata_path_str), tuple(selected_source_types) or None)
    query_catalog, query_catalog_path = build_query_catalog(metadata, artifact_namespace)
    default_query = query_catalog.iloc[0]["query"] if not query_catalog.empty else "유튜브 전사 검색"

    with st.sidebar:
        selected_query_id = st.selectbox(
            "평가/검색 질의",
            options=query_catalog["query_id"].tolist() if not query_catalog.empty else ["manual"],
            index=0,
        )
        selected_query_row = (
            query_catalog.loc[query_catalog["query_id"] == selected_query_id].iloc[0]
            if not query_catalog.empty
            else pd.Series({"query_id": "manual", "query": default_query})
        )
        query = st.text_input("검색 문장", value=str(selected_query_row.get("query", default_query)))
        vector_plot_mode = st.selectbox("벡터 분포 모드", options=list(PLOT_MODE_OPTIONS.keys()), index=0)
        cluster_plot_mode = st.selectbox("클러스터 보기 모드", options=list(PLOT_MODE_OPTIONS.keys()), index=0)
        vector_color_by = st.selectbox("벡터 색상 기준", options=["category", "cluster_id", "source_type"], index=0)

    query_row = selected_query_row.copy()
    query_row["query"] = query
    query_row["query_preview"] = truncate_text(query, 80)

    metadata_token = metadata_content_token(metadata, search_source)
    with st.spinner("검색/시각화 아티팩트 확인 중..."):
        ensure_artifacts(metadata, artifact_namespace, search_source, [model_a, model_b], cluster_method, n_clusters=6)

    keyword_engine = get_keyword_engine(tuple(selected_source_types), search_source, metadata_token)
    dense_engine_a = get_dense_engine(tuple(selected_source_types), model_a, search_source, artifact_namespace, metadata_token)
    dense_engine_b = get_dense_engine(tuple(selected_source_types), model_b, search_source, artifact_namespace, metadata_token)

    keyword_results = keyword_engine.search(query, top_k=top_k, method=keyword_method)
    dense_results_a = dense_engine_a.search(query, top_k=top_k)
    dense_results_b = dense_engine_b.search(query, top_k=top_k)
    dense_scores_a = dense_engine_a.score_all(query)
    dense_scores_b = dense_engine_b.score_all(query)

    keyword_table, keyword_relevant = annotate_search_results(keyword_results, query_row, metadata, top_k)
    dense_table_a, dense_relevant_a = annotate_search_results(dense_results_a, query_row, metadata, top_k)
    dense_table_b, dense_relevant_b = annotate_search_results(dense_results_b, query_row, metadata, top_k)
    query_summary = pd.DataFrame(
        [
            build_query_summary(f"keyword-{keyword_method}", keyword_table, keyword_relevant, top_k),
            build_query_summary(model_a, dense_table_a, dense_relevant_a, top_k),
            build_query_summary(model_b, dense_table_b, dense_relevant_b, top_k),
        ]
    )

    ensure_evaluation_artifacts(metadata, query_catalog, artifact_namespace)
    eval_token = evaluation_content_token(metadata, query_catalog)
    eval_summary = load_eval_summary(artifact_namespace, eval_token)
    eval_detail = load_eval_detail(artifact_namespace, eval_token)
    eval_comparison = load_eval_comparison(artifact_namespace, eval_token)
    run_summary = load_incremental_summary(file_token(INCREMENTAL_RUN_SUMMARY_JSON))

    tabs = st.tabs(["검색 비교", "문서 비교", "벡터 분포", "클러스터", "평가", "데이터셋"])
    tab_search, tab_detail, tab_vectors, tab_clusters, tab_eval, tab_data = tabs

    with tab_search:
        st.subheader("검색 비교")
        st.dataframe(query_summary.round(4), use_container_width=True, hide_index=True)
        if not eval_summary.empty:
            st.markdown("**평균 평가 요약**")
            st.dataframe(eval_summary.round(4), use_container_width=True, hide_index=True)
        with st.expander("질의 JSON 프리뷰", expanded=False):
            st.json(query_row.to_dict())
            st.caption(f"query catalog: {query_catalog_path}")

        left, center, right = st.columns(3)
        with left:
            st.markdown(f"**키워드 / {keyword_method.upper()}**")
            st.dataframe(build_search_table(keyword_table), use_container_width=True, height=420, hide_index=True)
        with center:
            st.markdown(f"**Dense / {model_a}**")
            st.dataframe(build_search_table(dense_table_a), use_container_width=True, height=420, hide_index=True)
        with right:
            st.markdown(f"**Dense / {model_b}**")
            st.dataframe(build_search_table(dense_table_b), use_container_width=True, height=420, hide_index=True)

    with tab_detail:
        st.subheader("문서 비교")
        candidate_ids = pd.unique(
            pd.concat([keyword_results["id"], dense_results_a["id"], dense_results_b["id"]], ignore_index=True)
        ).tolist()
        selected_doc_id = st.selectbox("문서 ID", options=candidate_ids or metadata["id"].tolist(), index=0)
        render_document_detail(metadata, selected_doc_id, search_source)

    with tab_vectors:
        st.subheader("벡터 분포")
        vector_method, vector_dimensions = PLOT_MODE_OPTIONS[vector_plot_mode]
        for model_alias, engine, score_frame, results in [
            (model_a, dense_engine_a, dense_scores_a, dense_results_a),
            (model_b, dense_engine_b, dense_scores_b, dense_results_b),
        ]:
            st.markdown(f"**{model_alias} / {search_source} / {vector_plot_mode}**")
            if vector_method == "pca":
                render_pca_metrics(model_alias, search_source, artifact_namespace)
            render_projection_panel(
                model_alias=model_alias,
                text_source=search_source,
                method=vector_method,
                dimensions=vector_dimensions,
                dense_engine=engine,
                score_frame=score_frame,
                top_k_ids=results["id"].tolist(),
                query=query,
                color_by=vector_color_by,
                cluster_method=cluster_method,
                artifact_namespace=artifact_namespace,
            )

    with tab_clusters:
        st.subheader("클러스터")
        cluster_view_method, cluster_view_dimensions = PLOT_MODE_OPTIONS[cluster_plot_mode]
        for model_alias, engine, score_frame, results in [
            (model_a, dense_engine_a, dense_scores_a, dense_results_a),
            (model_b, dense_engine_b, dense_scores_b, dense_results_b),
        ]:
            st.markdown(f"**{model_alias} / {cluster_plot_mode} / {cluster_method}**")
            summary = load_cluster_summary(model_alias, search_source, cluster_method, artifact_namespace)
            metric_cols = st.columns(3)
            metric_cols[0].metric("요청 클러스터 수", summary.get("n_clusters_requested", "-"))
            metric_cols[1].metric("실제 클러스터 수", summary.get("n_clusters_found", "-"))
            silhouette = summary.get("silhouette_score")
            metric_cols[2].metric("Silhouette", "-" if silhouette is None or pd.isna(silhouette) else f"{float(silhouette):.4f}")
            render_projection_panel(
                model_alias=model_alias,
                text_source=search_source,
                method=cluster_view_method,
                dimensions=cluster_view_dimensions,
                dense_engine=engine,
                score_frame=score_frame,
                top_k_ids=results["id"].tolist(),
                query=query,
                color_by="cluster_id",
                cluster_method=cluster_method,
                artifact_namespace=artifact_namespace,
            )
            st.dataframe(
                load_representatives(model_alias, text_source=search_source, method=cluster_method, artifact_namespace=artifact_namespace),
                use_container_width=True,
                height=220,
            )

    with tab_eval:
        st.subheader("평가")
        if eval_summary.empty:
            st.info("현재 평가 결과가 없습니다.")
        else:
            st.markdown("**시스템 평균 요약**")
            st.dataframe(eval_summary.round(4), use_container_width=True, hide_index=True)
            if not eval_comparison.empty:
                st.markdown("**텍스트 소스 비교**")
                st.dataframe(eval_comparison.round(4), use_container_width=True, hide_index=True)
            if not eval_detail.empty:
                st.markdown("**질의별 상세**")
                st.dataframe(eval_detail, use_container_width=True, height=360, hide_index=True)

    with tab_data:
        st.subheader("데이터셋")
        top_left, top_right = st.columns(2)
        with top_left:
            st.write(f"실제 metadata 경로: `{metadata_path_str}`")
            st.write(f"artifact namespace: `{artifact_namespace}`")
            st.write(f"문서 수: {len(metadata)}")
        with top_right:
            if run_summary:
                st.json(run_summary)
        st.markdown("**모델/가중치 상태**")
        st.dataframe(build_model_weight_frame(include_optional=False), use_container_width=True, hide_index=True)
        category_counts = metadata["category"].fillna("unknown").value_counts().reset_index()
        category_counts.columns = ["category", "count"]
        st.plotly_chart(px.bar(category_counts, x="category", y="count", title="카테고리 분포"), use_container_width=True)
        st.markdown("**질의 카탈로그**")
        st.dataframe(query_catalog, use_container_width=True, height=220, hide_index=True)
        st.markdown("**메타데이터 미리보기**")
        st.dataframe(metadata.head(30), use_container_width=True, height=360)


if __name__ == "__main__":
    main()
