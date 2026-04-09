from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CLUSTERS_DIR
from src.embedding.build_indices import DenseSearchEngine, artifact_stem
from src.embedding.vector_models import list_available_models
from src.evaluation.evaluate import evaluate_all, evaluation_artifact_path
from src.search.keyword_search import KeywordSearchEngine
from src.search.load_realdata_dataset import (
    available_dataset_options,
    dataset_artifact_namespace,
    default_search_metadata_path,
    load_search_metadata,
    resolve_dataset_path,
)
from src.search.text_source import resolve_primary_text, split_text_into_lines
from src.utils.io_utils import load_json
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


st.set_page_config(page_title="Audio STT Retrieval Experiment", layout="wide")


def dataset_refresh_token(dataset_key: str) -> str:
    metadata_path = resolve_dataset_path(dataset_key)
    if not metadata_path.exists():
        return str(metadata_path)
    stat = metadata_path.stat()
    return f"{metadata_path}:{stat.st_size}:{stat.st_mtime_ns}"


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


def evaluation_content_token(metadata: pd.DataFrame, queryset: pd.DataFrame) -> str:
    digest = hashlib.sha1()
    digest.update(b"evaluation")

    if metadata.empty:
        digest.update(b"__empty_metadata__")
    else:
        for row in metadata.fillna("").itertuples(index=False):
            digest.update(str(getattr(row, "id", "")).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(str(getattr(row, "source_type", "")).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(str(getattr(row, "category", "")).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(str(getattr(row, "stt_transcript", "")).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(str(getattr(row, "original_transcript", "")).encode("utf-8", errors="ignore"))
            digest.update(b"\n")

    if queryset.empty:
        digest.update(b"__empty_queryset__")
    else:
        for row in queryset.fillna("").itertuples(index=False):
            digest.update(str(getattr(row, "query_id", "")).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(str(getattr(row, "query", "")).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(str(getattr(row, "target_category", "")).encode("utf-8", errors="ignore"))
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
def load_metadata_for_app(
    dataset_key: str,
    source_types: tuple[str, ...],
    refresh_token: str,
) -> tuple[pd.DataFrame, str]:
    _ = refresh_token
    metadata, metadata_path = load_search_metadata(dataset_key, source_types or None)
    return metadata, str(metadata_path)


@st.cache_data
def load_queries() -> pd.DataFrame:
    from src.config import DEFAULT_QUERYSET_CSV

    if DEFAULT_QUERYSET_CSV.exists():
        return pd.read_csv(DEFAULT_QUERYSET_CSV)
    return pd.DataFrame(columns=["query_id", "query", "target_category"])


@st.cache_data
def load_pca_variance(model_alias: str, text_source: str, artifact_namespace: str) -> dict:
    from src.config import PLOTS_DIR

    return load_json(PLOTS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_pca_variance.json")


@st.cache_data
def load_cluster_frame(model_alias: str, text_source: str, method: str, artifact_namespace: str) -> pd.DataFrame:
    from src.visualize.clustering import load_cluster_frame as _load_cluster_frame

    return _load_cluster_frame(model_alias, text_source=text_source, method=method, artifact_namespace=artifact_namespace)


@st.cache_data
def load_cluster_summary(model_alias: str, text_source: str, method: str, artifact_namespace: str) -> dict:
    from src.visualize.clustering import load_cluster_summary as _load_cluster_summary

    return _load_cluster_summary(
        model_alias,
        text_source=text_source,
        method=method,
        artifact_namespace=artifact_namespace,
    )


@st.cache_data
def load_eval_summary(artifact_namespace: str, eval_token: str) -> pd.DataFrame:
    _ = eval_token
    return pd.read_csv(evaluation_artifact_path("retrieval_eval_summary.csv", artifact_namespace))


@st.cache_data
def load_eval_detail(artifact_namespace: str, eval_token: str) -> pd.DataFrame:
    _ = eval_token
    return pd.read_csv(evaluation_artifact_path("retrieval_eval_detail.csv", artifact_namespace))


@st.cache_data
def load_eval_comparison(artifact_namespace: str, eval_token: str) -> pd.DataFrame:
    _ = eval_token
    return pd.read_csv(evaluation_artifact_path("retrieval_eval_source_comparison.csv", artifact_namespace))


@st.cache_resource
def get_keyword_engine(
    dataset_key: str,
    source_types: tuple[str, ...],
    text_source: str,
    metadata_token: str,
) -> KeywordSearchEngine:
    _ = metadata_token
    metadata, _ = load_search_metadata(dataset_key, source_types or None)
    return KeywordSearchEngine(metadata, text_source=text_source)


@st.cache_resource
def get_dense_engine(
    dataset_key: str,
    source_types: tuple[str, ...],
    model_alias: str,
    text_source: str,
    artifact_namespace: str,
    metadata_token: str,
) -> DenseSearchEngine:
    _ = metadata_token
    metadata, _ = load_search_metadata(dataset_key, source_types or None)
    engine = DenseSearchEngine(
        metadata,
        model_alias,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
    )
    engine.load()
    return engine


def search_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ["raw_score", "normalized_score", "similarity_score", "best_match_similarity"]:
        if column in display.columns:
            display[column] = display[column].astype(float).round(4)
    if "best_match_text" in display.columns:
        display["best_match_text"] = display["best_match_text"].astype(str).str.slice(0, 140)
    return display


def ensure_artifacts(
    metadata: pd.DataFrame,
    artifact_namespace: str,
    text_source: str,
    model_aliases: list[str],
    cluster_method: str,
    n_clusters: int = 6,
) -> None:
    for text_source_item in [text_source]:
        KeywordSearchEngine(metadata, text_source=text_source_item).export_index_metadata(
            artifact_namespace=artifact_namespace
        )
        for model_alias in model_aliases:
            dense_engine = DenseSearchEngine(
                metadata,
                model_alias,
                text_source=text_source_item,
                artifact_namespace=artifact_namespace,
            )
            dense_engine.load()

            pca_path = projection_artifact_path(
                model_alias,
                text_source_item,
                "pca",
                3,
                artifact_namespace=artifact_namespace,
            )
            tsne_2d_path = projection_artifact_path(
                model_alias,
                text_source_item,
                "tsne",
                2,
                artifact_namespace=artifact_namespace,
            )
            tsne_3d_path = projection_artifact_path(
                model_alias,
                text_source_item,
                "tsne",
                3,
                artifact_namespace=artifact_namespace,
            )
            projection_is_stale = (
                not artifact_ids_match(pca_path, metadata)
                or (tsne_2d_path.exists() and not artifact_ids_match(tsne_2d_path, metadata))
                or (tsne_3d_path.exists() and not artifact_ids_match(tsne_3d_path, metadata))
            )
            if projection_is_stale:
                build_projection_artifacts(
                    metadata,
                    model_alias,
                    text_source=text_source_item,
                    optional_methods=("tsne",),
                    artifact_namespace=artifact_namespace,
                )

            cluster_csv_path = CLUSTERS_DIR / (
                f"{artifact_stem(model_alias, text_source_item, artifact_namespace)}_{cluster_method}_clusters.csv"
            )
            if not artifact_ids_match(cluster_csv_path, metadata):
                try:
                    cluster_embeddings(
                        metadata,
                        model_alias,
                        method=cluster_method,
                        n_clusters=n_clusters,
                        text_source=text_source_item,
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


def ensure_evaluation_artifacts(
    metadata: pd.DataFrame,
    queryset: pd.DataFrame,
    artifact_namespace: str,
) -> None:
    required_paths = [
        evaluation_artifact_path("retrieval_eval_summary.csv", artifact_namespace),
        evaluation_artifact_path("retrieval_eval_detail.csv", artifact_namespace),
        evaluation_artifact_path("retrieval_eval_source_comparison.csv", artifact_namespace),
    ]
    if all(path.exists() for path in required_paths):
        return

    evaluate_all(
        metadata=metadata,
        queryset=queryset,
        text_sources=evaluation_text_sources(metadata),
        include_optional=False,
        artifact_namespace=artifact_namespace,
    )


def available_plot_mode_labels(model_aliases: list[str], text_source: str, artifact_namespace: str) -> list[str]:
    labels = []
    for label, (method, dimensions) in PLOT_MODE_OPTIONS.items():
        if all(
            projection_artifact_path(
                model_alias,
                text_source,
                method,
                dimensions,
                artifact_namespace=artifact_namespace,
            ).exists()
            for model_alias in model_aliases
        ):
            labels.append(label)
    return labels or ["PCA 3D"]


def build_plot_frame(
    model_alias: str,
    text_source: str,
    method: str,
    dimensions: int,
    score_frame: pd.DataFrame,
    cluster_method: str,
    artifact_namespace: str,
) -> pd.DataFrame:
    projection = load_projection_frame(
        model_alias,
        text_source,
        method,
        dimensions,
        artifact_namespace=artifact_namespace,
    )
    cluster_frame = load_cluster_frame(model_alias, text_source, cluster_method, artifact_namespace)[["id", "cluster_id"]]
    merged = projection.merge(cluster_frame, on="id", how="left")
    merged = merged.merge(score_frame[["id", "raw_score", "normalized_score"]], on="id", how="left")
    merged["preview"] = merged["stt_transcript"].where(
        merged["stt_transcript"].astype(str).str.len() > 0,
        merged["original_transcript"],
    )
    merged["preview"] = merged["preview"].astype(str).str.slice(0, 160) + "..."
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
    plot_frame = build_plot_frame(
        model_alias,
        text_source,
        method,
        dimensions,
        score_frame,
        cluster_method,
        artifact_namespace,
    )
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
    if method == "tsne":
        st.caption("t-SNE는 query transform을 직접 지원하지 않아서 query 포인트는 표시되지 않을 수 있습니다.")


def render_pca_metrics(model_alias: str, text_source: str, artifact_namespace: str) -> None:
    variance = load_pca_variance(model_alias, text_source, artifact_namespace)
    ratios = variance["explained_variance_ratio"]
    metrics = st.columns(4)
    metrics[0].metric("PC1", f"{ratios[0]:.4f}")
    metrics[1].metric("PC2", f"{ratios[1]:.4f}")
    metrics[2].metric("PC3", f"{ratios[2]:.4f}")
    metrics[3].metric("PC1+PC2+PC3", f"{variance['cumulative_first3']:.4f}")
    st.dataframe(
        pd.DataFrame(
            {
                "component": ["PC1", "PC2", "PC3"],
                "explained_variance_ratio": [round(value, 4) for value in ratios[:3]],
            }
        ),
        use_container_width=True,
        height=160,
    )


def render_document_detail(metadata: pd.DataFrame, doc_id: str, text_source: str) -> None:
    selected = metadata.loc[metadata["id"] == doc_id]
    if selected.empty:
        st.info("선택한 문서를 찾지 못했습니다.")
        return

    row = selected.iloc[0]
    st.markdown(f"**문서 상세: {row['id']} / {row['title']}**")
    info = pd.DataFrame(
        [
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

    transcript_col, stt_col = st.columns(2)
    with transcript_col:
        st.markdown("**original_transcript**")
        st.text_area("원문", value=row["original_transcript"], height=260, key=f"orig_{doc_id}")
    with stt_col:
        st.markdown("**stt_transcript**")
        st.text_area("STT", value=row["stt_transcript"], height=260, key=f"stt_{doc_id}")

    st.markdown(f"**{text_source} 기준 라인 보기**")
    line_frame = pd.DataFrame(split_text_into_lines(resolve_primary_text(row, text_source=text_source)))
    if not line_frame.empty:
        st.dataframe(line_frame, use_container_width=True, height=260, hide_index=True)

    audio_path = Path(row["audio_path"]) if row["audio_path"] else None
    if audio_path and audio_path.exists():
        st.audio(str(audio_path))


def main() -> None:
    st.title("음성/STT 검색 비교 실험 앱")
    st.caption("기존 검색 비교, 문서 상세, 임베딩 투영, 클러스터, 평가 탭을 유지하면서 dummy / real youtube mp4 / combined 데이터셋을 전환할 수 있습니다.")

    dataset_options = available_dataset_options()
    option_keys = [key for key, _ in dataset_options]
    default_metadata = default_search_metadata_path()
    default_dataset_key = next(
        (key for key, path in dataset_options if path == default_metadata),
        option_keys[0],
    )

    with st.sidebar:
        st.header("앱 설정")
        dataset_key = st.selectbox("데이터셋", options=option_keys, index=option_keys.index(default_dataset_key))
        refresh_token = dataset_refresh_token(dataset_key)
        all_metadata, metadata_path_str = load_metadata_for_app(dataset_key, tuple(), refresh_token)
        source_type_options = sorted(all_metadata["source_type"].dropna().astype(str).unique().tolist())
        selected_source_types = st.multiselect(
            "source_type 필터",
            options=source_type_options,
            default=source_type_options,
        )
        source_type_tuple = tuple(selected_source_types)
        metadata, metadata_path_str = load_metadata_for_app(dataset_key, source_type_tuple, refresh_token)
        queryset = load_queries()
        example_queries = queryset["query"].tolist() if not queryset.empty else ["interview transcript"]
        selected_example = st.selectbox("예시 질의", options=example_queries, index=0)
        query = st.text_input("검색 질의", value=selected_example)
        search_source = st.selectbox(
            "검색 기준 텍스트",
            options=["stt_transcript", "original_transcript", "combined"],
            index=0,
        )
        top_k = st.slider("Top-K", min_value=3, max_value=15, value=5)
        keyword_method = st.selectbox("키워드 방식", options=["bm25", "tfidf"], index=0)
        model_aliases = list(list_available_models(include_optional=False).keys())
        model_a = st.selectbox("임베딩 모델 A", options=model_aliases, index=0)
        model_b = st.selectbox("임베딩 모델 B", options=model_aliases, index=min(1, len(model_aliases) - 1))
        artifact_namespace = dataset_artifact_namespace(Path(metadata_path_str), source_type_tuple or None)
        cluster_method = st.selectbox("클러스터 방식", options=["kmeans", "hdbscan"], index=0)
        with st.spinner("검색 및 시각화 아티팩트 확인 중..."):
            ensure_artifacts(
                metadata=metadata,
                artifact_namespace=artifact_namespace,
                text_source=search_source,
                model_aliases=[model_a, model_b],
                cluster_method=cluster_method,
                n_clusters=6,
            )
        plot_mode_labels = available_plot_mode_labels([model_a, model_b], search_source, artifact_namespace)
        vector_plot_mode_label = st.selectbox("벡터 분포 모드", options=plot_mode_labels, index=0)
        cluster_plot_mode_label = st.selectbox("클러스터 뷰 모드", options=plot_mode_labels, index=0)
        vector_color_by = st.selectbox("문서 분포 색상 기준", options=["category", "cluster_id", "source_type"], index=0)

    if metadata.empty:
        st.warning("선택한 데이터셋/필터에 해당하는 문서가 없습니다.")
        st.stop()

    vector_method, vector_dimensions = PLOT_MODE_OPTIONS[vector_plot_mode_label]
    cluster_view_method, cluster_view_dimensions = PLOT_MODE_OPTIONS[cluster_plot_mode_label]

    metadata_token = metadata_content_token(metadata, search_source)

    keyword_engine = get_keyword_engine(dataset_key, source_type_tuple, search_source, metadata_token)
    dense_engine_a = get_dense_engine(
        dataset_key,
        source_type_tuple,
        model_a,
        search_source,
        artifact_namespace,
        metadata_token,
    )
    dense_engine_b = get_dense_engine(
        dataset_key,
        source_type_tuple,
        model_b,
        search_source,
        artifact_namespace,
        metadata_token,
    )

    keyword_results = keyword_engine.search(query, top_k=top_k, method=keyword_method)
    dense_results_a = dense_engine_a.search(query, top_k=top_k)
    dense_results_b = dense_engine_b.search(query, top_k=top_k)
    dense_scores_a = dense_engine_a.score_all(query)
    dense_scores_b = dense_engine_b.score_all(query)

    tabs = st.tabs(["검색 비교", "문서 비교", "벡터 분포", "클러스터", "평가", "데이터셋"])
    tab_search, tab_detail, tab_vectors, tab_clusters, tab_eval, tab_data = tabs

    with tab_search:
        st.subheader("검색 결과 비교")
        st.caption(f"metadata: `{metadata_path_str}` / namespace: `{artifact_namespace}`")
        left, center, right = st.columns(3)
        with left:
            st.markdown(f"**키워드 검색 / {keyword_method.upper()} / {search_source}**")
            st.dataframe(search_table(keyword_results), use_container_width=True, height=420)
        with center:
            st.markdown(f"**임베딩 검색 / {model_a} / {search_source}**")
            st.dataframe(search_table(dense_results_a), use_container_width=True, height=420)
        with right:
            st.markdown(f"**임베딩 검색 / {model_b} / {search_source}**")
            st.dataframe(search_table(dense_results_b), use_container_width=True, height=420)

    with tab_detail:
        st.subheader("문서 상세")
        candidate_ids = pd.unique(
            pd.concat([keyword_results["id"], dense_results_a["id"], dense_results_b["id"]], ignore_index=True)
        ).tolist()
        selected_doc_id = st.selectbox("상세 비교 문서 ID", options=candidate_ids or metadata["id"].tolist(), index=0)
        render_document_detail(metadata, selected_doc_id, search_source)

    with tab_vectors:
        st.subheader("벡터 분포")
        for model_alias, engine, score_frame, results in [
            (model_a, dense_engine_a, dense_scores_a, dense_results_a),
            (model_b, dense_engine_b, dense_scores_b, dense_results_b),
        ]:
            st.markdown(f"**{model_alias} / {search_source} / {vector_plot_mode_label}**")
            if vector_method == "pca":
                render_pca_metrics(model_alias, search_source, artifact_namespace)
            try:
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
            except Exception as exc:
                st.warning(f"{model_alias} / {search_source} / {vector_plot_mode_label} 표시 실패: {exc}")

    with tab_clusters:
        st.subheader("클러스터 구조")
        for model_alias, engine, score_frame, results in [
            (model_a, dense_engine_a, dense_scores_a, dense_results_a),
            (model_b, dense_engine_b, dense_scores_b, dense_results_b),
        ]:
            st.markdown(f"**{model_alias} / {search_source} / {cluster_plot_mode_label} / {cluster_method}**")
            try:
                summary = load_cluster_summary(model_alias, search_source, cluster_method, artifact_namespace)
                metric_cols = st.columns(3)
                metric_cols[0].metric("요청 클러스터 수", summary.get("n_clusters_requested", "-"))
                metric_cols[1].metric("실제 클러스터 수", summary.get("n_clusters_found", "-"))
                silhouette_value = summary.get("silhouette_score")
                metric_cols[2].metric(
                    "Silhouette",
                    "-" if silhouette_value is None or pd.isna(silhouette_value) else f"{float(silhouette_value):.4f}",
                )
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
                st.markdown("**대표 샘플**")
                st.dataframe(
                    load_representatives(
                        model_alias,
                        text_source=search_source,
                        method=cluster_method,
                        artifact_namespace=artifact_namespace,
                    ),
                    use_container_width=True,
                    height=220,
                )
            except Exception as exc:
                st.warning(f"{model_alias} / {search_source} / {cluster_plot_mode_label} / {cluster_method} 표시 실패: {exc}")

    with tab_eval:
        st.subheader("평가 결과")
        with st.spinner("평가 아티팩트 확인 중..."):
            ensure_evaluation_artifacts(metadata, queryset, artifact_namespace)
        eval_token = evaluation_content_token(metadata, queryset)
        try:
            eval_summary = load_eval_summary(artifact_namespace, eval_token)
            eval_comparison = load_eval_comparison(artifact_namespace, eval_token)
            eval_detail = load_eval_detail(artifact_namespace, eval_token)
            if eval_summary.empty:
                st.info("현재 필터에 맞는 평가 질의가 없어 평가 결과가 비어 있습니다.")
            st.markdown("**시스템 요약**")
            st.dataframe(eval_summary.round(4), use_container_width=True)
            st.markdown("**원문 vs STT 비교**")
            st.dataframe(eval_comparison.round(4), use_container_width=True)
            st.markdown("**질의별 상세 결과**")
            st.dataframe(eval_detail, use_container_width=True, height=380)
        except Exception as exc:
            st.warning(f"현재 데이터셋 namespace `{artifact_namespace}`에 대한 평가 아티팩트가 없습니다: {exc}")

    with tab_data:
        st.subheader("데이터셋 미리보기")
        st.write(f"총 문서 수: {len(metadata)}")
        st.write(f"metadata path: `{metadata_path_str}`")
        st.write(f"artifact namespace: `{artifact_namespace}`")
        st.dataframe(metadata.head(20), use_container_width=True, height=360)
        category_counts = metadata["category"].value_counts().reset_index()
        category_counts.columns = ["category", "count"]
        fig = px.bar(category_counts, x="category", y="count", title="카테고리 분포")
        st.plotly_chart(fig, use_container_width=True)
        if not queryset.empty:
            st.markdown("**평가용 질의**")
            st.dataframe(queryset, use_container_width=True)


if __name__ == "__main__":
    main()
