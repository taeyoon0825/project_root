from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CLUSTERS_DIR, DEFAULT_METADATA_CSV, DEFAULT_QUERYSET_CSV, EVALUATION_DIR
from src.data.metadata_schema import load_metadata_frame
from src.embedding.build_indices import DenseSearchEngine
from src.embedding.vector_models import list_available_models
from src.search.keyword_search import KeywordSearchEngine
from src.search.text_source import resolve_primary_text, split_text_into_lines
from src.utils.io_utils import load_json
from src.visualize.clustering import load_representatives
from src.visualize.interactive_plot import (
    build_projection_figure,
    load_projection_frame,
    project_query_vector,
    projection_artifact_path,
    projection_columns,
)


PLOT_MODE_OPTIONS = {
    "PCA 3D": ("pca", 3),
    "PCA 2D": ("pca", 2),
    "UMAP 3D": ("umap", 3),
    "UMAP 2D": ("umap", 2),
    "t-SNE 3D": ("tsne", 3),
    "t-SNE 2D": ("tsne", 2),
}


st.set_page_config(page_title="Audio STT Retrieval Experiment", layout="wide")


@st.cache_data
def load_metadata() -> pd.DataFrame:
    return load_metadata_frame(DEFAULT_METADATA_CSV)


@st.cache_data
def load_queries() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_QUERYSET_CSV)


@st.cache_data
def load_pca_variance(model_alias: str, text_source: str) -> dict:
    from src.config import PLOTS_DIR
    from src.embedding.build_indices import artifact_stem

    return load_json(PLOTS_DIR / f"{artifact_stem(model_alias, text_source)}_pca_variance.json")


@st.cache_data
def load_cluster_frame(model_alias: str, text_source: str, method: str = "kmeans") -> pd.DataFrame:
    from src.embedding.build_indices import artifact_stem

    return pd.read_csv(CLUSTERS_DIR / f"{artifact_stem(model_alias, text_source)}_{method}_clusters.csv")


@st.cache_data
def load_cluster_summary(model_alias: str, text_source: str, method: str = "kmeans") -> dict:
    from src.embedding.build_indices import artifact_stem

    return load_json(CLUSTERS_DIR / f"{artifact_stem(model_alias, text_source)}_{method}_summary.json")


@st.cache_data
def load_eval_summary() -> pd.DataFrame:
    return pd.read_csv(EVALUATION_DIR / "retrieval_eval_summary.csv")


@st.cache_data
def load_eval_detail() -> pd.DataFrame:
    return pd.read_csv(EVALUATION_DIR / "retrieval_eval_detail.csv")


@st.cache_data
def load_eval_comparison() -> pd.DataFrame:
    return pd.read_csv(EVALUATION_DIR / "retrieval_eval_source_comparison.csv")


@st.cache_resource
def get_keyword_engine(text_source: str) -> KeywordSearchEngine:
    return KeywordSearchEngine(load_metadata(), text_source=text_source)


@st.cache_resource
def get_dense_engine(model_alias: str, text_source: str) -> DenseSearchEngine:
    engine = DenseSearchEngine(load_metadata(), model_alias, text_source=text_source)
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


def available_plot_mode_labels(model_aliases: list[str], text_source: str) -> list[str]:
    labels = []
    for label, (method, dimensions) in PLOT_MODE_OPTIONS.items():
        if all(projection_artifact_path(model_alias, text_source, method, dimensions).exists() for model_alias in model_aliases):
            labels.append(label)
    return labels or ["PCA 3D"]


def build_plot_frame(
    model_alias: str,
    text_source: str,
    method: str,
    dimensions: int,
    score_frame: pd.DataFrame,
    cluster_method: str,
) -> pd.DataFrame:
    projection = load_projection_frame(model_alias, text_source, method, dimensions)
    cluster_frame = load_cluster_frame(model_alias, text_source, cluster_method)[["id", "cluster_id"]]
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
) -> None:
    plot_frame = build_plot_frame(model_alias, text_source, method, dimensions, score_frame, cluster_method)
    query_coords = project_query_vector(
        model_alias,
        text_source,
        method,
        dimensions,
        dense_engine.encode_query(query),
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
        st.caption("t-SNE는 안정적인 query transform을 제공하지 않아 query 점 오버레이가 생략될 수 있습니다.")


def render_pca_metrics(model_alias: str, text_source: str) -> None:
    variance = load_pca_variance(model_alias, text_source)
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
            {"field": "category", "value": row["category"]},
            {"field": "file_name", "value": row["file_name"]},
            {"field": "file_path", "value": row["file_path"]},
            {"field": "processed_txt_path", "value": row["processed_txt_path"]},
            {"field": "audio_file_path", "value": row["audio_file_path"]},
            {"field": "stt_txt_path", "value": row["stt_txt_path"]},
            {"field": "keywords", "value": row["keywords"]},
            {"field": "tts_provider", "value": row["tts_provider"]},
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

    st.markdown(f"**{text_source} 기준 줄 단위 보기**")
    line_frame = pd.DataFrame(split_text_into_lines(resolve_primary_text(row, text_source=text_source)))
    if not line_frame.empty:
        st.dataframe(line_frame, use_container_width=True, height=260, hide_index=True)

    audio_path = Path(row["audio_file_path"]) if row["audio_file_path"] else None
    if audio_path and audio_path.exists():
        st.audio(str(audio_path))


def main() -> None:
    st.title("음성 → STT → 검색 비교 실험 앱")
    st.caption("검색 결과와 함께 2D/3D 분포, 군집 구조, query 위치, top-k 문서 위치를 함께 해석할 수 있도록 확장한 UI다.")

    if not DEFAULT_METADATA_CSV.exists():
        st.error("메타데이터가 없습니다. 먼저 `python run_audio_experiment_pipeline.py`를 실행하세요.")
        st.stop()

    metadata = load_metadata()
    queryset = load_queries()
    model_aliases = list(list_available_models(include_optional=False).keys())

    with st.sidebar:
        st.header("실험 설정")
        selected_example = st.selectbox("예시 질의", options=queryset["query"].tolist(), index=0)
        query = st.text_input("검색 질의", value=selected_example or "봄에 회의한 기획안")
        search_source = st.selectbox(
            "검색 기준 텍스트",
            options=["stt_transcript", "original_transcript", "combined"],
            index=0,
        )
        top_k = st.slider("Top-K", min_value=3, max_value=15, value=5)
        keyword_method = st.selectbox("키워드 방식", options=["bm25", "tfidf"], index=0)
        model_a = st.selectbox("임베딩 모델 A", options=model_aliases, index=0)
        model_b = st.selectbox("임베딩 모델 B", options=model_aliases, index=min(1, len(model_aliases) - 1))
        plot_mode_labels = available_plot_mode_labels([model_a, model_b], search_source)
        vector_plot_mode_label = st.selectbox("벡터 분포 모드", options=plot_mode_labels, index=0)
        cluster_plot_mode_label = st.selectbox("군집화 모드", options=plot_mode_labels, index=0)
        cluster_method = st.selectbox("군집 방식", options=["kmeans", "hdbscan"], index=0)
        vector_color_by = st.selectbox("문서 분포 색상 기준", options=["category", "cluster_id"], index=0)

    vector_method, vector_dimensions = PLOT_MODE_OPTIONS[vector_plot_mode_label]
    cluster_view_method, cluster_view_dimensions = PLOT_MODE_OPTIONS[cluster_plot_mode_label]

    keyword_engine = get_keyword_engine(search_source)
    dense_engine_a = get_dense_engine(model_a, search_source)
    dense_engine_b = get_dense_engine(model_b, search_source)

    keyword_results = keyword_engine.search(query, top_k=top_k, method=keyword_method)
    dense_results_a = dense_engine_a.search(query, top_k=top_k)
    dense_results_b = dense_engine_b.search(query, top_k=top_k)
    dense_scores_a = dense_engine_a.score_all(query)
    dense_scores_b = dense_engine_b.score_all(query)

    tabs = st.tabs(["검색 비교", "문서 비교", "벡터 분포", "군집화", "평가", "데이터셋"])
    tab_search, tab_detail, tab_vectors, tab_clusters, tab_eval, tab_data = tabs

    with tab_search:
        st.subheader("검색 결과 비교")
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
        st.subheader("원문 vs STT 비교")
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
                render_pca_metrics(model_alias, search_source)
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
                )
            except Exception as exc:
                st.warning(f"{model_alias} / {search_source} / {vector_plot_mode_label} 표시 실패: {exc}")

    with tab_clusters:
        st.subheader("군집 구조")
        for model_alias, engine, score_frame, results in [
            (model_a, dense_engine_a, dense_scores_a, dense_results_a),
            (model_b, dense_engine_b, dense_scores_b, dense_results_b),
        ]:
            st.markdown(f"**{model_alias} / {search_source} / {cluster_plot_mode_label} / {cluster_method}**")
            try:
                summary = load_cluster_summary(model_alias, search_source, cluster_method)
                metric_cols = st.columns(3)
                metric_cols[0].metric("요청 군집 수", summary.get("n_clusters_requested", "-"))
                metric_cols[1].metric("실제 군집 수", summary.get("n_clusters_found", "-"))
                metric_cols[2].metric(
                    "Silhouette",
                    "-" if summary.get("silhouette_score") is None else f"{summary.get('silhouette_score'):.4f}",
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
                )
                st.markdown("**대표 샘플**")
                st.dataframe(
                    load_representatives(model_alias, text_source=search_source, method=cluster_method),
                    use_container_width=True,
                    height=220,
                )
            except Exception as exc:
                st.warning(f"{model_alias} / {search_source} / {cluster_plot_mode_label} / {cluster_method} 표시 실패: {exc}")

    with tab_eval:
        st.subheader("평가 결과")
        try:
            st.markdown("**시스템별 요약**")
            st.dataframe(load_eval_summary().round(4), use_container_width=True)
            st.markdown("**원문 vs STT 비교**")
            st.dataframe(load_eval_comparison().round(4), use_container_width=True)
            st.markdown("**질의별 상세 결과**")
            st.dataframe(load_eval_detail(), use_container_width=True, height=380)
        except Exception as exc:
            st.warning(f"평가 결과를 불러오지 못했습니다: {exc}")

    with tab_data:
        st.subheader("데이터셋 미리보기")
        st.write(f"총 문서 수: {len(metadata)}")
        st.dataframe(metadata.head(20), use_container_width=True, height=360)
        category_counts = metadata["category"].value_counts().reset_index()
        category_counts.columns = ["category", "count"]
        fig = px.bar(category_counts, x="category", y="count", title="카테고리 분포")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**평가용 질의셋**")
        st.dataframe(queryset, use_container_width=True)


if __name__ == "__main__":
    main()
