from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.embedding.build_indices import DenseSearchEngine
from src.embedding.vector_models import list_available_models
from src.search.keyword_search import KeywordSearchEngine
from src.search.load_realdata_dataset import (
    available_dataset_options,
    dataset_artifact_namespace,
    default_search_metadata_path,
    load_search_metadata,
)


st.set_page_config(page_title="Real MP4 Search", layout="wide")


@st.cache_data
def load_metadata(dataset_key: str) -> tuple[pd.DataFrame, str]:
    frame, metadata_path = load_search_metadata(dataset_key)
    return frame, str(metadata_path)


@st.cache_resource
def get_keyword_engine(dataset_key: str, source_types: tuple[str, ...], text_source: str) -> KeywordSearchEngine:
    metadata, _ = load_search_metadata(dataset_key, source_types or None)
    return KeywordSearchEngine(metadata, text_source=text_source)


@st.cache_resource
def get_dense_engine(
    dataset_key: str,
    source_types: tuple[str, ...],
    model_alias: str,
    text_source: str,
) -> DenseSearchEngine:
    metadata, metadata_path = load_search_metadata(dataset_key, source_types or None)
    namespace = dataset_artifact_namespace(metadata_path, source_types or None)
    engine = DenseSearchEngine(
        metadata,
        model_alias,
        text_source=text_source,
        artifact_namespace=namespace,
    )
    if engine.embedding_path.exists():
        engine.load()
    else:
        engine.build()
    return engine


def search_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ["raw_score", "normalized_score", "similarity_score", "best_match_similarity"]:
        if column in display.columns:
            display[column] = display[column].astype(float).round(4)
    keep_columns = [
        "rank",
        "id",
        "source_type",
        "file_name",
        "file_path",
        "audio_path",
        "processed_txt_path",
        "similarity_score",
        "best_match_location",
        "transcript_preview",
    ]
    available_columns = [column for column in keep_columns if column in display.columns]
    return display[available_columns]


def render_detail(metadata: pd.DataFrame, doc_id: str) -> None:
    selected = metadata.loc[metadata["id"] == doc_id]
    if selected.empty:
        st.info("Selected document was not found in the current dataset.")
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
            {"field": "stt_model_name", "value": row["stt_model_name"]},
            {"field": "processing_status", "value": row["processing_status"]},
        ]
    )
    st.dataframe(info, use_container_width=True, hide_index=True)

    transcript_col, transcript_file_col = st.columns(2)
    with transcript_col:
        st.markdown("**Transcript**")
        transcript_value = row["stt_transcript"] or row["original_transcript"]
        st.text_area("Transcript text", value=transcript_value, height=320)
    with transcript_file_col:
        st.markdown("**Transcript File Preview**")
        transcript_path = Path(str(row["processed_txt_path"])) if str(row["processed_txt_path"]).strip() else None
        if transcript_path and transcript_path.exists():
            st.text_area("Transcript txt", value=transcript_path.read_text(encoding="utf-8"), height=320)
        else:
            st.info("Transcript txt file does not exist yet.")

    audio_path = Path(str(row["audio_path"])) if str(row["audio_path"]).strip() else None
    if audio_path and audio_path.exists():
        st.audio(str(audio_path))


def main() -> None:
    st.title("MP4 -> STT -> Search")
    st.caption("Reuse the existing keyword and embedding retrieval stack against dummy, real youtube mp4, or combined metadata.")

    dataset_options = available_dataset_options()
    option_keys = [key for key, _ in dataset_options]
    default_path = default_search_metadata_path()
    default_key = next(
        (key for key, path in dataset_options if Path(path) == default_path),
        option_keys[0],
    )

    with st.sidebar:
        dataset_key = st.selectbox("Dataset", options=option_keys, index=option_keys.index(default_key))
        metadata_frame, metadata_path = load_metadata(dataset_key)
        source_types = sorted(metadata_frame["source_type"].dropna().astype(str).unique().tolist())
        selected_source_types = st.multiselect(
            "source_type filter",
            options=source_types,
            default=source_types,
        )
        text_source = st.selectbox(
            "Text source",
            options=["stt_transcript", "original_transcript", "combined"],
            index=0,
        )
        keyword_method = st.selectbox("Keyword method", options=["bm25", "tfidf"], index=0)
        model_alias = st.selectbox(
            "Dense model",
            options=list(list_available_models(include_optional=False).keys()),
            index=0,
        )
        top_k = st.slider("Top-K", min_value=3, max_value=20, value=10)
        query = st.text_input("Query", value="interview discussion transcript")

    filtered_metadata = metadata_frame.loc[
        metadata_frame["source_type"].isin(selected_source_types or source_types)
    ].reset_index(drop=True)

    st.write(f"Metadata path: `{metadata_path}`")
    st.write(f"Rows in current view: `{len(filtered_metadata)}`")
    if filtered_metadata.empty:
        st.warning("No rows match the current dataset/source_type selection.")
        st.stop()

    source_tuple = tuple(selected_source_types or source_types)

    with st.spinner("Preparing keyword and dense search engines..."):
        keyword_engine = get_keyword_engine(dataset_key, source_tuple, text_source)
        dense_engine = get_dense_engine(dataset_key, source_tuple, model_alias, text_source)

    keyword_results = keyword_engine.search(query, top_k=top_k, method=keyword_method)
    dense_results = dense_engine.search(query, top_k=top_k)

    result_tab, detail_tab, data_tab = st.tabs(["Search Results", "Document Detail", "Dataset Preview"])

    with result_tab:
        left, right = st.columns(2)
        with left:
            st.subheader(f"Keyword Search ({keyword_method.upper()})")
            st.dataframe(search_table(keyword_results), use_container_width=True, height=420)
        with right:
            st.subheader(f"Dense Search ({model_alias})")
            st.dataframe(search_table(dense_results), use_container_width=True, height=420)

    with detail_tab:
        candidate_ids = pd.unique(
            pd.concat([keyword_results["id"], dense_results["id"]], ignore_index=True)
        ).tolist()
        if not candidate_ids:
            st.info("No search results to inspect.")
        else:
            selected_doc_id = st.selectbox(
                "Document ID",
                options=candidate_ids,
                index=0,
            )
            render_detail(filtered_metadata, selected_doc_id)

    with data_tab:
        preview_columns = [
            "id",
            "source_type",
            "file_name",
            "file_path",
            "audio_path",
            "processed_txt_path",
            "category",
            "keywords",
            "processing_status",
        ]
        st.dataframe(filtered_metadata[preview_columns], use_container_width=True, height=420)


if __name__ == "__main__":
    main()
