from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_METADATA_CSV, EMBEDDINGS_DIR, INDICES_DIR, ensure_project_dirs
from src.data.metadata_schema import ensure_metadata_columns, load_metadata_frame
from src.embedding.vector_models import EmbeddingModelWrapper, list_available_models
from src.search.match_locator import locate_best_dense_match
from src.search.text_source import (
    DEFAULT_TEXT_SOURCE,
    build_preview_text,
    build_search_text,
    resolve_primary_text,
    text_source_suffix,
)
from src.utils.io_utils import load_json, save_json, save_numpy

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def artifact_stem(model_alias: str, text_source: str) -> str:
    return f"{model_alias}__{text_source_suffix(text_source)}"


class DenseSearchEngine:
    def __init__(self, metadata: pd.DataFrame, model_alias: str, text_source: str = DEFAULT_TEXT_SOURCE):
        self.metadata = ensure_metadata_columns(metadata)
        self.model_alias = model_alias
        self.text_source = text_source
        self._artifact_stem = artifact_stem(model_alias, text_source)
        self.embedding_path = EMBEDDINGS_DIR / f"{self._artifact_stem}_embeddings.npy"
        self.doc_meta_path = EMBEDDINGS_DIR / f"{self._artifact_stem}_metadata.csv"
        self.index_path = INDICES_DIR / f"{self._artifact_stem}.faiss"
        self.index_summary_path = INDICES_DIR / f"{self._artifact_stem}_index_summary.json"
        self.wrapper = EmbeddingModelWrapper(model_alias)
        self.metadata["primary_text"] = self.metadata.apply(
            lambda row: resolve_primary_text(row, text_source=self.text_source),
            axis=1,
        )
        self.metadata["search_text"] = self.metadata.apply(
            lambda row: build_search_text(row, text_source=self.text_source),
            axis=1,
        )
        self.document_texts = self.metadata["search_text"].tolist()
        self.embeddings: np.ndarray | None = None
        self.index = None

    def build(self) -> None:
        ensure_project_dirs()
        self.embeddings = self.wrapper.encode_documents(self.document_texts)
        save_numpy(self.embedding_path, self.embeddings)
        self.metadata.to_csv(self.doc_meta_path, index=False, encoding="utf-8-sig")

        index_saved = False
        if faiss is not None:
            index = faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(self.embeddings)
            faiss.write_index(index, str(self.index_path))
            self.index = index
            index_saved = True

        save_json(
            self.index_summary_path,
            {
                "model_alias": self.model_alias,
                "model_name": self.wrapper.model_name,
                "text_source": self.text_source,
                "embedding_path": str(self.embedding_path.resolve()),
                "metadata_path": str(self.doc_meta_path.resolve()),
                "faiss_index_path": str(self.index_path.resolve()) if index_saved else None,
                "dimension": int(self.embeddings.shape[1]),
                "document_count": int(self.embeddings.shape[0]),
            },
        )

    def load(self) -> None:
        if not self.embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")
        self.embeddings = np.load(self.embedding_path)
        if faiss is not None and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))

    def encode_query(self, query: str) -> np.ndarray:
        return self.wrapper.encode_queries([query])[0]

    def score_all(self, query: str) -> pd.DataFrame:
        if self.embeddings is None:
            self.load()
        assert self.embeddings is not None

        query_embedding = self.encode_query(query)
        raw_scores = self.embeddings @ query_embedding
        normalized_scores = _normalize_scores(raw_scores)

        result_frame = self.metadata.copy()
        result_frame["raw_score"] = raw_scores
        result_frame["normalized_score"] = normalized_scores
        result_frame["similarity_score"] = normalized_scores
        result_frame["search_source"] = self.text_source
        result_frame["preview"] = result_frame.apply(
            lambda row: build_preview_text(row, text_source=self.text_source),
            axis=1,
        )
        result_frame["original_preview"] = result_frame["original_transcript"].str.slice(0, 140) + "..."
        result_frame["stt_preview"] = result_frame["stt_transcript"].str.slice(0, 140) + "..."
        return result_frame.sort_values("raw_score", ascending=False).reset_index(drop=True)

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        result_frame = self.score_all(query).head(top_k).reset_index(drop=True)
        match_details = result_frame["primary_text"].apply(
            lambda text: locate_best_dense_match(text, query, self.wrapper)
        )
        result_frame = pd.concat([result_frame, pd.DataFrame(match_details.tolist())], axis=1)
        result_frame["best_match_summary"] = result_frame.apply(
            lambda row: f"{row['id']}의 {row['best_match_location']}" if row["best_match_location"] else str(row["id"]),
            axis=1,
        )
        result_frame.insert(0, "rank", result_frame.index + 1)
        return result_frame[
            [
                "rank",
                "id",
                "file_name",
                "file_path",
                "processed_txt_path",
                "audio_file_path",
                "stt_txt_path",
                "category",
                "raw_score",
                "normalized_score",
                "similarity_score",
                "best_match_summary",
                "best_match_location",
                "best_match_similarity",
                "best_match_text",
                "search_source",
                "preview",
                "original_preview",
                "stt_preview",
            ]
        ]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_score = float(scores.min())
    max_score = float(scores.max())
    if abs(max_score - min_score) < 1e-12:
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - min_score) / (max_score - min_score)).astype(np.float32)


def build_all_indices(
    metadata_path: Path,
    include_optional: bool = False,
    text_sources: list[str] | tuple[str, ...] = (DEFAULT_TEXT_SOURCE,),
) -> list[tuple[str, str]]:
    ensure_project_dirs()
    metadata = load_metadata_frame(metadata_path)
    model_aliases = list(list_available_models(include_optional=include_optional).keys())
    built: list[tuple[str, str]] = []
    for text_source in text_sources:
        for alias in model_aliases:
            engine = DenseSearchEngine(metadata, alias, text_source=text_source)
            engine.build()
            built.append((alias, text_source))
    return built


def load_index_summary(model_alias: str, text_source: str = DEFAULT_TEXT_SOURCE) -> dict:
    return load_json(INDICES_DIR / f"{artifact_stem(model_alias, text_source)}_index_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense embedding indices.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument(
        "--text-sources",
        nargs="+",
        default=[DEFAULT_TEXT_SOURCE, "original_transcript"],
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    args = parser.parse_args()

    built = build_all_indices(
        args.metadata_path,
        include_optional=args.include_optional,
        text_sources=args.text_sources,
    )
    print("Built embedding artifacts for:")
    for alias, text_source in built:
        print(f" - {alias} / {text_source}")


if __name__ == "__main__":
    main()
