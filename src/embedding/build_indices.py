from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_METADATA_CSV, EMBEDDINGS_DIR, INDICES_DIR, ensure_project_dirs
from src.data.metadata_schema import ensure_metadata_columns, load_metadata_frame
from src.embedding.vector_models import EmbeddingModelWrapper, list_available_models
from src.search.explainability import SEMANTIC_SCORE_WEIGHT, explain_frame
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


def artifact_stem(model_alias: str, text_source: str, artifact_namespace: str | None = None) -> str:
    base = f"{model_alias}__{text_source_suffix(text_source)}"
    if not artifact_namespace:
        return base
    safe_namespace = re.sub(r"[^0-9A-Za-z._-]+", "_", artifact_namespace).strip("._")
    return f"{safe_namespace}__{base}" if safe_namespace else base


class DenseSearchEngine:
    def __init__(
        self,
        metadata: pd.DataFrame,
        model_alias: str,
        text_source: str = DEFAULT_TEXT_SOURCE,
        artifact_namespace: str | None = None,
    ):
        self.model_alias = model_alias
        self.text_source = text_source
        self.artifact_namespace = artifact_namespace
        self._artifact_stem = artifact_stem(model_alias, text_source, artifact_namespace)
        self.embedding_path = EMBEDDINGS_DIR / f"{self._artifact_stem}_embeddings.npy"
        self.doc_meta_path = EMBEDDINGS_DIR / f"{self._artifact_stem}_metadata.csv"
        self.index_path = INDICES_DIR / f"{self._artifact_stem}.faiss"
        self.index_summary_path = INDICES_DIR / f"{self._artifact_stem}_index_summary.json"
        self.wrapper = EmbeddingModelWrapper(model_alias)
        self._set_metadata(metadata)
        self.embeddings: np.ndarray | None = None
        self.index = None

    def _set_metadata(self, metadata: pd.DataFrame) -> None:
        frame = ensure_metadata_columns(metadata).reset_index(drop=True)
        frame["primary_text"] = frame.apply(
            lambda row: resolve_primary_text(row, text_source=self.text_source),
            axis=1,
        )
        frame["search_text"] = frame.apply(
            lambda row: build_search_text(row, text_source=self.text_source),
            axis=1,
        )
        self.metadata = frame
        self.document_texts = self.metadata["search_text"].astype(str).tolist()

    def _artifact_alignment_status(
        self,
        embeddings: np.ndarray,
        stored_metadata: pd.DataFrame | None,
    ) -> tuple[bool, str]:
        current = self.metadata.reset_index(drop=True)
        if len(embeddings) != len(current):
            return False, f"embedding rows={len(embeddings)} / metadata rows={len(current)}"
        if stored_metadata is None:
            return False, "saved metadata file is missing"

        stored = ensure_metadata_columns(stored_metadata).reset_index(drop=True)
        if len(stored) != len(current):
            return False, f"saved metadata rows={len(stored)} / current metadata rows={len(current)}"

        current_ids = current["id"].astype(str).tolist()
        stored_ids = stored["id"].astype(str).tolist()
        if stored_ids != current_ids:
            return False, "saved metadata ids do not match current metadata ids"

        if "search_text" in stored.columns:
            stored_search_text = stored["search_text"].fillna("").astype(str).tolist()
        else:
            stored_search_text = stored.apply(
                lambda row: build_search_text(row, text_source=self.text_source),
                axis=1,
            ).astype(str).tolist()
        current_search_text = current["search_text"].fillna("").astype(str).tolist()
        if stored_search_text != current_search_text:
            return False, "saved search text does not match current metadata search text"

        return True, ""

    def _load_stored_metadata(self) -> pd.DataFrame | None:
        if not self.doc_meta_path.exists():
            return None
        return load_metadata_frame(self.doc_meta_path)

    def _search_texts_for_frame(self, frame: pd.DataFrame) -> list[str]:
        normalized = ensure_metadata_columns(frame).reset_index(drop=True)
        if "search_text" in normalized.columns:
            return normalized["search_text"].fillna("").astype(str).tolist()
        return normalized.apply(
            lambda row: build_search_text(row, text_source=self.text_source),
            axis=1,
        ).astype(str).tolist()

    def _append_candidates(self, stored_metadata: pd.DataFrame) -> pd.DataFrame | None:
        stored = ensure_metadata_columns(stored_metadata).reset_index(drop=True)
        current = self.metadata.reset_index(drop=True)
        if len(current) < len(stored):
            return None

        stored_ids = stored["id"].astype(str).tolist()
        current_ids = current["id"].astype(str).tolist()
        if current_ids[: len(stored_ids)] != stored_ids:
            return None

        stored_search_texts = self._search_texts_for_frame(stored)
        current_prefix_search_texts = current.iloc[: len(stored)].apply(
            lambda row: build_search_text(row, text_source=self.text_source),
            axis=1,
        ).astype(str).tolist()
        if current_prefix_search_texts != stored_search_texts:
            return None

        appended = current.iloc[len(stored) :].copy().reset_index(drop=True)
        if appended.empty:
            return appended
        return appended

    def _save_index_summary(self, document_count: int, index_saved: bool, build_mode: str) -> None:
        save_json(
            self.index_summary_path,
            {
                "model_alias": self.model_alias,
                "model_name": self.wrapper.model_name,
                "text_source": self.text_source,
                "artifact_namespace": self.artifact_namespace,
                "embedding_path": str(self.embedding_path.resolve()),
                "metadata_path": str(self.doc_meta_path.resolve()),
                "faiss_index_path": str(self.index_path.resolve()) if index_saved else None,
                "dimension": int(self.embeddings.shape[1]) if self.embeddings is not None else None,
                "document_count": int(document_count),
                "build_mode": build_mode,
            },
        )

    def _write_full_artifacts(self, build_mode: str = "full_rebuild") -> None:
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
        self._save_index_summary(document_count=len(self.metadata), index_saved=index_saved, build_mode=build_mode)

    def _append_artifacts(
        self,
        stored_embeddings: np.ndarray,
        stored_metadata: pd.DataFrame,
        appended_metadata: pd.DataFrame,
    ) -> None:
        appended_texts = appended_metadata["search_text"].fillna("").astype(str).tolist()
        appended_embeddings = self.wrapper.encode_documents(appended_texts)
        self.embeddings = np.vstack([stored_embeddings, appended_embeddings]).astype(np.float32)
        save_numpy(self.embedding_path, self.embeddings)

        combined_metadata = pd.concat([stored_metadata, appended_metadata], ignore_index=True)
        combined_metadata.to_csv(self.doc_meta_path, index=False, encoding="utf-8-sig")
        self.metadata = combined_metadata.reset_index(drop=True)
        self.document_texts = self.metadata["search_text"].fillna("").astype(str).tolist()

        index_saved = False
        if faiss is not None:
            if self.index_path.exists():
                index = faiss.read_index(str(self.index_path))
                if getattr(index, "ntotal", 0) == int(len(stored_embeddings)):
                    index.add(appended_embeddings)
                else:
                    index = faiss.IndexFlatIP(self.embeddings.shape[1])
                    index.add(self.embeddings)
            else:
                index = faiss.IndexFlatIP(self.embeddings.shape[1])
                index.add(self.embeddings)
            faiss.write_index(index, str(self.index_path))
            self.index = index
            index_saved = True
        self._save_index_summary(
            document_count=len(self.metadata),
            index_saved=index_saved,
            build_mode=f"incremental_append:{len(appended_metadata)}",
        )

    def build(self, incremental: bool = True) -> None:
        ensure_project_dirs()
        self._set_metadata(self.metadata)
        if incremental and self.embedding_path.exists() and self.doc_meta_path.exists():
            stored_embeddings = np.load(self.embedding_path)
            stored_metadata = self._load_stored_metadata()
            if stored_metadata is not None:
                is_aligned, _ = self._artifact_alignment_status(stored_embeddings, stored_metadata)
                if is_aligned:
                    self.embeddings = stored_embeddings
                    if faiss is not None and self.index_path.exists():
                        self.index = faiss.read_index(str(self.index_path))
                    self._save_index_summary(document_count=len(self.metadata), index_saved=self.index is not None, build_mode="no_update")
                    return

                appended_metadata = self._append_candidates(stored_metadata)
                if appended_metadata is not None and not appended_metadata.empty:
                    self._append_artifacts(stored_embeddings, ensure_metadata_columns(stored_metadata), appended_metadata)
                    return

        self._write_full_artifacts(build_mode="full_rebuild")

    def load(self, rebuild_if_missing: bool = True, rebuild_if_mismatch: bool = True) -> None:
        if not self.embedding_path.exists():
            if rebuild_if_missing:
                self.build()
                return
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")

        embeddings = np.load(self.embedding_path)
        stored_metadata = load_metadata_frame(self.doc_meta_path) if self.doc_meta_path.exists() else None
        is_aligned, reason = self._artifact_alignment_status(embeddings, stored_metadata)
        if not is_aligned:
            if rebuild_if_mismatch:
                print(f"[DenseSearchEngine] Rebuilding stale artifacts for {self._artifact_stem}: {reason}")
                self.build()
                return
            raise ValueError(f"Embedding artifacts are stale for {self._artifact_stem}: {reason}")

        self.embeddings = embeddings
        if faiss is not None and self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if getattr(index, "ntotal", 0) != int(self.embeddings.shape[0]):
                if rebuild_if_mismatch:
                    print(
                        f"[DenseSearchEngine] Rebuilding stale FAISS index for {self._artifact_stem}: "
                        f"ntotal={getattr(index, 'ntotal', 0)} / embeddings={self.embeddings.shape[0]}"
                    )
                    self.build()
                    return
                raise ValueError(f"FAISS index is stale for {self._artifact_stem}")
            self.index = index

    def encode_query(self, query: str) -> np.ndarray:
        return self.wrapper.encode_queries([query])[0]

    def score_all(self, query: str) -> pd.DataFrame:
        if self.embeddings is None or len(self.metadata) != len(self.embeddings):
            self.load()
        assert self.embeddings is not None
        if len(self.metadata) != len(self.embeddings):
            raise ValueError(
                f"Embedding rows ({len(self.embeddings)}) do not match metadata rows ({len(self.metadata)}) "
                f"for namespace={self.artifact_namespace or 'default'}"
            )

        query_embedding = self.encode_query(query)
        semantic_raw_scores = self.embeddings @ query_embedding
        semantic_normalized_scores = np.clip((semantic_raw_scores + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

        result_frame = self.metadata.copy()
        result_frame["semantic_raw_score"] = semantic_raw_scores
        result_frame["semantic_normalized_score"] = semantic_normalized_scores
        result_frame = explain_frame(
            result_frame,
            query,
            text_source=self.text_source,
            semantic_scores=[score * SEMANTIC_SCORE_WEIGHT for score in semantic_normalized_scores],
            score_kind="cosine_similarity",
        )
        result_frame["raw_score"] = result_frame["final_score"].astype(float)
        result_frame["normalized_score"] = _normalize_scores(result_frame["final_score"].to_numpy())
        result_frame["display_score"] = result_frame["normalized_score"] * 100.0
        result_frame["similarity_score"] = result_frame["display_score"]
        result_frame["search_source"] = self.text_source
        result_frame["score_kind"] = "weighted_lexical_semantic"
        result_frame["raw_score_explanation"] = (
            "raw_score는 L2 정규화된 임베딩 간 내적이며 cosine similarity와 동일합니다. "
            "display_score는 cosine similarity를 0~100 범위로 선형 변환한 값입니다."
        )
        result_frame["raw_score_explanation"] = (
            "final_score = lexical_score + semantic_score. "
            "lexical_score = title*5 + tags*4 + description*3 + transcript*2. "
            f"semantic_score = cosine similarity normalized score*{SEMANTIC_SCORE_WEIGHT:.1f}."
        )
        result_frame["preview"] = result_frame.apply(
            lambda row: build_preview_text(row, text_source=self.text_source),
            axis=1,
        )
        result_frame["transcript_preview"] = result_frame["preview"]
        result_frame["original_preview"] = result_frame["original_transcript"].str.slice(0, 140) + "..."
        result_frame["stt_preview"] = result_frame["stt_transcript"].str.slice(0, 140) + "..."
        return result_frame.sort_values("final_score", ascending=False).reset_index(drop=True)

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        result_frame = self.score_all(query).head(top_k).reset_index(drop=True)
        match_details = result_frame["primary_text"].apply(
            lambda text: locate_best_dense_match(text, query, self.wrapper)
        )
        result_frame = pd.concat([result_frame, pd.DataFrame(match_details.tolist())], axis=1)
        result_frame["best_match_summary"] = result_frame.apply(
            lambda row: f"{row['id']} / {row['best_match_location']}" if row["best_match_location"] else str(row["id"]),
            axis=1,
        )
        result_frame.insert(0, "rank", result_frame.index + 1)
        return result_frame[
            [
                "rank",
                "id",
                "source_type",
                "file_name",
                "title",
                "file_path",
                "audio_path",
                "processed_txt_path",
                "audio_file_path",
                "stt_txt_path",
                "category",
                "raw_score",
                "semantic_raw_score",
                "semantic_normalized_score",
                "normalized_score",
                "display_score",
                "similarity_score",
                "matched_tokens",
                "title_match_count",
                "description_match_count",
                "tags_match_count",
                "transcript_match_count",
                "field_weight_score",
                "ranker_score",
                "lexical_score",
                "semantic_score",
                "final_score",
                "reason",
                "score_kind",
                "raw_score_explanation",
                "best_match_summary",
                "best_match_location",
                "best_match_similarity",
                "best_match_text",
                "search_source",
                "transcript_preview",
                "preview",
                "original_preview",
                "stt_preview",
            ]
        ]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_score = float(scores.min())
    max_score = float(scores.max())
    if abs(max_score - min_score) < 1e-12:
        fill_value = 1.0 if max_score > 0 else 0.0
        return np.full_like(scores, fill_value, dtype=np.float32)
    return ((scores - min_score) / (max_score - min_score)).astype(np.float32)


def build_all_indices(
    metadata_path: Path,
    include_optional: bool = False,
    text_sources: list[str] | tuple[str, ...] = (DEFAULT_TEXT_SOURCE,),
    artifact_namespace: str | None = None,
    incremental: bool = True,
) -> list[tuple[str, str]]:
    ensure_project_dirs()
    metadata = load_metadata_frame(metadata_path)
    model_aliases = list(list_available_models(include_optional=include_optional).keys())
    built: list[tuple[str, str]] = []
    for text_source in text_sources:
        for alias in model_aliases:
            engine = DenseSearchEngine(
                metadata,
                alias,
                text_source=text_source,
                artifact_namespace=artifact_namespace,
            )
            engine.build(incremental=incremental)
            built.append((alias, text_source))
    return built


def load_index_summary(
    model_alias: str,
    text_source: str = DEFAULT_TEXT_SOURCE,
    artifact_namespace: str | None = None,
) -> dict:
    return load_json(INDICES_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_index_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense embedding indices.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--artifact-namespace", type=str, default=None)
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
        artifact_namespace=args.artifact_namespace,
    )
    print("Built embedding artifacts for:")
    for alias, text_source in built:
        print(f" - {alias} / {text_source}")


if __name__ == "__main__":
    main()
