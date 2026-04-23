from __future__ import annotations

"""밀집 검색용 임베딩 아티팩트를 생성하고 재사용하는 엔진.

이 모듈은 메타데이터를 검색용 텍스트로 바꾸고, 임베딩을 만들고, FAISS 인덱스와
요약 메타데이터를 저장하며, 질의 시점에는 같은 규칙으로 점수를 계산한다.
인덱싱과 검색을 같은 모듈에 두는 이유는 아티팩트 정합성 규칙이 한곳에 있어야
오래된 임베딩과 새 메타데이터가 조용히 섞이는 문제를 막을 수 있기 때문이다.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.adaptive.parameter_resolver import (
    AdaptiveContext,
    build_adaptive_context,
    resolve_query_search_config,
    resolve_top_k,
)
from src.adaptive.query_features import extract_query_features
from src.config import DEFAULT_METADATA_CSV, EMBEDDINGS_DIR, INDICES_DIR, ensure_project_dirs
from src.data.metadata_schema import ensure_metadata_columns, load_metadata_frame
from src.embedding.vector_models import EmbeddingModelWrapper, list_available_models
from src.search.explainability import explain_frame
from src.search.match_locator import locate_best_dense_match
from src.search.text_source import (
    DEFAULT_TEXT_SOURCE,
    DEFAULT_DENSE_NORMALIZATION_MODE,
    build_preview_text,
    build_search_text,
    prepare_query_for_dense,
    resolve_dense_normalization_mode,
    resolve_primary_text,
    text_source_suffix,
)
from src.utils.io_utils import load_json, save_json, save_numpy
from src.utils.device import resolve_torch_device

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def artifact_stem(model_alias: str, text_source: str, artifact_namespace: str | None = None) -> str:
    """모델, 텍스트 소스, 네임스페이스를 합쳐 아티팩트 파일명 뼈대를 만든다."""
    base = f"{model_alias}__{text_source_suffix(text_source)}"
    if not artifact_namespace:
        return base
    safe_namespace = re.sub(r"[^0-9A-Za-z._-]+", "_", artifact_namespace).strip("._")
    return f"{safe_namespace}__{base}" if safe_namespace else base


class DenseSearchEngine:
    """임베딩 생성, 저장, 로딩, 질의 점수 계산을 함께 담당하는 밀집 검색 엔진."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        model_alias: str,
        text_source: str = DEFAULT_TEXT_SOURCE,
        artifact_namespace: str | None = None,
        adaptive_context: AdaptiveContext | None = None,
        dense_normalization_mode: str | None = None,
    ):
        """아티팩트 경로와 모델 래퍼를 초기화하고 메타데이터를 검색 형태로 정리한다."""
        self.model_alias = model_alias
        self.text_source = text_source
        self.artifact_namespace = artifact_namespace
        self._artifact_stem = artifact_stem(model_alias, text_source, artifact_namespace)
        self.embedding_path = EMBEDDINGS_DIR / f"{self._artifact_stem}_embeddings.npy"
        self.doc_meta_path = EMBEDDINGS_DIR / f"{self._artifact_stem}_metadata.csv"
        self.index_path = INDICES_DIR / f"{self._artifact_stem}.faiss"
        self.index_summary_path = INDICES_DIR / f"{self._artifact_stem}_index_summary.json"
        self.wrapper = EmbeddingModelWrapper(model_alias)
        self.device = resolve_torch_device()
        self.adaptive_context = adaptive_context
        self.normalization_resources = adaptive_context.normalization if adaptive_context is not None else None
        self.dense_normalization_mode = resolve_dense_normalization_mode(
            dense_normalization_mode,
            resources=self.normalization_resources,
        )
        self._set_metadata(metadata)
        self.embeddings: np.ndarray | None = None
        self.index = None

    def _set_metadata(self, metadata: pd.DataFrame) -> None:
        """메타데이터를 현재 검색 설정에 맞는 텍스트 컬럼 구조로 재구성한다."""
        frame = ensure_metadata_columns(metadata).reset_index(drop=True)
        frame["primary_text"] = frame.apply(
            lambda row: resolve_primary_text(row, text_source=self.text_source),
            axis=1,
        )
        frame["search_text"] = frame.apply(
                lambda row: build_search_text(
                    row,
                    text_source=self.text_source,
                    for_dense=True,
                    normalization_mode=self.dense_normalization_mode,
                    resources=self.normalization_resources,
                ),
                axis=1,
            )
        self.metadata = frame
        self.document_texts = self.metadata["search_text"].astype(str).tolist()

    def _ensure_adaptive_context(self) -> None:
        """적응형 컨텍스트가 없으면 현재 메타데이터와 임베딩 상태로 즉석 생성한다."""
        if self.adaptive_context is None:
            self.adaptive_context = build_adaptive_context(
                self.metadata,
                text_source=self.text_source,
                embedding_model_alias=self.model_alias,
                embeddings=self.embeddings,
                artifact_namespace=self.artifact_namespace,
            )
            self.normalization_resources = self.adaptive_context.normalization
            self.dense_normalization_mode = resolve_dense_normalization_mode(
                self.dense_normalization_mode,
                resources=self.normalization_resources,
            )
            self._set_metadata(self.metadata)

    def _artifact_alignment_status(
        self,
        embeddings: np.ndarray,
        stored_metadata: pd.DataFrame | None,
    ) -> tuple[bool, str]:
        """저장된 임베딩/메타데이터 아티팩트가 현재 메타데이터와 정합적인지 검사한다.

        길이, ID 순서, 검색 텍스트가 조금만 어긋나도 점수 해석이 깨지므로
        느슨하게 넘어가지 않고 이유 문자열까지 함께 반환한다.
        """
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
                lambda row: build_search_text(
                    row,
                    text_source=self.text_source,
                    for_dense=True,
                    normalization_mode=self.dense_normalization_mode,
                    resources=self.normalization_resources,
                ),
                axis=1,
            ).astype(str).tolist()
        current_search_text = current["search_text"].fillna("").astype(str).tolist()
        if stored_search_text != current_search_text:
            return False, "saved search text does not match current metadata search text"

        return True, ""

    def _load_stored_metadata(self) -> pd.DataFrame | None:
        """저장된 문서 메타데이터 CSV가 있으면 읽어 온다."""
        if not self.doc_meta_path.exists():
            return None
        return load_metadata_frame(self.doc_meta_path)

    def _search_texts_for_frame(self, frame: pd.DataFrame) -> list[str]:
        """프레임이 가진 검색 텍스트를 현재 규칙으로 재구성해 비교에 사용한다."""
        normalized = ensure_metadata_columns(frame).reset_index(drop=True)
        if "search_text" in normalized.columns:
            return normalized["search_text"].fillna("").astype(str).tolist()
        return normalized.apply(
            lambda row: build_search_text(
                row,
                text_source=self.text_source,
                for_dense=True,
                normalization_mode=self.dense_normalization_mode,
                resources=self.normalization_resources,
            ),
            axis=1,
        ).astype(str).tolist()

    def _append_candidates(self, stored_metadata: pd.DataFrame) -> pd.DataFrame | None:
        """증분 추가가 안전한 경우에만 새로 붙은 행 집합을 계산한다.

        순서나 기존 검색 텍스트가 어긋난 경우에는 append가 아니라 전체 재생성이
        필요하므로 None을 돌려준다.
        """
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
            lambda row: build_search_text(
                row,
                text_source=self.text_source,
                for_dense=True,
                normalization_mode=self.dense_normalization_mode,
            ),
            axis=1,
        ).astype(str).tolist()
        if current_prefix_search_texts != stored_search_texts:
            return None

        appended = current.iloc[len(stored) :].copy().reset_index(drop=True)
        if appended.empty:
            return appended
        return appended

    def _save_index_summary(self, document_count: int, index_saved: bool, build_mode: str) -> None:
        """나중에 디버그와 감사가 가능하도록 인덱스 생성 요약 정보를 저장한다."""
        self._ensure_adaptive_context()
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
                "adaptive_profile": self.adaptive_context.profile.to_dict(),
                "adaptive_search": self.adaptive_context.search.to_dict(),
                "adaptive_cluster": self.adaptive_context.cluster.to_dict(),
                "adaptive_visualization": self.adaptive_context.visualization.to_dict(),
            },
        )

    def _write_full_artifacts(self, build_mode: str = "full_rebuild") -> None:
        """임베딩과 인덱스를 처음부터 다시 생성해 전체 아티팩트를 갱신한다."""
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
        """저장된 임베딩 뒤에 안전하게 추가 가능한 새 행만 증분 반영한다."""
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
        """현재 메타데이터 기준으로 밀집 검색 아티팩트를 준비한다.

        가능하면 기존 아티팩트를 재사용하지만, 정합성이 조금이라도 깨졌다면
        전체 재생성을 선택해 침묵하는 불일치를 막는다.
        """
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
                    self._save_index_summary(
                        document_count=len(self.metadata),
                        index_saved=self.index is not None,
                        build_mode="no_update",
                    )
                    return
                # 오래된 흔적이 보이면 부분 append보다 전체 재생성을 강제한다.
                # 그렇지 않으면 아티팩트 간 문서 순서와 정렬 축이 틀어질 수 있다.
        self._write_full_artifacts(build_mode="full_rebuild")

    def load(self, rebuild_if_missing: bool = True, rebuild_if_mismatch: bool = True) -> None:
        """저장된 임베딩/인덱스를 로드하고 필요하면 재생성한다."""
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
        """질의 하나를 현재 모델 규칙으로 임베딩한다."""
        return self.wrapper.encode_queries([query])[0]

    def score_all(self, query: str) -> pd.DataFrame:
        """전체 문서에 대해 질의 점수를 계산하고 디버그 컬럼까지 붙여 반환한다.

        이 함수는 검색 랭킹의 핵심 계산 경로다. 질의 정규화, 임베딩 유사도,
        적응형 가중치, 설명 컬럼을 한 번에 조립한다.
        """
        if self.embeddings is None or len(self.metadata) != len(self.embeddings):
            self.load()
        assert self.embeddings is not None
        self._ensure_adaptive_context()
        if len(self.metadata) != len(self.embeddings):
            raise ValueError(
                f"Embedding rows ({len(self.embeddings)}) do not match metadata rows ({len(self.metadata)}) "
                f"for namespace={self.artifact_namespace or 'default'}"
            )

        query_features = extract_query_features(
            query,
            self.adaptive_context.profile,
            self.normalization_resources,
            self.adaptive_context.performance,
        )
        adaptive_query_mode = self.dense_normalization_mode
        # 질의 특성이 STT 노이즈/구어체 쪽에 더 가깝다면
        # 문서 정규화 리소스를 적극 반영하는 adaptive_corpus 모드로 전환한다.
        adaptive_score = float(
            np.mean(
                [
                    query_features.question_likeness,
                    query_features.spoken_style,
                    query_features.stt_noise_score,
                    query_features.reranker_value_signal,
                    self.normalization_resources.normalization_preference,
                ]
            )
        )
        baseline_score = float(np.mean([query_features.lexical_precision, self.adaptive_context.profile.stt_quality_score]))
        if adaptive_score >= baseline_score:
            adaptive_query_mode = "adaptive_corpus"

        dense_query = prepare_query_for_dense(
            query,
            mode=adaptive_query_mode,
            resources=self.normalization_resources,
        )
        query_embedding = self.encode_query(dense_query)
        if self.device.startswith("cuda"):
            # GPU가 있으면 대규모 내적 계산을 torch로 수행해 점수 계산 비용을 줄인다.
            with torch.inference_mode():
                emb_t = torch.from_numpy(self.embeddings).to(self.device, non_blocking=True)
                query_t = torch.from_numpy(query_embedding).to(self.device, non_blocking=True)
                semantic_raw_scores = torch.matmul(emb_t, query_t).detach().cpu().numpy()
        else:
            semantic_raw_scores = self.embeddings @ query_embedding
        semantic_normalized_scores = np.clip((semantic_raw_scores + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        query_config = resolve_query_search_config(
            query,
            self.adaptive_context,
            dense_scores=semantic_normalized_scores,
        )

        result_frame = self.metadata.copy()
        result_frame["semantic_raw_score"] = semantic_raw_scores
        result_frame["semantic_normalized_score"] = semantic_normalized_scores
        result_frame = explain_frame(
            result_frame,
            query,
            text_source=self.text_source,
            semantic_scores=[score * query_config.dense_semantic_weight for score in semantic_normalized_scores],
            score_kind="cosine_similarity",
            field_weights=query_config.field_weights,
        )
        result_frame["raw_score"] = result_frame["final_score"].astype(float)
        # 최종 점수는 UI와 평가가 쓰기 쉬운 0..1, 0..100 표현도 함께 만든다.
        result_frame["normalized_score"] = _normalize_scores(result_frame["final_score"].to_numpy())
        result_frame["display_score"] = result_frame["normalized_score"] * 100.0
        result_frame["similarity_score"] = result_frame["display_score"]
        result_frame["search_source"] = self.text_source
        result_frame["score_kind"] = "weighted_lexical_semantic"
        result_frame["adaptive_field_weights"] = str(query_config.field_weights)
        result_frame["adaptive_keyword_alpha"] = query_config.keyword_alpha
        result_frame["adaptive_dense_alpha"] = query_config.dense_alpha
        result_frame["adaptive_semantic_weight"] = query_config.dense_semantic_weight
        result_frame["adaptive_preview_length"] = query_config.preview_length
        result_frame["adaptive_reason"] = query_config.reasoning
        result_frame["dense_normalization_mode"] = self.dense_normalization_mode
        result_frame["query_normalization_mode"] = adaptive_query_mode
        result_frame["used_fallback_tuning"] = int(bool(self.adaptive_context.tuning_status.get("used_safe_fallback", False)))
        result_frame["fallback_reason"] = str(self.adaptive_context.tuning_status.get("fallback_reason", ""))
        result_frame["inference_device"] = self.device
        result_frame["raw_score_explanation"] = (
            f"field_weights={query_config.field_weights}; "
            f"dense_semantic_weight={query_config.dense_semantic_weight:.3f}; "
            f"keyword_alpha={query_config.keyword_alpha:.3f}; "
            f"dense_alpha={query_config.dense_alpha:.3f}; "
            "final_score = lexical_score + semantic_score with adaptive weighting."
        )
        result_frame["preview"] = result_frame.apply(
            lambda row: build_preview_text(row, text_source=self.text_source, length=query_config.preview_length),
            axis=1,
        )
        result_frame["transcript_preview"] = result_frame["preview"]
        result_frame["original_preview"] = result_frame["original_transcript"].fillna("").astype(str).str.slice(0, query_config.preview_length) + "..."
        result_frame["stt_preview"] = result_frame["stt_transcript"].fillna("").astype(str).str.slice(0, query_config.preview_length) + "..."
        return result_frame.sort_values("final_score", ascending=False).reset_index(drop=True)

    def search(self, query: str, top_k: int | None = None) -> pd.DataFrame:
        """전체 점수 계산 결과에서 상위 k개만 남기고 문장 anchor를 추가한다."""
        self._ensure_adaptive_context()
        top_k = resolve_top_k(self.adaptive_context.profile, top_k)
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
                "adaptive_field_weights",
                "adaptive_keyword_alpha",
                "adaptive_dense_alpha",
                "adaptive_semantic_weight",
                "adaptive_preview_length",
                "adaptive_reason",
            ]
        ]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """점수 벡터를 0..1로 정규화해 표시와 후속 결합에 재사용한다."""
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
    """지정한 데이터셋과 텍스트 소스 전부에 대해 밀집 인덱스를 생성한다."""
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
    """저장된 밀집 인덱스 요약 JSON을 읽어 온다."""
    return load_json(INDICES_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_index_summary.json")


def main() -> None:
    """밀집 인덱스를 생성하는 CLI 진입점."""
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
    for alias, text_source in built:
        summary = load_index_summary(alias, text_source, artifact_namespace=args.artifact_namespace)
        print(summary)


if __name__ == "__main__":
    main()
