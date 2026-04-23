from __future__ import annotations

"""정규화된 메타데이터 텍스트 위에서 BM25 또는 TF-IDF 기반 키워드 검색을 수행한다.

이 모듈은 어휘 점수 계산, 적응형 가중치 적용, 미리보기 생성, 인덱스 메타데이터
내보내기까지 함께 담당한다. 어휘 검색 규칙을 한 엔진에 모아야
오프라인 평가와 온라인 검색이 같은 토큰화와 같은 점수 후처리 규칙을 사용한다.
"""

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.adaptive.parameter_resolver import (
    AdaptiveContext,
    build_adaptive_context,
    resolve_query_search_config,
    resolve_top_k,
)
from src.config import DEFAULT_METADATA_CSV, INDICES_DIR, ensure_project_dirs
from src.data.metadata_schema import ensure_metadata_columns, load_metadata_frame
from src.search.explainability import explain_frame
from src.search.match_locator import locate_best_keyword_match, simple_tokenize
from src.search.text_source import (
    DEFAULT_TEXT_SOURCE,
    build_preview_text,
    build_search_text,
    resolve_primary_text,
    text_source_suffix,
)
from src.utils.io_utils import load_json, save_json


def _normalize_scores(scores) -> list[float]:
    """임의 범위의 점수를 0..1로 맞춰 이후 표시 및 결합 단계에서 재사용한다."""
    scores = list(map(float, scores))
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(min_score, max_score):
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _score_metadata(method: str) -> tuple[str, str]:
    """원시 점수 계열을 설명해 디버그 출력에서 점수 의미를 잃지 않게 한다."""
    normalized_method = method.lower()
    if normalized_method == "tfidf":
        return (
            "tfidf_dot",
            "raw_score is the TF-IDF dot product and display_score is query-relative 0~100 normalization.",
        )
    return (
        "bm25",
        "raw_score is the BM25 score and display_score is query-relative 0~100 normalization.",
    )


def _artifact_prefix(artifact_namespace: str | None) -> str:
    """여러 데이터셋 아티팩트가 충돌 없이 공존하도록 파일명 접두어를 만든다."""
    if not artifact_namespace:
        return ""
    safe_namespace = re.sub(r"[^0-9A-Za-z._-]+", "_", artifact_namespace).strip("._")
    return f"{safe_namespace}__" if safe_namespace else ""


class KeywordSearchEngine:
    """어휘 검색과 그에 따른 설명/내보내기 부수효과를 함께 캡슐화한다."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        text_source: str = DEFAULT_TEXT_SOURCE,
        adaptive_context: AdaptiveContext | None = None,
        artifact_namespace: str | None = None,
    ):
        """정규화된 메타데이터 필드로부터 어휘 검색 코퍼스를 한 번만 구성한다."""
        self.text_source = text_source
        self.artifact_namespace = artifact_namespace
        self.metadata = ensure_metadata_columns(metadata)
        # 이후 미리보기 생성과 문장 수준 근거 탐색이
        # 랭커 입력과 정확히 같은 텍스트 변형을 쓰도록 미리 물질화해 둔다.
        self.metadata["primary_text"] = self.metadata.apply(
            lambda row: resolve_primary_text(row, text_source=self.text_source),
            axis=1,
        )
        self.metadata["search_text"] = self.metadata.apply(
            lambda row: build_search_text(row, text_source=self.text_source),
            axis=1,
        )
        self.adaptive_context = adaptive_context or build_adaptive_context(
            self.metadata,
            text_source=self.text_source,
            artifact_namespace=artifact_namespace,
        )
        self.corpus_texts = self.metadata["search_text"].tolist()
        self.tokenized_corpus = [simple_tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.tfidf = TfidfVectorizer(tokenizer=simple_tokenize, lowercase=True)
        self.tfidf_matrix = self.tfidf.fit_transform(self.corpus_texts)

    @classmethod
    def from_csv(
        cls,
        metadata_path: Path = DEFAULT_METADATA_CSV,
        text_source: str = DEFAULT_TEXT_SOURCE,
        adaptive_context: AdaptiveContext | None = None,
        artifact_namespace: str | None = None,
    ) -> "KeywordSearchEngine":
        """CLI와 디버그 스크립트가 바로 쓸 수 있는 편의 생성자."""
        return cls(
            load_metadata_frame(metadata_path),
            text_source=text_source,
            adaptive_context=adaptive_context,
            artifact_namespace=artifact_namespace,
        )

    def search(self, query: str, top_k: int | None = None, method: str = "bm25") -> pd.DataFrame:
        """어휘 검색을 수행하고 설명/디버그 컬럼까지 함께 붙여 반환한다.

        이 엔진은 단순 랭커 점수만 돌려주지 않는다. 나머지 앱이
        미리보기, 필드 일치 수, 적응형 설정값, 문장 anchor까지
        이미 계산된 상태를 기대하기 때문이다.
        """
        top_k = resolve_top_k(self.adaptive_context.profile, top_k)
        query_tokens = simple_tokenize(query)
        normalized_method = method.lower()
        if normalized_method == "tfidf":
            query_vector = self.tfidf.transform([query])
            scores = (self.tfidf_matrix @ query_vector.T).toarray().ravel()
        else:
            scores = self.bm25.get_scores(query_tokens)

        results = self.metadata.copy()
        ranker_normalized_scores = _normalize_scores(scores)
        # 적응형 질의 설정은 원시 랭커 점수를 본 뒤 계산해야
        # 현재 질의의 점수 분포가 뾰족한지 평평한지에 반응할 수 있다.
        query_config = resolve_query_search_config(
            query,
            self.adaptive_context,
            keyword_scores=ranker_normalized_scores,
        )
        results["search_source"] = self.text_source
        score_kind, raw_score_explanation = _score_metadata(method)
        results = explain_frame(
            results,
            query,
            text_source=self.text_source,
            ranker_scores=[score * query_config.keyword_ranker_weight for score in ranker_normalized_scores],
            score_kind=score_kind,
            field_weights=query_config.field_weights,
        )
        results["ranker_raw_score"] = scores
        results["ranker_normalized_score"] = ranker_normalized_scores
        results["raw_score"] = results["final_score"].astype(float)
        # 최종 점수는 UI 표시용으로 한 번 더 정규화하되,
        # 원시값과 최종값은 디버그와 평가를 위해 별도 보존한다.
        results["normalized_score"] = _normalize_scores(results["final_score"])
        results["display_score"] = results["normalized_score"].astype(float) * 100.0
        results["similarity_score"] = results["display_score"]
        results["score_kind"] = score_kind
        results["adaptive_field_weights"] = str(query_config.field_weights)
        results["adaptive_keyword_alpha"] = query_config.keyword_alpha
        results["adaptive_dense_alpha"] = query_config.dense_alpha
        results["adaptive_ranker_weight"] = query_config.keyword_ranker_weight
        results["adaptive_preview_length"] = query_config.preview_length
        results["adaptive_reason"] = query_config.reasoning
        results["raw_score_explanation"] = (
            f"field_weights={query_config.field_weights}; "
            f"keyword_ranker_weight={query_config.keyword_ranker_weight:.3f}; "
            f"keyword_alpha={query_config.keyword_alpha:.3f}; "
            f"dense_alpha={query_config.dense_alpha:.3f}; "
            f"{raw_score_explanation}"
        )
        results["preview"] = results.apply(
            lambda row: build_preview_text(row, text_source=self.text_source, length=query_config.preview_length),
            axis=1,
        )
        results["transcript_preview"] = results["preview"]
        results["original_preview"] = results["original_transcript"].fillna("").astype(str).str.slice(0, query_config.preview_length) + "..."
        results["stt_preview"] = results["stt_transcript"].fillna("").astype(str).str.slice(0, query_config.preview_length) + "..."
        results = results.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)
        # 문장 수준 근거 탐색은 top-k로 줄인 뒤에만 수행해
        # 코퍼스 전체 문서에 대해 스니펫 비용을 지불하지 않도록 한다.
        match_details = results["primary_text"].apply(lambda text: locate_best_keyword_match(text, query, method=method))
        results = pd.concat([results, pd.DataFrame(match_details.tolist())], axis=1)
        results["best_match_summary"] = results.apply(
            lambda row: f"{row['id']} / {row['best_match_location']}" if row["best_match_location"] else str(row["id"]),
            axis=1,
        )
        results.insert(0, "rank", results.index + 1)
        return results[
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
                "ranker_raw_score",
                "ranker_normalized_score",
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
                "adaptive_ranker_weight",
                "adaptive_preview_length",
                "adaptive_reason",
            ]
        ]

    def export_index_metadata(
        self,
        output_path: Path | None = None,
        artifact_namespace: str | None = None,
    ) -> Path:
        """어휘 인덱스가 어떻게 구성되었는지 감사할 수 있을 만큼의 메타데이터를 저장한다."""
        ensure_project_dirs()
        suffix = text_source_suffix(self.text_source)
        payload = {
            "method": ["bm25", "tfidf"],
            "text_source": self.text_source,
            "document_count": len(self.metadata),
            "columns": self.metadata.columns.tolist(),
            "artifact_namespace": artifact_namespace,
            "adaptive_profile": self.adaptive_context.profile.to_dict(),
            "adaptive_search": self.adaptive_context.search.to_dict(),
        }
        output_path = output_path or INDICES_DIR / f"{_artifact_prefix(artifact_namespace)}keyword_index_metadata__{suffix}.json"
        save_json(output_path, payload)
        return output_path


def load_keyword_index_summary(
    text_source: str = DEFAULT_TEXT_SOURCE,
    path: Path | None = None,
    artifact_namespace: str | None = None,
) -> dict:
    """export_index_metadata가 저장한 어휘 인덱스 요약을 다시 읽는다."""
    suffix = text_source_suffix(text_source)
    path = path or INDICES_DIR / f"{_artifact_prefix(artifact_namespace)}keyword_index_metadata__{suffix}.json"
    return load_json(path)


def main() -> None:
    """어휘 검색을 빠르게 점검하는 CLI 진입점."""
    parser = argparse.ArgumentParser(description="Build and smoke-test keyword search.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--query", type=str, default="youtube interview transcript")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--method", type=str, default="bm25", choices=["bm25", "tfidf"])
    parser.add_argument("--artifact-namespace", type=str, default=None)
    parser.add_argument(
        "--text-source",
        type=str,
        default=DEFAULT_TEXT_SOURCE,
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    args = parser.parse_args()

    engine = KeywordSearchEngine.from_csv(
        args.metadata_path,
        text_source=args.text_source,
        artifact_namespace=args.artifact_namespace,
    )
    output_path = engine.export_index_metadata(artifact_namespace=args.artifact_namespace)
    print(f"Keyword index metadata saved to {output_path}")
    print(engine.search(args.query, top_k=args.top_k, method=args.method).to_string(index=False))


if __name__ == "__main__":
    main()
