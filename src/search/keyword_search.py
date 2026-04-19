from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DEFAULT_METADATA_CSV, INDICES_DIR, ensure_project_dirs
from src.data.metadata_schema import ensure_metadata_columns, load_metadata_frame
from src.search.explainability import RANKER_SCORE_WEIGHT, explain_frame
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
    scores = list(map(float, scores))
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(min_score, max_score):
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _score_metadata(method: str) -> tuple[str, str]:
    normalized_method = method.lower()
    if normalized_method == "tfidf":
        return (
            "tfidf_dot",
            "raw_score는 TF-IDF 문서 벡터와 질의 벡터의 내적 값이며, display_score는 현재 질의 기준 0~100 정규화 점수입니다.",
        )
    return (
        "bm25",
        "raw_score는 BM25 점수이며, display_score는 현재 질의 기준 0~100 정규화 점수입니다.",
    )


def _artifact_prefix(artifact_namespace: str | None) -> str:
    if not artifact_namespace:
        return ""
    safe_namespace = re.sub(r"[^0-9A-Za-z._-]+", "_", artifact_namespace).strip("._")
    return f"{safe_namespace}__" if safe_namespace else ""


class KeywordSearchEngine:
    def __init__(self, metadata: pd.DataFrame, text_source: str = DEFAULT_TEXT_SOURCE):
        self.text_source = text_source
        self.metadata = ensure_metadata_columns(metadata)
        self.metadata["primary_text"] = self.metadata.apply(
            lambda row: resolve_primary_text(row, text_source=self.text_source),
            axis=1,
        )
        self.metadata["search_text"] = self.metadata.apply(
            lambda row: build_search_text(row, text_source=self.text_source),
            axis=1,
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
    ) -> "KeywordSearchEngine":
        return cls(load_metadata_frame(metadata_path), text_source=text_source)

    def search(self, query: str, top_k: int = 10, method: str = "bm25") -> pd.DataFrame:
        query_tokens = simple_tokenize(query)
        normalized_method = method.lower()
        if normalized_method == "tfidf":
            query_vector = self.tfidf.transform([query])
            scores = (self.tfidf_matrix @ query_vector.T).toarray().ravel()
        else:
            scores = self.bm25.get_scores(query_tokens)

        results = self.metadata.copy()
        ranker_normalized_scores = _normalize_scores(scores)
        results["search_source"] = self.text_source
        score_kind, raw_score_explanation = _score_metadata(method)
        results = explain_frame(
            results,
            query,
            text_source=self.text_source,
            ranker_scores=[score * RANKER_SCORE_WEIGHT for score in ranker_normalized_scores],
            score_kind=score_kind,
        )
        results["ranker_raw_score"] = scores
        results["ranker_normalized_score"] = ranker_normalized_scores
        results["raw_score"] = results["final_score"].astype(float)
        results["normalized_score"] = _normalize_scores(results["final_score"])
        results["display_score"] = results["normalized_score"].astype(float) * 100.0
        results["similarity_score"] = results["display_score"]
        results["score_kind"] = score_kind
        results["raw_score_explanation"] = (
            f"final_score = lexical_score + semantic_score. "
            f"lexical_score = title*5 + tags*4 + description*3 + transcript*2 + "
            f"{score_kind} normalized score*{RANKER_SCORE_WEIGHT:.1f}. "
            f"{raw_score_explanation}"
        )
        results["preview"] = results.apply(lambda row: build_preview_text(row, text_source=self.text_source), axis=1)
        results["transcript_preview"] = results["preview"]
        results["original_preview"] = results["original_transcript"].str.slice(0, 140) + "..."
        results["stt_preview"] = results["stt_transcript"].str.slice(0, 140) + "..."
        results = results.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)
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
            ]
        ]

    def export_index_metadata(
        self,
        output_path: Path | None = None,
        artifact_namespace: str | None = None,
    ) -> Path:
        ensure_project_dirs()
        suffix = text_source_suffix(self.text_source)
        payload = {
            "method": ["bm25", "tfidf"],
            "text_source": self.text_source,
            "document_count": len(self.metadata),
            "columns": self.metadata.columns.tolist(),
            "artifact_namespace": artifact_namespace,
        }
        output_path = output_path or INDICES_DIR / f"{_artifact_prefix(artifact_namespace)}keyword_index_metadata__{suffix}.json"
        save_json(output_path, payload)
        return output_path


def load_keyword_index_summary(
    text_source: str = DEFAULT_TEXT_SOURCE,
    path: Path | None = None,
    artifact_namespace: str | None = None,
) -> dict:
    suffix = text_source_suffix(text_source)
    path = path or INDICES_DIR / f"{_artifact_prefix(artifact_namespace)}keyword_index_metadata__{suffix}.json"
    return load_json(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and smoke-test keyword search.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--query", type=str, default="youtube interview transcript")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--method", type=str, default="bm25", choices=["bm25", "tfidf"])
    parser.add_argument("--artifact-namespace", type=str, default=None)
    parser.add_argument(
        "--text-source",
        type=str,
        default=DEFAULT_TEXT_SOURCE,
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    args = parser.parse_args()

    engine = KeywordSearchEngine.from_csv(args.metadata_path, text_source=args.text_source)
    output_path = engine.export_index_metadata(artifact_namespace=args.artifact_namespace)
    print(f"Keyword index metadata saved to {output_path}")
    print(engine.search(args.query, top_k=args.top_k, method=args.method).to_string(index=False))


if __name__ == "__main__":
    main()
