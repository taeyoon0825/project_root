from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.adaptive.parameter_resolver import AdaptiveContext, resolve_query_search_config, resolve_top_k
from src.embedding.build_indices import DenseSearchEngine
from src.search.keyword_search import KeywordSearchEngine


def _normalize_minmax(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _normalize_rank(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    ranked = values.rank(method="average", ascending=False)
    return (1.0 - ((ranked - 1.0) / max(1.0, len(values) - 1.0))).astype(np.float32)


def _hybrid_normalize(values: pd.Series) -> pd.Series:
    minmax = _normalize_minmax(values)
    rank = _normalize_rank(values)
    return (0.7 * minmax + 0.3 * rank).astype(np.float32)


def _score_margin(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    arr = np.sort(values.astype(float).to_numpy())
    top = float(arr[-1])
    ref = float(arr[max(0, len(arr) - min(5, len(arr)))])
    return float(max(0.0, min(1.0, top - ref)))


def _query_tokens(query: str) -> list[str]:
    import re

    return re.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", str(query or ""))


def _query_features(query: str) -> dict[str, float]:
    tokens = _query_tokens(query)
    token_count = len(tokens)
    numeric_ratio = sum(1 for token in tokens if any(ch.isdigit() for ch in token)) / max(1, token_count)
    short_keyword = 1.0 if 0 < token_count <= 3 else 0.0
    long_query = 1.0 if token_count >= 8 else 0.0
    question_like = 1.0 if str(query or "").strip().endswith("?") else 0.0
    return {
        "token_count": float(token_count),
        "numeric_ratio": float(numeric_ratio),
        "short_keyword": short_keyword,
        "long_query": long_query,
        "question_like": question_like,
    }


@dataclass
class FusionWeights:
    bm25: float
    minilm: float
    e5: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bm25": self.bm25,
            "minilm": self.minilm,
            "e5": self.e5,
            "reasoning": self.reasoning,
        }


def resolve_fusion_weights(
    query: str,
    context: AdaptiveContext,
    *,
    bm25_scores: pd.Series,
    minilm_scores: pd.Series,
    e5_scores: pd.Series,
) -> FusionWeights:
    profile = context.profile
    base_keyword = float(context.search.keyword_alpha)
    base_dense = float(context.search.dense_alpha)
    base_minilm_share = 0.5 if profile.dominant_language == "ko" else 0.45 if profile.dominant_language == "en" else 0.4
    base_e5_share = 1.0 - base_minilm_share

    query_stat = _query_features(query)
    bm25_margin = _score_margin(bm25_scores)
    minilm_margin = _score_margin(minilm_scores)
    e5_margin = _score_margin(e5_scores)

    w_bm25 = base_keyword
    w_bm25 += 0.12 * query_stat["short_keyword"]
    w_bm25 += 0.10 * query_stat["numeric_ratio"]
    w_bm25 += 0.10 * bm25_margin
    w_bm25 -= 0.12 * query_stat["long_query"]
    w_bm25 -= 0.08 * (1.0 - profile.stt_quality_score)

    dense_total = base_dense
    dense_total += 0.14 * query_stat["long_query"]
    dense_total += 0.10 * query_stat["question_like"]
    dense_total += 0.08 * (1.0 - profile.stt_quality_score)
    dense_total -= 0.10 * query_stat["short_keyword"]

    minilm_share = base_minilm_share + (0.08 * minilm_margin) - (0.05 * query_stat["numeric_ratio"])
    e5_share = base_e5_share + (0.08 * e5_margin) + (0.05 * query_stat["numeric_ratio"])
    share_total = max(1e-9, minilm_share + e5_share)
    minilm_share = min(max(minilm_share / share_total, 0.2), 0.8)
    e5_share = 1.0 - minilm_share

    w_bm25 = min(max(w_bm25, 0.1), 0.7)
    dense_total = min(max(dense_total, 0.2), 0.8)
    total = w_bm25 + dense_total
    w_bm25 = w_bm25 / total
    dense_total = dense_total / total
    w_minilm = dense_total * minilm_share
    w_e5 = dense_total * e5_share
    final_total = w_bm25 + w_minilm + w_e5
    w_bm25, w_minilm, w_e5 = w_bm25 / final_total, w_minilm / final_total, w_e5 / final_total

    reasoning = (
        f"base_keyword={base_keyword:.3f}, base_dense={base_dense:.3f}, "
        f"short={query_stat['short_keyword']:.2f}, long={query_stat['long_query']:.2f}, numeric={query_stat['numeric_ratio']:.2f}, "
        f"margins(bm25/minilm/e5)=({bm25_margin:.3f}/{minilm_margin:.3f}/{e5_margin:.3f}), "
        f"weights=({w_bm25:.3f}/{w_minilm:.3f}/{w_e5:.3f})"
    )
    return FusionWeights(bm25=w_bm25, minilm=w_minilm, e5=w_e5, reasoning=reasoning)


class FusedSearchEngine:
    def __init__(
        self,
        metadata: pd.DataFrame,
        *,
        text_source: str,
        artifact_namespace: str | None,
        adaptive_context: AdaptiveContext,
        minilm_alias: str = "paraphrase-multilingual-MiniLM-L12-v2",
        e5_alias: str = "multilingual-e5-base",
    ):
        self.metadata = metadata
        self.text_source = text_source
        self.artifact_namespace = artifact_namespace
        self.context = adaptive_context
        self.keyword = KeywordSearchEngine(metadata, text_source=text_source, adaptive_context=adaptive_context, artifact_namespace=artifact_namespace)
        self.minilm = DenseSearchEngine(
            metadata,
            minilm_alias,
            text_source=text_source,
            artifact_namespace=artifact_namespace,
            adaptive_context=adaptive_context,
        )
        self.e5 = DenseSearchEngine(
            metadata,
            e5_alias,
            text_source=text_source,
            artifact_namespace=artifact_namespace,
            adaptive_context=adaptive_context,
        )
        self.minilm.load()
        self.e5.load()

    def _join_scores(self, bm25: pd.DataFrame, mini: pd.DataFrame, e5: pd.DataFrame) -> pd.DataFrame:
        base = self.metadata.copy()
        base["id"] = base["id"].astype(str)
        bm = bm25[["id", "final_score", "display_score", "adaptive_reason", "best_match_text", "best_match_location"]].copy()
        bm.columns = ["id", "bm25_final", "bm25_display", "bm25_reason", "bm25_best_match_text", "bm25_best_match_location"]
        mi = mini[["id", "final_score", "display_score", "adaptive_reason", "best_match_text", "best_match_location"]].copy()
        mi.columns = ["id", "minilm_final", "minilm_display", "minilm_reason", "minilm_best_match_text", "minilm_best_match_location"]
        e5f = e5[["id", "final_score", "display_score", "adaptive_reason", "best_match_text", "best_match_location"]].copy()
        e5f.columns = ["id", "e5_final", "e5_display", "e5_reason", "e5_best_match_text", "e5_best_match_location"]
        merged = base.merge(bm, on="id", how="left").merge(mi, on="id", how="left").merge(e5f, on="id", how="left")
        for column in ["bm25_final", "bm25_display", "minilm_final", "minilm_display", "e5_final", "e5_display"]:
            merged[column] = merged[column].fillna(0.0).astype(float)
        return merged

    def search(self, query: str, top_k: int | None = None, keyword_method: str = "bm25") -> pd.DataFrame:
        resolved_top_k = resolve_top_k(self.context.profile, top_k)
        candidate_k = max(20, min(50, resolved_top_k * 5))
        bm25 = self.keyword.search(query, top_k=candidate_k, method=keyword_method)
        minilm = self.minilm.search(query, top_k=candidate_k)
        e5 = self.e5.search(query, top_k=candidate_k)

        merged = self._join_scores(bm25, minilm, e5)
        merged["norm_bm25"] = _hybrid_normalize(merged["bm25_final"])
        merged["norm_minilm"] = _hybrid_normalize(merged["minilm_final"])
        merged["norm_e5"] = _hybrid_normalize(merged["e5_final"])

        weights = resolve_fusion_weights(
            query,
            self.context,
            bm25_scores=merged["norm_bm25"],
            minilm_scores=merged["norm_minilm"],
            e5_scores=merged["norm_e5"],
        )
        merged["fused_score"] = (
            weights.bm25 * merged["norm_bm25"]
            + weights.minilm * merged["norm_minilm"]
            + weights.e5 * merged["norm_e5"]
        )
        merged["display_score"] = (_hybrid_normalize(merged["fused_score"]) * 100.0).astype(float)
        merged["similarity_score"] = merged["display_score"]

        query_cfg = resolve_query_search_config(
            query,
            self.context,
            keyword_scores=merged["norm_bm25"].to_numpy(),
            dense_scores=((merged["norm_minilm"] + merged["norm_e5"]) / 2.0).to_numpy(),
        )

        def _choose_preview_reason(row: pd.Series) -> tuple[str, str]:
            keyword_part = weights.bm25 * float(row["norm_bm25"])
            dense_part = (weights.minilm * float(row["norm_minilm"])) + (weights.e5 * float(row["norm_e5"]))
            if keyword_part >= dense_part:
                return "keyword", str(row.get("bm25_best_match_text", "") or "")
            if weights.e5 >= weights.minilm:
                return "dense_e5", str(row.get("e5_best_match_text", "") or "")
            return "dense_minilm", str(row.get("minilm_best_match_text", "") or "")

        preview_reasons = merged.apply(_choose_preview_reason, axis=1)
        merged["chosen_preview_reason"] = preview_reasons.apply(lambda item: item[0])
        merged["best_match_text"] = preview_reasons.apply(lambda item: item[1])
        merged["best_match_location"] = merged.apply(
            lambda row: str(row.get("bm25_best_match_location") or row.get("minilm_best_match_location") or row.get("e5_best_match_location") or ""),
            axis=1,
        )
        merged["rank"] = merged["fused_score"].rank(method="first", ascending=False).astype(int)
        merged = merged.sort_values("fused_score", ascending=False).head(resolved_top_k).reset_index(drop=True)
        merged.insert(0, "rank", merged.index + 1)
        merged["adaptive_field_weights"] = str(query_cfg.field_weights)
        merged["adaptive_keyword_alpha"] = float(query_cfg.keyword_alpha)
        merged["adaptive_dense_alpha"] = float(query_cfg.dense_alpha)
        merged["adaptive_preview_length"] = int(query_cfg.preview_length)
        merged["adaptive_reason"] = query_cfg.reasoning
        merged["fusion_weights"] = json.dumps(weights.to_dict(), ensure_ascii=False)
        merged["raw_score"] = merged["fused_score"].astype(float)
        merged["final_score"] = merged["fused_score"].astype(float)
        merged["raw_score_explanation"] = (
            "fused_score = w_bm25*norm_bm25 + w_minilm*norm_minilm + w_e5*norm_e5; "
            f"weights={weights.to_dict()}"
        )
        merged["score_kind"] = "fused_weighted_score"
        merged["reason"] = "adaptive fusion retrieval"
        merged["search_source"] = self.text_source
        merged["top_model_contribution"] = merged.apply(
            lambda row: max(
                [
                    ("bm25", weights.bm25 * float(row["norm_bm25"])),
                    ("minilm", weights.minilm * float(row["norm_minilm"])),
                    ("e5", weights.e5 * float(row["norm_e5"])),
                ],
                key=lambda item: item[1],
            )[0],
            axis=1,
        )
        merged["ranking_reason"] = merged.apply(
            lambda row: f"fusion winner={row['top_model_contribution']} with weights={weights.to_dict()}",
            axis=1,
        )
        merged["matched_tokens"] = ""
        merged["title_match_count"] = 0
        merged["description_match_count"] = 0
        merged["tags_match_count"] = 0
        merged["transcript_match_count"] = 0
        merged["field_weight_score"] = 0.0
        merged["ranker_score"] = 0.0
        merged["lexical_score"] = merged["norm_bm25"].astype(float)
        merged["semantic_score"] = ((merged["norm_minilm"] + merged["norm_e5"]) / 2.0).astype(float)
        merged["best_match_summary"] = merged.apply(
            lambda row: f"{row['id']} / {row['best_match_location']}" if str(row["best_match_location"]).strip() else str(row["id"]),
            axis=1,
        )
        merged["adaptive_search_reason"] = weights.reasoning
        merged["bm25_score"] = merged["bm25_final"]
        merged["minilm_score"] = merged["minilm_final"]
        merged["e5_score"] = merged["e5_final"]
        merged["normalized_bm25"] = merged["norm_bm25"]
        merged["normalized_minilm"] = merged["norm_minilm"]
        merged["normalized_e5"] = merged["norm_e5"]
        merged["fused_score"] = merged["fused_score"]
        return merged

