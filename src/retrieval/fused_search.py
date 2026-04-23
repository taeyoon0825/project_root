from __future__ import annotations

"""어휘 검색, 밀집 검색, 리랭커를 결합하는 융합 검색 엔진.

이 모듈은 여러 검색기의 점수를 한 테이블로 모으고, 질의 특성에 따라
가중치를 조정하고, 필요하면 리랭커까지 적용해 최종 순위를 만든다.
각 단계가 왜 존재하는지 코드 가까이에 남겨 두어야 점수 해석과 디버깅이 가능하다.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.adaptive.parameter_resolver import AdaptiveContext, QueryAdaptiveConfig, resolve_query_search_config, resolve_top_k
from src.adaptive.query_features import QueryFeatureVector, extract_query_features
from src.adaptive.tuning_config import load_tuning_config
from src.embedding.build_indices import DenseSearchEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.search.keyword_search import KeywordSearchEngine
from src.search.text_source import (
    DEFAULT_DENSE_NORMALIZATION_MODE,
    build_preview_text,
    build_search_text,
    normalize_text_for_reranker,
    prepare_query_for_reranker,
    resolve_dense_normalization_mode,
    resolve_primary_text,
)


_TUNING = load_tuning_config().get("fused", {})
_TUNING_RERANK = _TUNING.get("reranker", {}) if isinstance(_TUNING, dict) else {}


def _normalize_minmax(values: pd.Series) -> pd.Series:
    """최솟값/최댓값 기준으로 점수를 0..1 범위에 맞춘다."""
    if values.empty:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _normalize_rank(values: pd.Series) -> pd.Series:
    """값의 상대적 순위만 보존하는 순위 기반 정규화를 계산한다."""
    if values.empty:
        return values
    raw = values.astype(float).to_numpy()
    # 모든 점수가 같은 경우 순위 정규화를 강제로 0으로 두어
    # 신호가 없는 후보가 가짜 상대 우위를 얻지 않게 한다.
    if raw.size == 0 or np.allclose(raw, raw[0], atol=1e-12, rtol=0.0):
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    ranked = values.rank(method="average", ascending=False)
    return (1.0 - ((ranked - 1.0) / max(1.0, len(values) - 1.0))).astype(np.float32)


def _score_margin(values: pd.Series | np.ndarray | list[float]) -> float:
    """상위 후보와 나머지 후보의 분리 정도를 0..1 값으로 요약한다."""
    array = np.asarray(values, dtype=np.float32)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    ordered = np.sort(array)
    top1 = float(ordered[-1])
    pivot = float(ordered[max(0, len(ordered) - min(5, len(ordered)))])
    span = abs(top1) + abs(pivot) + 1e-9
    return float(max(0.0, min(1.0, (top1 - pivot) / span)))


def _hybrid_normalize(values: pd.Series) -> pd.Series:
    """분포 형태에 따라 min-max와 rank 정규화를 혼합한다.

    점수 분포가 넓고 의미 있는 경우에는 실제 간격을 살리고,
    분포가 납작한 경우에는 순위 정보 비중을 높여 모델 간 스케일 차이를 완화한다.
    """
    if values.empty:
        return values
    raw = values.astype(float).to_numpy()
    # 입력이 완전히 평평하면 정규화 결과도 0으로 두어
    # 신호 없는 엔진이 융합 점수에 기본점수처럼 기여하지 않게 한다.
    if raw.size == 0 or np.allclose(raw, raw[0], atol=1e-12, rtol=0.0):
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    minmax = _normalize_minmax(values)
    rank = _normalize_rank(values)
    span = float(np.ptp(raw)) if raw.size else 0.0
    std = float(np.std(raw)) if raw.size else 0.0
    mean_abs = float(np.mean(np.abs(raw))) if raw.size else 0.0
    distribution_signal = 0.0 if (span + std + mean_abs) <= 0 else (span + std) / (span + std + mean_abs)
    minmax_weight = max(0.0, min(1.0, distribution_signal))
    rank_weight = 1.0 - minmax_weight

    override = _TUNING.get("hybrid_normalize", {}) if isinstance(_TUNING, dict) else {}
    if isinstance(override, dict) and {"minmax_weight", "rank_weight"} <= set(override):
        override_total = max(
            1e-9,
            float(override.get("minmax_weight", 0.0) or 0.0) + float(override.get("rank_weight", 0.0) or 0.0),
        )
        override_minmax = float(override.get("minmax_weight", 0.0) or 0.0) / override_total
        override_rank = float(override.get("rank_weight", 0.0) or 0.0) / override_total
        minmax_weight = float(np.mean([minmax_weight, override_minmax]))
        rank_weight = float(np.mean([rank_weight, override_rank]))

    return ((minmax_weight * minmax) + (rank_weight * rank)).astype(np.float32)


def _query_tokens(query: str) -> list[str]:
    """제목/키워드 보너스와 anchor overlap 계산에 쓸 토큰 집합을 만든다."""
    return re.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", str(query or "").lower())


def _token_overlap_ratio(text: str, query_tokens: list[str]) -> float:
    """주어진 텍스트가 질의 토큰을 얼마나 직접 포함하는지 비율로 측정한다."""
    if not query_tokens:
        return 0.0
    text_tokens = set(_query_tokens(text))
    if not text_tokens:
        return 0.0
    return float(sum(1 for token in query_tokens if token in text_tokens) / max(1, len(query_tokens)))


def _token_salience_overlap(text: str, token_salience: dict[str, float]) -> float:
    """질의에서 중요한 토큰이 텍스트에 얼마나 포함됐는지 가중 비율로 계산한다."""
    if not token_salience:
        return 0.0
    text_tokens = set(_query_tokens(text))
    if not text_tokens:
        return 0.0
    matched = sum(weight for token, weight in token_salience.items() if token in text_tokens)
    total = sum(token_salience.values()) or 1.0
    return float(matched / total)


def _mean_score(frame: pd.DataFrame, column: str) -> float:
    """점수 컬럼 평균을 안전하게 읽는 작은 도우미."""
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(frame[column].astype(float).mean())


@dataclass
class FusionWeights:
    """융합 점수 계산에 사용할 최종 모델별 가중치 묶음."""

    bm25: float
    minilm: float
    e5: float
    semantic_push: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """디버그와 아티팩트 저장용으로 dataclass를 사전으로 바꾼다."""
        return {
            "bm25": self.bm25,
            "minilm": self.minilm,
            "e5": self.e5,
            "semantic_push": self.semantic_push,
            "reasoning": self.reasoning,
        }


def resolve_fusion_weights(
    query: str,
    context: AdaptiveContext,
    *,
    bm25_scores: pd.Series,
    minilm_scores: pd.Series,
    e5_scores: pd.Series,
    query_features: QueryFeatureVector,
) -> tuple[FusionWeights, QueryAdaptiveConfig]:
    """질의 특성과 각 엔진의 점수 분포를 바탕으로 융합 가중치를 계산한다.

    같은 질의라도 정확 일치형인지, 의역 검색형인지, STT 노이즈가 많은지에 따라
    BM25와 dense 모델의 비중이 달라져야 하므로 이 로직이 융합 엔진의 핵심이다.
    """
    query_cfg = resolve_query_search_config(
        query,
        context,
        keyword_scores=bm25_scores.to_numpy(),
        dense_scores=((minilm_scores + e5_scores) / 2.0).to_numpy(),
    )
    bm25_margin = _score_margin(bm25_scores)
    minilm_margin = _score_margin(minilm_scores)
    e5_margin = _score_margin(e5_scores)

    lexical_signal = float(
        np.mean(
            [
                query_cfg.keyword_alpha,
                query_features.lexical_precision,
                query_features.exact_affinity,
                context.profile.exact_match_ratio,
                bm25_margin,
                _mean_score(pd.DataFrame({"score": bm25_scores}), "score"),
            ]
        )
    )
    semantic_signal = float(
        np.mean(
            [
                query_cfg.dense_alpha,
                query_features.semantic_need,
                query_features.question_likeness,
                query_features.spoken_style,
                query_features.natural_affinity,
                query_features.stt_affinity,
                query_features.reranker_value_signal,
                context.profile.semantic_need_score,
                context.performance.reranker_value_prior,
            ]
        )
    )
    bm25_raw = float(np.mean([lexical_signal, bm25_margin, query_features.exact_affinity]))
    minilm_raw = float(
        np.mean(
            [
                semantic_signal,
                minilm_margin,
                query_features.paraphrase_affinity,
                context.profile.embedding_profile.dense_preference_score if context.profile.embedding_profile else 0.0,
            ]
        )
    )
    e5_raw = float(
        np.mean(
            [
                semantic_signal,
                e5_margin,
                query_features.natural_affinity,
                context.profile.embedding_profile.dense_separation_score if context.profile.embedding_profile else 0.0,
            ]
        )
    )
    # 엔진별 원시 신호를 같은 의미의 양수 가중치 공간으로 모은 뒤
    # 마지막에 합이 1이 되도록 정규화한다.
    normalized = {
        key: value
        for key, value in {
            "bm25": bm25_raw,
            "minilm": minilm_raw,
            "e5": e5_raw,
        }.items()
    }
    weights = normalized if sum(normalized.values()) > 0 else {"bm25": 1.0, "minilm": 1.0, "e5": 1.0}
    weights = {key: value / sum(weights.values()) for key, value in weights.items()}
    semantic_push = float(
        np.mean(
            [
                semantic_signal,
                1.0 - lexical_signal,
                query_features.natural_affinity,
                query_features.stt_affinity,
                query_features.reranker_value_signal,
                query_features.question_likeness,
            ]
        )
    )
    reasoning = (
        f"{query_cfg.reasoning}; lexical_signal={lexical_signal:.3f}, semantic_signal={semantic_signal:.3f}, "
        f"margins=({bm25_margin:.3f}/{minilm_margin:.3f}/{e5_margin:.3f})"
    )
    return (
        FusionWeights(
            bm25=weights["bm25"],
            minilm=weights["minilm"],
            e5=weights["e5"],
            semantic_push=semantic_push,
            reasoning=reasoning,
        ),
        query_cfg,
    )


class FusedSearchEngine:
    """키워드, MiniLM, E5, 리랭커를 묶어 최종 순위를 만드는 검색 엔진."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        *,
        text_source: str,
        artifact_namespace: str | None,
        adaptive_context: AdaptiveContext,
        minilm_alias: str = "paraphrase-multilingual-MiniLM-L12-v2",
        e5_alias: str = "multilingual-e5-base",
        dense_normalization_mode: str | None = None,
        enable_reranker: bool | None = None,
    ):
        """융합 검색에 필요한 하위 엔진과 공통 정규화 컨텍스트를 준비한다."""
        self.metadata = metadata.copy()
        self.text_source = text_source
        self.artifact_namespace = artifact_namespace
        self.context = adaptive_context
        self.normalization_resources = adaptive_context.normalization
        self.dense_normalization_mode = resolve_dense_normalization_mode(
            dense_normalization_mode,
            resources=self.normalization_resources,
        )
        self.reranker_enabled = bool(enable_reranker) if enable_reranker is not None else True
        if "search_text" not in self.metadata.columns:
            # 융합 단계도 하위 dense 엔진과 같은 검색 텍스트를 봐야 하므로
            # 메타데이터에 없다면 여기서 미리 만들어 둔다.
            self.metadata["search_text"] = self.metadata.apply(
                lambda row: build_search_text(
                    row,
                    text_source=text_source,
                    for_dense=True,
                    normalization_mode=self.dense_normalization_mode,
                    resources=self.normalization_resources,
                ),
                axis=1,
            )
        self.keyword = KeywordSearchEngine(
            metadata,
            text_source=text_source,
            adaptive_context=adaptive_context,
            artifact_namespace=artifact_namespace,
        )
        self.minilm = DenseSearchEngine(
            metadata,
            minilm_alias,
            text_source=text_source,
            artifact_namespace=artifact_namespace,
            adaptive_context=adaptive_context,
            dense_normalization_mode=self.dense_normalization_mode,
        )
        self.e5 = DenseSearchEngine(
            metadata,
            e5_alias,
            text_source=text_source,
            artifact_namespace=artifact_namespace,
            adaptive_context=adaptive_context,
            dense_normalization_mode=self.dense_normalization_mode,
        )
        self.minilm.load()
        self.e5.load()
        model_name = str(_TUNING_RERANK.get("model_name") or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.reranker = CrossEncoderReranker(model_name)

    def _join_scores(self, bm25: pd.DataFrame, mini: pd.DataFrame, e5: pd.DataFrame) -> pd.DataFrame:
        """하위 엔진 결과를 문서 ID 기준 하나의 테이블로 합친다."""
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

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """리랭커 입력 길이를 자르되 가능하면 단어 경계에서 끊는다."""
        value = str(text or "").strip()
        if len(value) <= max_chars:
            return value
        truncated = value[:max_chars].rsplit(" ", 1)[0].strip()
        return truncated or value[:max_chars]

    def _build_rerank_candidate_text(
        self,
        row: pd.Series,
        query_cfg: QueryAdaptiveConfig,
        query_features: QueryFeatureVector,
    ) -> str:
        """리랭커가 읽을 문서 요약 텍스트를 구성한다.

        제목, 설명, 키워드, 매칭된 passage, 대표 전사문을 한 텍스트로 묶어
        리랭커가 순위 재조정을 위한 근거를 충분히 볼 수 있게 한다.
        """
        preview_chars = max(query_cfg.preview_length, int(round(query_cfg.preview_length * (1.0 + query_features.semantic_need))))
        matched_passages: list[str] = []
        sections: list[str] = []

        def _add_section(label: str, value: str) -> None:
            normalized = normalize_text_for_reranker(
                self._truncate_text(value, preview_chars),
                mode=self.dense_normalization_mode,
                resources=self.normalization_resources,
            )
            if normalized:
                sections.append(f"{label}: {normalized}")

        _add_section("title", str(row.get("title", "") or ""))
        _add_section("description", str(row.get("description", "") or ""))
        _add_section(
            "keywords",
            " ".join(
                part
                for part in [
                    str(row.get("keywords", "") or ""),
                    str(row.get("tags", "") or ""),
                    str(row.get("category", "") or ""),
                ]
                if part
            ),
        )

        for column in ["bm25_best_match_text", "minilm_best_match_text", "e5_best_match_text"]:
            passage = normalize_text_for_reranker(
                str(row.get(column, "") or ""),
                mode=self.dense_normalization_mode,
                resources=self.normalization_resources,
            )
            if passage and passage not in matched_passages:
                matched_passages.append(passage)
        if matched_passages:
            # 의미 필요도가 높을수록 매칭 passage를 조금 더 넣어
            # 리랭커가 문서 내부 근거를 넓게 볼 수 있게 한다.
            passage_count = max(1, int(round(len(matched_passages) * max(0.2, query_features.semantic_need))))
            sections.append("matched_passages: " + " | ".join(matched_passages[:passage_count]))

        _add_section("transcript", build_preview_text(row, text_source=self.text_source, length=preview_chars))
        combined_text = resolve_primary_text(row, text_source="combined")
        primary_text = resolve_primary_text(row, text_source=self.text_source)
        if combined_text and combined_text != primary_text:
            _add_section("combined_transcript", combined_text)
        return "\n".join(sections)

    def _resolve_candidate_pool_size(self, resolved_top_k: int, query_features: QueryFeatureVector) -> int:
        """질의 난이도와 코퍼스 크기에 따라 1차 후보 풀 크기를 조절한다."""
        doc_count = max(1, int(self.context.profile.document_count))
        corpus_signal = math.log1p(doc_count) / max(1.0, math.log1p(doc_count) + math.sqrt(doc_count))
        pool_signal = float(
            np.mean(
                [
                    query_features.candidate_pressure,
                    query_features.semantic_need,
                    query_features.ambiguity,
                    query_features.natural_affinity,
                    query_features.stt_affinity,
                    query_features.reranker_value_signal,
                    corpus_signal,
                ]
            )
        )
        candidate_k = resolved_top_k + int(round((doc_count - resolved_top_k) * min(1.0, pool_signal)))
        return max(resolved_top_k, min(doc_count, candidate_k))

    def _resolve_query_normalization_mode(self, query_features: QueryFeatureVector) -> str:
        """질의 특성이 구어체/STT형이면 더 공격적인 정규화 모드로 전환한다."""
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
        baseline_score = float(np.mean([query_features.lexical_precision, self.context.profile.stt_quality_score]))
        return "adaptive_corpus" if adaptive_score >= baseline_score else self.dense_normalization_mode

    def _model_disagreement(self, merged: pd.DataFrame) -> float:
        """세 하위 모델의 top1 불일치 정도를 0..1로 측정한다."""
        if merged.empty:
            return 0.0
        top_ids = [
            str(merged.sort_values("norm_bm25", ascending=False).iloc[0]["id"]),
            str(merged.sort_values("norm_minilm", ascending=False).iloc[0]["id"]),
            str(merged.sort_values("norm_e5", ascending=False).iloc[0]["id"]),
        ]
        return float((len(set(top_ids)) - 1) / max(1, len(top_ids) - 1))

    def _resolve_rerank_depth(
        self,
        candidate_k: int,
        resolved_top_k: int,
        query_features: QueryFeatureVector,
        merged: pd.DataFrame,
    ) -> int:
        """질의 가치와 모델 불확실성을 바탕으로 리랭킹 깊이를 정한다."""
        pool_span = max(0, candidate_k - resolved_top_k)
        disagreement = self._model_disagreement(merged)
        uncertainty = 1.0 - _score_margin(merged["fused_score"])
        depth_signal = float(
            np.mean(
                [
                    query_features.candidate_pressure,
                    query_features.reranker_value_signal,
                    query_features.natural_affinity,
                    query_features.stt_affinity,
                    disagreement,
                    uncertainty,
                ]
            )
        )
        return max(resolved_top_k, min(candidate_k, resolved_top_k + int(round(pool_span * depth_signal))))

    def _resolve_rerank_alpha(
        self,
        query_features: QueryFeatureVector,
        weights: FusionWeights,
        base_scores: pd.Series,
        rerank_scores: pd.Series,
    ) -> float:
        """기본 융합 점수와 리랭커 점수를 얼마나 섞을지 결정한다."""
        base_uncertainty = 1.0 - _score_margin(base_scores)
        rerank_confidence = _score_margin(rerank_scores)
        disagreement_gain = 0.0
        if not base_scores.empty and not rerank_scores.empty:
            base_winner = str(base_scores.idxmax())
            rerank_winner = str(rerank_scores.idxmax())
            if base_winner != rerank_winner:
                disagreement_gain = float(
                    max(
                        0.0,
                        float(rerank_scores.max()) - float(rerank_scores.get(base_winner, 0.0) or 0.0),
                    )
                )
        alpha = float(
            np.mean(
                [
                    weights.semantic_push,
                    query_features.semantic_need,
                    query_features.question_likeness,
                    query_features.spoken_style,
                    query_features.natural_affinity,
                    query_features.stt_affinity,
                    1.0 - query_features.lexical_precision,
                    query_features.reranker_value_signal,
                    base_uncertainty,
                    rerank_confidence,
                    disagreement_gain,
                ]
            )
        )
        alpha = 1.0 - ((1.0 - alpha) * (1.0 - disagreement_gain) * (1.0 - rerank_confidence))
        alpha_scale = _TUNING_RERANK.get("alpha_scale")
        if alpha_scale not in (None, ""):
            try:
                alpha = float(np.mean([alpha, alpha * float(alpha_scale)]))
            except (TypeError, ValueError):
                pass
        return float(max(0.0, min(1.0, alpha)))

    def _fallback_info(self) -> tuple[int, str]:
        """튜닝/성능/정규화 fallback 사유를 한 문자열로 모은다."""
        reasons: list[str] = []
        if bool(self.context.tuning_status.get("used_safe_fallback", False)):
            reasons.append(str(self.context.tuning_status.get("fallback_reason", "")))
        if self.context.performance.used_fallback_stats:
            reasons.append(str(self.context.performance.fallback_reason))
        if self.context.normalization.used_fallback_resources:
            reasons.append(str(self.context.normalization.fallback_reason))
        reason = ";".join(reason for reason in reasons if reason)
        return int(bool(reason)), reason

    def search(self, query: str, top_k: int | None = None, keyword_method: str = "bm25") -> pd.DataFrame:
        """융합 검색 전체 파이프라인을 실행해 최종 ranked DataFrame을 반환한다."""
        query_features = extract_query_features(
            query,
            self.context.profile,
            self.normalization_resources,
            self.context.performance,
        )
        query_normalization_mode = self._resolve_query_normalization_mode(query_features)
        resolved_top_k = resolve_top_k(self.context.profile, top_k, query_features=query_features)
        candidate_k = self._resolve_candidate_pool_size(resolved_top_k, query_features)
        # 하위 엔진은 같은 질의로 각자 top-k보다 넓은 후보 풀을 만들고,
        # 이후 융합 단계가 그 후보들을 다시 평가한다.
        bm25 = self.keyword.search(query, top_k=candidate_k, method=keyword_method)
        minilm = self.minilm.search(query, top_k=candidate_k)
        e5 = self.e5.search(query, top_k=candidate_k)

        merged = self._join_scores(bm25, minilm, e5)
        merged["norm_bm25"] = _hybrid_normalize(merged["bm25_final"])
        merged["norm_minilm"] = _hybrid_normalize(merged["minilm_final"])
        merged["norm_e5"] = _hybrid_normalize(merged["e5_final"])

        weights, query_cfg = resolve_fusion_weights(
            query,
            self.context,
            bm25_scores=merged["norm_bm25"],
            minilm_scores=merged["norm_minilm"],
            e5_scores=merged["norm_e5"],
            query_features=query_features,
        )
        # 모델별 정규화 점수와 적응형 가중치를 곱해 1차 융합 점수를 만든다.
        merged["fused_score"] = (
            weights.bm25 * merged["norm_bm25"]
            + weights.minilm * merged["norm_minilm"]
            + weights.e5 * merged["norm_e5"]
        )

        normalized_query = prepare_query_for_reranker(
            query,
            mode=query_normalization_mode,
            resources=self.normalization_resources,
        )
        query_tokens = _query_tokens(normalized_query or query)
        # 제목/키워드 직접 일치는 사용자 체감 품질에 큰 영향을 주므로
        # 별도 overlap 보너스로 약하게 반영한다.
        title_bonus_scale = float(
            np.mean(
                [
                    query_features.lexical_precision,
                    self.context.profile.field_query_overlap.get("title", 0.0),
                    self.context.profile.lexical_field_quality.get("title", 0.0),
                ]
            )
        )
        keyword_bonus_scale = float(
            np.mean(
                [
                    query_features.lexical_precision,
                    query_features.numeric_salience,
                    self.context.profile.field_query_overlap.get("tags", 0.0),
                    self.context.profile.lexical_field_quality.get("tags", 0.0),
                ]
            )
        )
        merged["title_overlap_ratio"] = merged["title"].fillna("").astype(str).apply(lambda value: _token_overlap_ratio(value, query_tokens))
        merged["keyword_overlap_ratio"] = merged["keywords"].fillna("").astype(str).apply(
            lambda value: _token_overlap_ratio(value, query_tokens)
        )
        merged["fused_score"] = merged["fused_score"] + (title_bonus_scale * merged["title_overlap_ratio"]) + (
            keyword_bonus_scale * merged["keyword_overlap_ratio"]
        )

        merged["reranker_score"] = 0.0
        merged["reranker_alpha"] = 0.0
        merged["rerank_anchor_overlap"] = 0.0
        rerank_top_n = resolved_top_k
        if self.reranker_enabled:
            # 리랭커는 비용이 크므로 상위 후보 일부에만 적용한다.
            rerank_top_n = self._resolve_rerank_depth(candidate_k, resolved_top_k, query_features, merged)
            rerank_frame = merged.sort_values("fused_score", ascending=False).head(rerank_top_n).copy()
            rerank_idx = rerank_frame.index
            rerank_frame["rerank_candidate_text"] = rerank_frame.apply(
                lambda row: self._build_rerank_candidate_text(row, query_cfg, query_features),
                axis=1,
            )
            rerank_scores = self.reranker.score(normalized_query or query, rerank_frame["rerank_candidate_text"].tolist())
            if len(rerank_scores) == len(rerank_idx):
                rerank_score_series = pd.Series(rerank_scores, index=rerank_idx, dtype=np.float32)
                rerank_base_scores = _hybrid_normalize(rerank_frame["fused_score"])
                anchor_overlap = rerank_frame["rerank_candidate_text"].apply(
                    lambda text: _token_salience_overlap(text, query_features.token_salience)
                )
                # 질의가 정확 일치형에 가까우면 리랭커 점수보다
                # salient token overlap을 더 많이 반영해 보수적으로 섞는다.
                rerank_score_series = (
                    ((1.0 - query_features.lexical_precision) * rerank_score_series)
                    + (query_features.lexical_precision * anchor_overlap.astype(np.float32))
                ).astype(np.float32)
                alpha = self._resolve_rerank_alpha(query_features, weights, rerank_base_scores, rerank_score_series)
                blended_scores = ((1.0 - alpha) * rerank_base_scores) + (alpha * rerank_score_series)
                merged.loc[rerank_idx, "reranker_score"] = rerank_score_series.to_numpy(dtype=np.float32)
                merged.loc[rerank_idx, "reranker_alpha"] = alpha
                merged.loc[rerank_idx, "rerank_anchor_overlap"] = anchor_overlap.to_numpy(dtype=np.float32)
                merged.loc[rerank_idx, "fused_score"] = blended_scores.to_numpy(dtype=np.float32)

        # 최종 fused_score는 UI와 평가가 읽기 쉽게 0..100 표현도 함께 만든다.
        merged["display_score"] = (_hybrid_normalize(merged["fused_score"]) * 100.0).astype(float)
        merged["similarity_score"] = merged["display_score"]

        def _choose_preview_reason(row: pd.Series) -> tuple[str, str]:
            """어느 하위 모델의 근거 문장을 미리보기로 쓸지 결정한다."""
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
            lambda row: str(
                row.get("bm25_best_match_location")
                or row.get("minilm_best_match_location")
                or row.get("e5_best_match_location")
                or ""
            ),
            axis=1,
        )
        merged["rank"] = merged["fused_score"].rank(method="first", ascending=False).astype(int)
        merged = merged.sort_values("fused_score", ascending=False).head(resolved_top_k).reset_index(drop=True)
        if "rank" in merged.columns:
            merged = merged.drop(columns=["rank"])
        merged.insert(0, "rank", merged.index + 1)

        used_fallback_tuning, fallback_reason = self._fallback_info()
        merged["adaptive_field_weights"] = str(query_cfg.field_weights)
        merged["adaptive_keyword_alpha"] = float(query_cfg.keyword_alpha)
        merged["adaptive_dense_alpha"] = float(query_cfg.dense_alpha)
        merged["adaptive_preview_length"] = int(query_cfg.preview_length)
        merged["adaptive_reason"] = query_cfg.reasoning
        merged["adaptive_query_features"] = json.dumps(query_cfg.feature_vector, ensure_ascii=False)
        merged["adaptive_candidate_pool_k"] = int(candidate_k)
        merged["adaptive_rerank_top_n"] = int(rerank_top_n)
        merged["adaptive_query_bucket"] = query_features.dominant_bucket
        merged["adaptive_semantic_need"] = float(query_features.semantic_need)
        merged["adaptive_ambiguity"] = float(query_features.ambiguity)
        merged["adaptive_reranker_value"] = float(query_features.reranker_value_signal)
        merged["fusion_weights"] = json.dumps(weights.to_dict(), ensure_ascii=False)
        merged["raw_score"] = merged["fused_score"].astype(float)
        merged["final_score"] = merged["fused_score"].astype(float)
        merged["raw_score_explanation"] = (
            "fused_score = adaptive fusion(bm25/minilm/e5) + adaptive title/keyword overlap + adaptive rerank blend; "
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
            lambda row: (
                f"bucket={query_features.dominant_bucket}, top_model={row['top_model_contribution']}, "
                f"reranker_alpha={float(row.get('reranker_alpha', 0.0) or 0.0):.3f}"
            ),
            axis=1,
        )
        # 융합 결과는 문서 단위 가중치 결합이 중심이므로
        # 어휘 엔진처럼 단일 matched_tokens 집합을 강제하지 않는다.
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
        merged["reranker_enabled"] = int(self.reranker_enabled)
        merged["dense_normalization_mode"] = self.dense_normalization_mode
        merged["query_normalization_mode"] = query_normalization_mode
        merged["used_fallback_tuning"] = used_fallback_tuning
        merged["fallback_reason"] = fallback_reason
        return merged
