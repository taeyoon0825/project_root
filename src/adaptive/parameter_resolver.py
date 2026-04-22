from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.adaptive.dataset_profile import DatasetProfile, build_dataset_profile
from src.adaptive.normalization_resources import NormalizationResources, build_normalization_resources
from src.adaptive.performance_profile import PerformanceProfile, load_recent_performance_profile
from src.adaptive.query_features import QueryFeatureVector, extract_query_features
from src.adaptive.tuning_config import load_tuning_status
from src.config import ADAPTIVE_DIR, STATIC_REFERENCE_BASELINE_JSON
from src.search.text_source import DEFAULT_TEXT_SOURCE
from src.utils.io_utils import save_json


REPRODUCIBILITY_SEED = 42


@dataclass
class SearchWeightConfig:
    field_weights: dict[str, float]
    field_weight_total: float
    keyword_alpha: float
    dense_alpha: float
    keyword_ranker_weight: float
    dense_semantic_weight: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryAdaptiveConfig:
    query: str
    field_weights: dict[str, float]
    keyword_alpha: float
    dense_alpha: float
    keyword_ranker_weight: float
    dense_semantic_weight: float
    preview_length: int
    recommended_top_k: int
    feature_vector: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LanguageConfig:
    dominant_language: str
    whisper_language: str | None
    whisper_model: str
    tts_provider: str
    edge_voice: str
    gtts_lang: str
    sample_rate: int | None
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricConfig:
    hallucination_threshold: float
    soft_accuracy_weights: dict[str, float]
    soft_precision_exact_weight: float
    soft_recall_exact_weight: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterConfig:
    n_clusters: int
    min_cluster_size: int
    preferred_method: str
    candidate_scores: list[dict[str, float | int]]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualizationConfig:
    tsne_perplexity: float
    umap_n_neighbors: int
    umap_min_dist: float
    preview_length: int
    random_seed: int
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveContext:
    profile: DatasetProfile
    search: SearchWeightConfig
    language: LanguageConfig
    metric: MetricConfig
    cluster: ClusterConfig
    visualization: VisualizationConfig
    performance: PerformanceProfile
    normalization: NormalizationResources
    tuning_status: dict[str, Any]
    text_source: str = DEFAULT_TEXT_SOURCE
    artifact_namespace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "search": self.search.to_dict(),
            "language": self.language.to_dict(),
            "metric": self.metric.to_dict(),
            "cluster": self.cluster.to_dict(),
            "visualization": self.visualization.to_dict(),
            "performance": self.performance.to_dict(),
            "normalization": self.normalization.to_dict(),
            "tuning_status": self.tuning_status,
            "text_source": self.text_source,
            "artifact_namespace": self.artifact_namespace,
        }


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _normalize_weights(values: dict[str, float]) -> dict[str, float]:
    clean = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(clean.values())
    if total <= 0.0:
        uniform = 1.0 / max(1, len(clean))
        return {key: uniform for key in clean}
    return {key: value / total for key, value in clean.items()}


def _score_margin(scores: np.ndarray | list[float] | None) -> float:
    values = np.asarray(scores if scores is not None else [], dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    ordered = np.sort(values)
    top1 = float(ordered[-1])
    topn_index = max(0, len(ordered) - min(5, len(ordered)))
    pivot = float(ordered[topn_index])
    span = abs(top1) + abs(pivot) + 1e-9
    return _clip((top1 - pivot) / span)


def resolve_search_weights(profile: DatasetProfile, performance: PerformanceProfile | None = None) -> SearchWeightConfig:
    performance = performance or PerformanceProfile()
    field_scores = {
        field: float(
            np.mean(
                [
                    profile.field_query_overlap.get(field, 0.0),
                    1.0 - profile.field_missing_rates.get(field, 1.0),
                    profile.field_information_density.get(field, 0.0),
                    profile.field_retrieval_gain.get(field, 0.0),
                    profile.lexical_field_quality.get(field, 0.0),
                ]
            )
        )
        for field in profile.field_query_overlap
    }
    normalized_fields = _normalize_weights(field_scores)
    field_weight_total = float(
        1.0
        + math.log1p(max(1, profile.document_count))
        * np.mean(
            [
                profile.exact_match_ratio,
                profile.query_document_overlap,
                profile.corpus_token_coverage,
                1.0 - profile.field_missing_rates.get("transcript", 1.0),
            ]
        )
    )

    lexical_signal = float(
        np.mean(
            [
                profile.exact_match_ratio,
                profile.query_document_overlap,
                np.mean(list(profile.lexical_field_quality.values()) or [0.0]),
                profile.corpus_token_coverage,
            ]
        )
    )
    semantic_signal = float(
        np.mean(
            [
                profile.semantic_need_score,
                1.0 - profile.stt_quality_score,
                profile.embedding_profile.dense_separation_score if profile.embedding_profile else 0.0,
                performance.reranker_value_prior,
            ]
        )
    )
    alpha_pair = _normalize_weights({"keyword": lexical_signal, "dense": semantic_signal})
    keyword_alpha = alpha_pair["keyword"]
    dense_alpha = alpha_pair["dense"]
    corpus_scale = math.log1p(max(1, profile.document_count))
    keyword_ranker_weight = 1.0 + (corpus_scale * keyword_alpha)
    dense_semantic_weight = 1.0 + (corpus_scale * dense_alpha)
    reasoning = (
        f"field_scores={json.dumps(field_scores, ensure_ascii=False)}, lexical_signal={lexical_signal:.3f}, "
        f"semantic_signal={semantic_signal:.3f}, performance_prior={performance.reranker_value_prior:.3f}"
    )
    return SearchWeightConfig(
        field_weights={field: normalized_fields[field] * field_weight_total for field in normalized_fields},
        field_weight_total=field_weight_total,
        keyword_alpha=keyword_alpha,
        dense_alpha=dense_alpha,
        keyword_ranker_weight=keyword_ranker_weight,
        dense_semantic_weight=dense_semantic_weight,
        reasoning=reasoning,
    )


def resolve_query_search_config(
    query: str,
    context: AdaptiveContext | DatasetProfile,
    *,
    keyword_scores: np.ndarray | list[float] | None = None,
    dense_scores: np.ndarray | list[float] | None = None,
) -> QueryAdaptiveConfig:
    if isinstance(context, AdaptiveContext):
        profile = context.profile
        base = context.search
        visualization = context.visualization
        normalization = context.normalization
        performance = context.performance
    else:
        profile = context
        performance = load_recent_performance_profile()
        normalization = build_normalization_resources(
            metadata=pd.DataFrame(),
            queryset=None,
            stt_quality_score=profile.stt_quality_score,
            corpus_token_coverage=profile.corpus_token_coverage,
        )
        base = resolve_search_weights(profile, performance)
        visualization = resolve_visualization_config(profile)

    features = extract_query_features(query, profile, normalization, performance)
    keyword_margin = _score_margin(keyword_scores)
    dense_margin = _score_margin(dense_scores)

    keyword_signal = float(
        np.mean(
            [
                base.keyword_alpha,
                features.lexical_precision,
                1.0 - features.semantic_need,
                keyword_margin,
                features.exact_affinity,
            ]
        )
    )
    dense_signal = float(
        np.mean(
            [
                base.dense_alpha,
                features.semantic_need,
                features.question_likeness,
                features.spoken_style,
                dense_margin,
                features.paraphrase_affinity,
                features.natural_affinity,
                features.stt_affinity,
                features.reranker_value_signal,
            ]
        )
    )
    alpha_pair = _normalize_weights({"keyword": keyword_signal, "dense": dense_signal})
    keyword_alpha = alpha_pair["keyword"]
    dense_alpha = alpha_pair["dense"]

    base_normalized = _normalize_weights(base.field_weights)
    field_bias = {
        "title": float(
            np.mean(
                [
                    profile.field_query_overlap.get("title", 0.0),
                    profile.lexical_field_quality.get("title", 0.0),
                    features.lexical_precision,
                    features.named_entity_score,
                ]
            )
        ),
        "tags": float(
            np.mean(
                [
                    profile.field_query_overlap.get("tags", 0.0),
                    profile.lexical_field_quality.get("tags", 0.0),
                    features.lexical_precision,
                    features.numeric_salience,
                ]
            )
        ),
        "description": float(
            np.mean(
                [
                    profile.field_query_overlap.get("description", 0.0),
                    profile.lexical_field_quality.get("description", 0.0),
                    features.question_likeness,
                    features.semantic_need,
                ]
            )
        ),
        "transcript": float(
            np.mean(
                [
                    profile.field_query_overlap.get("transcript", 0.0),
                    profile.lexical_field_quality.get("transcript", 0.0),
                    features.spoken_style,
                    features.semantic_need,
                    features.stt_noise_score,
                ]
            )
        ),
    }
    raw_field_weights = {
        field: base_normalized.get(field, 0.0) * max(1e-9, field_bias.get(field, 0.0))
        for field in base_normalized
    }
    normalized_fields = _normalize_weights(raw_field_weights)
    field_weights = {field: normalized_fields[field] * base.field_weight_total for field in normalized_fields}

    corpus_scale = math.log1p(max(1, profile.document_count))
    keyword_ranker_weight = 1.0 + corpus_scale * float(np.mean([keyword_alpha, features.lexical_precision, keyword_margin]))
    dense_semantic_weight = 1.0 + corpus_scale * float(
        np.mean([dense_alpha, features.semantic_need, features.reranker_value_signal, dense_margin])
    )

    preview_length = resolve_preview_length(
        profile,
        query=query,
        search_mode="dense" if dense_alpha >= keyword_alpha else "keyword",
        ui_width_chars=visualization.preview_length,
    )
    recommended_top_k = resolve_top_k(profile, query_features=features)
    reasoning = (
        f"{features.reasoning}, keyword_margin={keyword_margin:.3f}, dense_margin={dense_margin:.3f}, "
        f"keyword_alpha={keyword_alpha:.3f}, dense_alpha={dense_alpha:.3f}"
    )
    return QueryAdaptiveConfig(
        query=query,
        field_weights=field_weights,
        keyword_alpha=keyword_alpha,
        dense_alpha=dense_alpha,
        keyword_ranker_weight=keyword_ranker_weight,
        dense_semantic_weight=dense_semantic_weight,
        preview_length=preview_length,
        recommended_top_k=recommended_top_k,
        feature_vector=features.to_dict(),
        reasoning=reasoning,
    )


def resolve_language_config(profile: DatasetProfile) -> LanguageConfig:
    dominant = profile.dominant_language
    if dominant == "ko":
        whisper_language, edge_voice, gtts_lang = "ko", "ko-KR-SunHiNeural", "ko"
    elif dominant == "en":
        whisper_language, edge_voice, gtts_lang = "en", "en-US-AriaNeural", "en"
    else:
        whisper_language, edge_voice, gtts_lang = None, "en-US-AriaNeural", "en"

    audio_signals = [
        profile.audio_profile.avg_duration_seconds / max(1.0, profile.audio_profile.avg_duration_seconds + 60.0),
        1.0 - profile.stt_quality_score,
        profile.document_count / max(1.0, profile.document_count + 24.0),
    ]
    whisper_complexity = float(np.mean(audio_signals))
    if whisper_complexity >= np.mean(audio_signals):
        whisper_model = "base" if whisper_complexity < 0.6 else "small"
    else:
        whisper_model = "tiny"

    resolved_sample_rate = int(profile.audio_profile.sample_rate_mode) if profile.audio_profile.sample_rate_mode > 0 else None
    reasoning = (
        f"dominant_language={dominant}, stt_quality={profile.stt_quality_score:.3f}, "
        f"avg_audio_duration={profile.audio_profile.avg_duration_seconds:.2f}"
    )
    return LanguageConfig(
        dominant_language=dominant,
        whisper_language=whisper_language,
        whisper_model=whisper_model,
        tts_provider="edge",
        edge_voice=edge_voice,
        gtts_lang=gtts_lang,
        sample_rate=resolved_sample_rate,
        reasoning=reasoning,
    )


def resolve_metric_config(
    profile: DatasetProfile,
    score_values: list[float] | np.ndarray | None = None,
    performance: PerformanceProfile | None = None,
) -> MetricConfig:
    performance = performance or PerformanceProfile()
    values = np.asarray(score_values if score_values is not None else [], dtype=np.float32)
    values = values[np.isfinite(values)]
    distribution_threshold = 0.0
    if values.size:
        median_score = float(np.median(values))
        upper_score = float(np.quantile(values, 0.75))
        distribution_threshold = max(median_score, upper_score)
    profile_threshold = 100.0 * float(
        np.mean(
            [
                profile.exact_match_ratio,
                profile.query_document_overlap,
                profile.stt_quality_score,
                1.0 - profile.semantic_need_score,
            ]
        )
    )
    if distribution_threshold > 0.0:
        threshold = float(np.mean([profile_threshold, distribution_threshold]))
    else:
        threshold = profile_threshold

    soft_weights = _normalize_weights(
        {
            "exact_match": float(np.mean([profile.exact_match_ratio, 1.0 - profile.semantic_need_score])),
            "search_score": float(
                np.mean(
                    [
                        profile.embedding_profile.dense_separation_score if profile.embedding_profile else 0.0,
                        performance.overall_semantic_success_rate,
                    ]
                )
            ),
            "text_overlap": float(np.mean([profile.query_document_overlap, profile.stt_quality_score])),
            "rank_weight": float(np.mean([1.0 / max(1.0, math.log1p(profile.document_count + 1.0)), performance.reranker_value_prior])),
        }
    )
    exact_weight_signal = float(np.mean([soft_weights["exact_match"], profile.exact_match_ratio]))
    precision_exact_weight = exact_weight_signal / max(
        exact_weight_signal + soft_weights["search_score"] + soft_weights["text_overlap"],
        1e-9,
    )
    recall_exact_weight = exact_weight_signal / max(exact_weight_signal + soft_weights["text_overlap"], 1e-9)
    reasoning = (
        f"threshold={threshold:.2f}, weights={json.dumps(soft_weights, ensure_ascii=False)}, "
        f"performance_success={performance.overall_semantic_success_rate:.3f}"
    )
    return MetricConfig(
        hallucination_threshold=threshold,
        soft_accuracy_weights=soft_weights,
        soft_precision_exact_weight=precision_exact_weight,
        soft_recall_exact_weight=recall_exact_weight,
        reasoning=reasoning,
    )


def resolve_cluster_config(profile: DatasetProfile, embeddings: np.ndarray | None = None) -> ClusterConfig:
    candidate_scores = profile.cluster_profile.candidate_scores if profile.cluster_profile else []
    recommended_k = profile.cluster_profile.recommended_k if profile.cluster_profile else 0
    if recommended_k <= 0:
        base_scale = math.sqrt(max(1, profile.document_count)) + math.log1p(max(1, profile.category_count))
        recommended_k = max(1, int(round(min(profile.document_count, base_scale))))
    min_cluster_size = max(1, int(round(math.sqrt(max(1, profile.document_count)))))
    preferred_method = "hdbscan" if (profile.cluster_profile and profile.cluster_profile.density_label == "dense") else "kmeans"
    reasoning = (
        f"recommended_k={recommended_k}, min_cluster_size={min_cluster_size}, "
        f"density={profile.cluster_profile.density_label if profile.cluster_profile else 'unknown'}"
    )
    return ClusterConfig(
        n_clusters=recommended_k,
        min_cluster_size=min_cluster_size,
        preferred_method=preferred_method,
        candidate_scores=candidate_scores,
        reasoning=reasoning,
    )


def resolve_visualization_config(profile: DatasetProfile, embeddings: np.ndarray | None = None) -> VisualizationConfig:
    sample_count = max(2, profile.document_count)
    density_factor = 1.0 + float(profile.cluster_profile.density_label == "dense") if profile.cluster_profile else 1.0
    tsne_perplexity = max(2.0, min(float(sample_count - 1), math.sqrt(sample_count) * density_factor))
    umap_n_neighbors = max(2, min(sample_count - 1, int(round(math.sqrt(sample_count) * density_factor))))
    umap_min_dist = float(np.mean([profile.semantic_need_score, 1.0 - profile.exact_match_ratio]))
    preview_length = resolve_preview_length(profile, search_mode="generic")
    reasoning = (
        f"tsne_perplexity={tsne_perplexity:.1f}, umap_n_neighbors={umap_n_neighbors}, "
        f"umap_min_dist={umap_min_dist:.2f}"
    )
    return VisualizationConfig(
        tsne_perplexity=tsne_perplexity,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        preview_length=preview_length,
        random_seed=REPRODUCIBILITY_SEED,
        reasoning=reasoning,
    )


def resolve_preview_length(
    profile: DatasetProfile,
    *,
    query: str | None = None,
    text: str | None = None,
    search_mode: str = "generic",
    ui_width_chars: int | None = None,
) -> int:
    query_token_count = len([token for token in str(query or "").split() if token.strip()])
    sentence_hint = profile.avg_document_length / max(1.0, profile.avg_sentence_count) if profile.avg_sentence_count else 0.0
    text_len = len(str(text or "").strip())
    mode_multiplier = {"keyword": 0.9, "generic": 1.0, "dense": 1.1}.get(search_mode, 1.0)
    base = (sentence_hint + (query_token_count * max(1.0, profile.avg_query_length or 1.0))) * mode_multiplier
    if ui_width_chars is not None:
        base = min(base or ui_width_chars, float(ui_width_chars))
    if text_len:
        base = min(base or text_len, float(text_len))
    if base <= 0:
        base = max(64.0, profile.avg_document_length / max(1.0, profile.avg_sentence_count or 1.0))
    return int(max(48, round(base)))


def resolve_top_k(
    profile: DatasetProfile,
    requested_top_k: int | None = None,
    *,
    query_features: QueryFeatureVector | None = None,
) -> int:
    base = math.sqrt(max(1, profile.document_count)) + math.log1p(max(1, profile.category_count + profile.query_count))
    pressure = query_features.candidate_pressure if query_features is not None else profile.semantic_need_score
    adaptive_top_k = max(1, int(round(min(profile.document_count, base * (1.0 + pressure)))))
    if requested_top_k is None:
        return adaptive_top_k
    return max(1, min(profile.document_count, int(requested_top_k)))


def _save_context(context: AdaptiveContext) -> None:
    if not context.artifact_namespace:
        return
    ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = ADAPTIVE_DIR / f"{context.artifact_namespace}__adaptive_context.json"
    save_json(path, context.to_dict())


def _resolve_effective_normalization_mode(
    profile: DatasetProfile,
    performance: PerformanceProfile,
    normalization: NormalizationResources,
) -> str:
    adaptive_score = float(
        np.mean(
            [
                normalization.normalization_preference,
                1.0 - profile.stt_quality_score,
                max(0.0, performance.bucket("natural_question").delta_semantic_success_rate),
                max(0.0, performance.bucket("stt_style").delta_semantic_success_rate),
                performance.reranker_value_prior,
            ]
        )
    )
    baseline_score = float(
        np.mean(
            [
                1.0 - normalization.normalization_preference,
                profile.stt_quality_score,
                performance.overall_semantic_success_rate,
                profile.corpus_token_coverage,
            ]
        )
    )
    normalization.recommended_mode = "adaptive_corpus" if adaptive_score >= baseline_score else "baseline"
    return normalization.recommended_mode


def build_static_reference_context(
    metadata,
    queryset=None,
    *,
    text_source: str = DEFAULT_TEXT_SOURCE,
    embedding_model_alias: str | None = None,
    embeddings: np.ndarray | None = None,
    artifact_namespace: str | None = None,
) -> AdaptiveContext:
    profile = build_dataset_profile(
        metadata,
        queryset,
        text_source=text_source,
        embedding_model_alias=embedding_model_alias,
        embeddings=embeddings,
    )
    baseline = {}
    if STATIC_REFERENCE_BASELINE_JSON.exists():
        try:
            baseline = json.loads(STATIC_REFERENCE_BASELINE_JSON.read_text(encoding="utf-8"))
        except Exception:
            baseline = {}
    performance = load_recent_performance_profile()
    normalization = build_normalization_resources(
        metadata,
        queryset,
        stt_quality_score=profile.stt_quality_score,
        corpus_token_coverage=profile.corpus_token_coverage,
    )
    _resolve_effective_normalization_mode(profile, performance, normalization)
    search = SearchWeightConfig(
        field_weights=baseline.get("search", {}).get("field_weights", resolve_search_weights(profile, performance).field_weights),
        field_weight_total=float(
            baseline.get("search", {}).get("field_weight_total", resolve_search_weights(profile, performance).field_weight_total)
        ),
        keyword_alpha=float(baseline.get("search", {}).get("keyword_alpha", 0.5)),
        dense_alpha=float(baseline.get("search", {}).get("dense_alpha", 0.5)),
        keyword_ranker_weight=float(baseline.get("search", {}).get("keyword_ranker_weight", 1.0)),
        dense_semantic_weight=float(baseline.get("search", {}).get("dense_semantic_weight", 1.0)),
        reasoning=str(baseline.get("search", {}).get("reasoning", "reference-only static baseline")),
    )
    language = resolve_language_config(profile)
    metric = resolve_metric_config(profile, performance=performance)
    cluster = resolve_cluster_config(profile, embeddings=embeddings)
    visualization = resolve_visualization_config(profile, embeddings=embeddings)
    return AdaptiveContext(
        profile=profile,
        search=search,
        language=language,
        metric=metric,
        cluster=cluster,
        visualization=visualization,
        performance=performance,
        normalization=normalization,
        tuning_status=load_tuning_status(),
        text_source=text_source,
        artifact_namespace=artifact_namespace,
    )


def build_adaptive_context(
    metadata,
    queryset=None,
    *,
    text_source: str = DEFAULT_TEXT_SOURCE,
    embedding_model_alias: str | None = None,
    embeddings: np.ndarray | None = None,
    artifact_namespace: str | None = None,
) -> AdaptiveContext:
    profile = build_dataset_profile(
        metadata,
        queryset,
        text_source=text_source,
        embedding_model_alias=embedding_model_alias,
        embeddings=embeddings,
    )
    performance = load_recent_performance_profile()
    normalization = build_normalization_resources(
        metadata,
        queryset,
        stt_quality_score=profile.stt_quality_score,
        corpus_token_coverage=profile.corpus_token_coverage,
    )
    _resolve_effective_normalization_mode(profile, performance, normalization)
    search = resolve_search_weights(profile, performance)
    language = resolve_language_config(profile)
    metric = resolve_metric_config(profile, performance=performance)
    cluster = resolve_cluster_config(profile, embeddings=embeddings)
    visualization = resolve_visualization_config(profile, embeddings=embeddings)
    context = AdaptiveContext(
        profile=profile,
        search=search,
        language=language,
        metric=metric,
        cluster=cluster,
        visualization=visualization,
        performance=performance,
        normalization=normalization,
        tuning_status=load_tuning_status(),
        text_source=text_source,
        artifact_namespace=artifact_namespace,
    )
    _save_context(context)
    return context
