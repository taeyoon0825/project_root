from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.adaptive.dataset_profile import ClusterProfile, DatasetProfile, build_dataset_profile
from src.config import ADAPTIVE_DIR
from src.search.text_source import DEFAULT_TEXT_SOURCE
from src.utils.io_utils import save_json


FIELD_BOUNDS = {
    "title": (0.10, 0.50),
    "tags": (0.10, 0.40),
    "description": (0.10, 0.40),
    "transcript": (0.10, 0.60),
}
PREVIEW_BOUNDS = (80, 240)
TOP_K_BOUNDS = (3, 15)
CLUSTER_BOUNDS = (2, 12)
REPRODUCIBILITY_SEED = 42
QUESTION_RE = re.compile(r"\?$|what|why|how|who|where|when|which|설명|알려|무엇|왜|어떻게|누가|어디|언제", re.IGNORECASE)


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
    reasoning: str

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
            "text_source": self.text_source,
            "artifact_namespace": self.artifact_namespace,
        }


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _waterfill_normalize(values: dict[str, float], bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    remaining = set(values)
    result = {key: 0.0 for key in values}
    target_total = 1.0
    raw = values.copy()

    while remaining:
        raw_total = sum(max(raw[key], 0.0) for key in remaining)
        if raw_total <= 0:
            share = target_total / max(1, len(remaining))
            for key in list(remaining):
                lower, upper = bounds[key]
                value = _clamp(share, lower, upper)
                result[key] = value
                target_total -= value
                remaining.remove(key)
            break

        adjusted = False
        for key in list(remaining):
            lower, upper = bounds[key]
            proposal = target_total * (max(raw[key], 0.0) / raw_total)
            if proposal < lower:
                result[key] = lower
                target_total -= lower
                remaining.remove(key)
                adjusted = True
            elif proposal > upper:
                result[key] = upper
                target_total -= upper
                remaining.remove(key)
                adjusted = True
        if not adjusted:
            for key in list(remaining):
                raw_total = sum(max(raw[item], 0.0) for item in remaining)
                result[key] = target_total * (max(raw[key], 0.0) / max(raw_total, 1e-9))
            break

    total = sum(result.values())
    if total <= 0:
        midpoint = {key: (bounds[key][0] + bounds[key][1]) / 2.0 for key in bounds}
        total = sum(midpoint.values()) or 1.0
        return {key: value / total for key, value in midpoint.items()}
    return {key: value / total for key, value in result.items()}


def _normalize_pair(keyword_alpha: float, dense_alpha: float) -> tuple[float, float]:
    keyword_alpha = _clamp(keyword_alpha, 0.2, 0.8)
    dense_alpha = _clamp(dense_alpha, 0.2, 0.8)
    total = keyword_alpha + dense_alpha
    if total <= 0:
        return 0.5, 0.5
    keyword_alpha = keyword_alpha / total
    dense_alpha = dense_alpha / total
    keyword_alpha = _clamp(keyword_alpha, 0.2, 0.8)
    dense_alpha = _clamp(dense_alpha, 0.2, 0.8)
    total = keyword_alpha + dense_alpha
    return keyword_alpha / total, dense_alpha / total


def _query_features(query: str) -> dict[str, float]:
    tokens = re.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", str(query or ""))
    token_count = len(tokens)
    numeric_ratio = sum(1 for token in tokens if any(char.isdigit() for char in token)) / max(1, token_count)
    short_keyword = 1.0 if 0 < token_count <= 3 else 0.0
    long_query = 1.0 if token_count >= 8 else 0.0
    question_like = 1.0 if QUESTION_RE.search(str(query or "")) else 0.0
    lexical_exactness = _clamp((short_keyword * 0.6) + (numeric_ratio * 0.4), 0.0, 1.0)
    semantic_need = _clamp((long_query * 0.5) + (question_like * 0.5), 0.0, 1.0)
    return {
        "token_count": float(token_count),
        "numeric_ratio": numeric_ratio,
        "short_keyword": short_keyword,
        "long_query": long_query,
        "question_like": question_like,
        "lexical_exactness": lexical_exactness,
        "semantic_need": semantic_need,
    }


def _score_margin(scores: np.ndarray | list[float] | None) -> float:
    if scores is None:
        return 0.0
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values)
    top1 = float(sorted_values[-1])
    pivot = float(sorted_values[max(0, len(sorted_values) - min(5, len(sorted_values)))])
    return _clamp(top1 - pivot, 0.0, 1.0)


def resolve_search_weights(profile: DatasetProfile) -> SearchWeightConfig:
    raw_field_scores: dict[str, float] = {}
    for field in FIELD_BOUNDS:
        raw_field_scores[field] = (
            (0.35 * profile.field_query_overlap.get(field, 0.0))
            + (0.25 * (1.0 - profile.field_missing_rates.get(field, 1.0)))
            + (0.20 * profile.field_information_density.get(field, 0.0))
            + (0.20 * profile.field_retrieval_gain.get(field, 0.0))
        )

    normalized = _waterfill_normalize(raw_field_scores, FIELD_BOUNDS)
    field_weight_total = _clamp(
        2.0
        + (1.8 * profile.exact_match_ratio)
        + (1.2 * profile.query_document_overlap)
        + (1.0 * (1.0 - profile.field_missing_rates.get("transcript", 1.0)))
        + (0.8 * math.log1p(profile.avg_query_length + 1.0)),
        2.0,
        8.0,
    )
    absolute_weights = {field: normalized[field] * field_weight_total for field in normalized}

    lexical_signal = _clamp(
        (0.45 * profile.exact_match_ratio)
        + (0.35 * profile.query_document_overlap)
        + (0.20 * max(profile.field_query_overlap.values(), default=0.0)),
        0.0,
        1.0,
    )
    semantic_signal = _clamp(
        (0.45 * profile.semantic_need_score)
        + (0.25 * (1.0 - profile.stt_quality_score))
        + (0.30 * (profile.embedding_profile.dense_separation_score if profile.embedding_profile else 0.5)),
        0.0,
        1.0,
    )
    keyword_alpha, dense_alpha = _normalize_pair(
        0.35 + (0.65 * lexical_signal) - (0.20 * semantic_signal),
        0.35 + (0.65 * semantic_signal) - (0.10 * lexical_signal),
    )
    keyword_ranker_weight = _clamp(1.0 + (4.5 * keyword_alpha) + (1.2 * lexical_signal), 1.0, 6.0)
    dense_semantic_weight = _clamp(1.0 + (4.5 * dense_alpha) + (1.2 * semantic_signal), 1.0, 6.5)
    reasoning = (
        f"field_weights derived from overlap={json.dumps(profile.field_query_overlap, ensure_ascii=False)}, "
        f"coverage={json.dumps({k: 1.0 - v for k, v in profile.field_missing_rates.items()}, ensure_ascii=False)}, "
        f"information_density={json.dumps(profile.field_information_density, ensure_ascii=False)}, "
        f"retrieval_gain={json.dumps(profile.field_retrieval_gain, ensure_ascii=False)}; "
        f"keyword_alpha={keyword_alpha:.3f}, dense_alpha={dense_alpha:.3f}, "
        f"stt_quality={profile.stt_quality_score:.3f}, semantic_need={profile.semantic_need_score:.3f}"
    )
    return SearchWeightConfig(
        field_weights=absolute_weights,
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
        base_visualization = context.visualization
    else:
        profile = context
        base = resolve_search_weights(profile)
        base_visualization = resolve_visualization_config(profile)

    features = _query_features(query)
    keyword_margin = _score_margin(keyword_scores)
    dense_margin = _score_margin(dense_scores)

    keyword_alpha = base.keyword_alpha
    keyword_alpha += 0.18 * features["short_keyword"]
    keyword_alpha += 0.12 * features["numeric_ratio"]
    keyword_alpha += 0.12 * keyword_margin
    keyword_alpha -= 0.16 * features["question_like"]
    keyword_alpha -= 0.12 * features["long_query"]
    keyword_alpha -= 0.10 * (1.0 - profile.stt_quality_score)

    dense_alpha = base.dense_alpha
    dense_alpha += 0.18 * features["question_like"]
    dense_alpha += 0.14 * features["long_query"]
    dense_alpha += 0.12 * dense_margin
    dense_alpha += 0.10 * (1.0 - profile.stt_quality_score)
    dense_alpha -= 0.10 * features["short_keyword"]

    keyword_alpha, dense_alpha = _normalize_pair(keyword_alpha, dense_alpha)

    normalized_fields = {
        field: weight / max(base.field_weight_total, 1e-9)
        for field, weight in base.field_weights.items()
    }
    if features["short_keyword"] or features["numeric_ratio"] > 0.0:
        normalized_fields["title"] += 0.05
        normalized_fields["tags"] += 0.05
        normalized_fields["transcript"] -= 0.04
        normalized_fields["description"] -= 0.03
    if features["question_like"] or features["long_query"]:
        normalized_fields["transcript"] += 0.08
        normalized_fields["description"] += 0.04
        normalized_fields["title"] -= 0.05
        normalized_fields["tags"] -= 0.03
    normalized_fields = _waterfill_normalize(normalized_fields, FIELD_BOUNDS)
    field_weights = {field: normalized_fields[field] * base.field_weight_total for field in normalized_fields}

    keyword_ranker_weight = _clamp(
        base.keyword_ranker_weight * (0.70 + (0.60 * keyword_alpha) + (0.20 * keyword_margin)),
        0.8,
        6.0,
    )
    dense_semantic_weight = _clamp(
        base.dense_semantic_weight * (0.70 + (0.60 * dense_alpha) + (0.20 * dense_margin)),
        0.8,
        6.5,
    )

    preview_length = resolve_preview_length(
        profile,
        query=query,
        search_mode="dense" if dense_alpha >= keyword_alpha else "keyword",
        ui_width_chars=base_visualization.preview_length,
    )
    recommended_top_k = resolve_top_k(profile)
    reasoning = (
        f"query_tokens={int(features['token_count'])}, question_like={features['question_like']:.2f}, "
        f"numeric_ratio={features['numeric_ratio']:.2f}, keyword_margin={keyword_margin:.3f}, dense_margin={dense_margin:.3f}, "
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
        reasoning=reasoning,
    )


def resolve_language_config(profile: DatasetProfile) -> LanguageConfig:
    dominant = profile.dominant_language
    language_map = {
        "ko": ("ko", "ko-KR-SunHiNeural", "ko"),
        "en": ("en", "en-US-AriaNeural", "en"),
        "mixed": (None, "en-US-AriaNeural", "en"),
        "unknown": (None, "en-US-AriaNeural", "en"),
    }
    whisper_language, edge_voice, gtts_lang = language_map.get(dominant, (None, "en-US-AriaNeural", "en"))
    duration = profile.audio_profile.avg_duration_seconds
    if dominant == "mixed" or duration >= 240 or profile.stt_quality_score < 0.55:
        whisper_model = "small"
    elif duration >= 60 or profile.document_count >= 40:
        whisper_model = "base"
    else:
        whisper_model = "tiny"

    sample_rate = profile.audio_profile.sample_rate_mode or 0
    if sample_rate <= 0:
        resolved_sample_rate = None
    else:
        resolved_sample_rate = int(sample_rate)

    reasoning = (
        f"dominant_language={dominant}, ratios={json.dumps(profile.language_ratios, ensure_ascii=False)}, "
        f"avg_audio_duration={duration:.2f}, stt_quality={profile.stt_quality_score:.3f}, "
        f"sample_rate_mode={profile.audio_profile.sample_rate_mode}"
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
) -> MetricConfig:
    if score_values is not None:
        values = np.asarray(score_values, dtype=np.float32)
        values = values[np.isfinite(values)]
    else:
        values = np.asarray([], dtype=np.float32)

    base_threshold = _clamp(
        55.0
        + (15.0 * profile.exact_match_ratio)
        + (12.0 * profile.query_document_overlap)
        + (10.0 * profile.stt_quality_score)
        - (8.0 * profile.semantic_need_score),
        55.0,
        92.0,
    )
    if values.size:
        median_score = float(np.median(values))
        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))
        mad = float(np.median(np.abs(values - median_score)))
        distribution_threshold = max(q3, median_score + (1.3 * mad))
        threshold = _clamp((0.45 * base_threshold) + (0.55 * distribution_threshold), 55.0, 95.0)
    else:
        threshold = base_threshold

    exact_raw = 0.30 + (0.40 * profile.exact_match_ratio) + (0.10 * profile.query_document_overlap)
    search_raw = 0.25 + (0.35 * (profile.embedding_profile.dense_separation_score if profile.embedding_profile else 0.5))
    overlap_raw = 0.20 + (0.30 * profile.query_document_overlap) + (0.15 * profile.stt_quality_score)
    rank_raw = 0.05 + (0.10 / max(1.0, math.log1p(profile.document_count + 1.0)))
    soft_weights = _waterfill_normalize(
        {
            "exact_match": exact_raw,
            "search_score": search_raw,
            "text_overlap": overlap_raw,
            "rank_weight": rank_raw,
        },
        {
            "exact_match": (0.25, 0.60),
            "search_score": (0.15, 0.45),
            "text_overlap": (0.10, 0.35),
            "rank_weight": (0.03, 0.15),
        },
    )
    soft_precision_exact_weight = _clamp(
        0.45 + (0.30 * profile.exact_match_ratio) - (0.15 * (1.0 - profile.stt_quality_score)),
        0.40,
        0.85,
    )
    soft_recall_exact_weight = _clamp(
        0.45 + (0.25 * profile.exact_match_ratio) - (0.10 * profile.semantic_need_score),
        0.40,
        0.85,
    )
    reasoning = (
        f"threshold={threshold:.2f} from base={base_threshold:.2f}, "
        f"exact_match_ratio={profile.exact_match_ratio:.3f}, query_overlap={profile.query_document_overlap:.3f}, "
        f"stt_quality={profile.stt_quality_score:.3f}, semantic_need={profile.semantic_need_score:.3f}"
    )
    return MetricConfig(
        hallucination_threshold=threshold,
        soft_accuracy_weights=soft_weights,
        soft_precision_exact_weight=soft_precision_exact_weight,
        soft_recall_exact_weight=soft_recall_exact_weight,
        reasoning=reasoning,
    )


def resolve_cluster_config(profile: DatasetProfile, embeddings: np.ndarray | None = None) -> ClusterConfig:
    candidate_scores = profile.cluster_profile.candidate_scores if profile.cluster_profile else []
    recommended_k = profile.cluster_profile.recommended_k if profile.cluster_profile else 0
    max_k = max(CLUSTER_BOUNDS[0], min(CLUSTER_BOUNDS[1], int(math.sqrt(max(1, profile.document_count))) + max(1, profile.category_count // 2)))
    if recommended_k <= 0:
        recommended_k = _clamp(max(profile.category_count, 2), CLUSTER_BOUNDS[0], max_k)
    recommended_k = int(_clamp(recommended_k, CLUSTER_BOUNDS[0], min(max_k, max(1, profile.document_count))))
    if profile.document_count < 4:
        recommended_k = max(1, min(profile.document_count, recommended_k))
    min_cluster_size = int(
        _clamp(
            2 + round(math.sqrt(max(1, profile.document_count)) / 2.0),
            2,
            max(2, min(8, profile.document_count)),
        )
    )
    preferred_method = "hdbscan" if (profile.cluster_profile and profile.cluster_profile.density_label == "dense") else "kmeans"
    reasoning = (
        f"recommended_k={recommended_k}, category_count={profile.category_count}, document_count={profile.document_count}, "
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
    density_bonus = 0.1 if profile.cluster_profile and profile.cluster_profile.density_label == "dense" else 0.0
    tsne_perplexity = _clamp(min(45.0, max(5.0, math.sqrt(sample_count) * (1.5 + density_bonus))), 5.0, max(5.0, sample_count - 1.0))
    umap_n_neighbors = int(_clamp(round(math.sqrt(sample_count) * (1.4 + density_bonus)), 5, min(60, sample_count - 1)))
    umap_min_dist = _clamp(0.05 + (0.35 * profile.semantic_need_score) + (0.15 * (1.0 - profile.exact_match_ratio)), 0.05, 0.70)
    preview_length = resolve_preview_length(profile, search_mode="generic")
    reasoning = (
        f"tsne_perplexity={tsne_perplexity:.1f}, umap_n_neighbors={umap_n_neighbors}, "
        f"umap_min_dist={umap_min_dist:.2f}, document_count={profile.document_count}"
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
    query_len = len(re.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", str(query or "")))
    text_len = len(str(text or ""))
    sentence_hint = profile.avg_document_length / max(1.0, profile.avg_sentence_count) if profile.avg_sentence_count else 120.0
    base = 90.0 + (0.35 * min(sentence_hint, 160.0)) + (6.0 * query_len)
    if search_mode == "dense":
        base += 20.0
    elif search_mode == "keyword":
        base -= 5.0
    if text_len:
        base = min(base, max(PREVIEW_BOUNDS[0], min(PREVIEW_BOUNDS[1], text_len)))
    if ui_width_chars:
        base = min(base, float(ui_width_chars))
    return int(round(_clamp(base, PREVIEW_BOUNDS[0], PREVIEW_BOUNDS[1])))


def resolve_top_k(profile: DatasetProfile, requested_top_k: int | None = None) -> int:
    adaptive_top_k = int(
        round(
            _clamp(
                math.sqrt(max(1.0, profile.document_count)) + math.log1p(max(1, profile.category_count)),
                TOP_K_BOUNDS[0],
                TOP_K_BOUNDS[1],
            )
        )
    )
    if requested_top_k is None:
        return adaptive_top_k
    return int(_clamp(requested_top_k, 1, 50))


def _save_context(context: AdaptiveContext) -> None:
    if not context.artifact_namespace:
        return
    ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = ADAPTIVE_DIR / f"{context.artifact_namespace}__adaptive_context.json"
    save_json(path, context.to_dict())


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
    search = SearchWeightConfig(
        field_weights={
            "title": 5.0,
            "tags": 4.0,
            "description": 3.0,
            "transcript": 2.0,
        },
        field_weight_total=14.0,
        keyword_alpha=0.50,
        dense_alpha=0.50,
        keyword_ranker_weight=3.0,
        dense_semantic_weight=5.0,
        reasoning="reference-only static baseline mirroring the pre-adaptive experiment defaults",
    )
    language = LanguageConfig(
        dominant_language="ko",
        whisper_language="ko",
        whisper_model="base",
        tts_provider="edge",
        edge_voice="ko-KR-SunHiNeural",
        gtts_lang="ko",
        sample_rate=16000,
        reasoning="reference-only static baseline for comparison",
    )
    metric = MetricConfig(
        hallucination_threshold=75.0,
        soft_accuracy_weights={
            "exact_match": 0.50,
            "search_score": 0.30,
            "text_overlap": 0.15,
            "rank_weight": 0.05,
        },
        soft_precision_exact_weight=0.70,
        soft_recall_exact_weight=0.70,
        reasoning="reference-only static baseline for comparison",
    )
    cluster = ClusterConfig(
        n_clusters=6,
        min_cluster_size=4,
        preferred_method="kmeans",
        candidate_scores=[],
        reasoning="reference-only static baseline for comparison",
    )
    visualization = VisualizationConfig(
        tsne_perplexity=30.0,
        umap_n_neighbors=15,
        umap_min_dist=0.10,
        preview_length=160,
        random_seed=REPRODUCIBILITY_SEED,
        reasoning="reference-only static baseline for comparison",
    )
    return AdaptiveContext(
        profile=profile,
        search=search,
        language=language,
        metric=metric,
        cluster=cluster,
        visualization=visualization,
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
    search = resolve_search_weights(profile)
    language = resolve_language_config(profile)
    metric = resolve_metric_config(profile)
    cluster = resolve_cluster_config(profile, embeddings=embeddings)
    visualization = resolve_visualization_config(profile, embeddings=embeddings)
    context = AdaptiveContext(
        profile=profile,
        search=search,
        language=language,
        metric=metric,
        cluster=cluster,
        visualization=visualization,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
    )
    _save_context(context)
    return context
