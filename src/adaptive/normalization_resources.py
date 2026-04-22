from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import NORMALIZATION_SEED_RULES_JSON


TOKEN_RE = re.compile(r"[0-9A-Za-z\uac00-\ud7a3]+")
SURFACE_CLEAN_RE = re.compile(r"[\s\-_']+")


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _surface_forms(text: Any, max_ngram: int = 4) -> list[str]:
    raw = _normalize_text(text).lower()
    if not raw:
        return []
    tokens = raw.split()
    forms: list[str] = []
    for size in range(1, min(max_ngram, len(tokens)) + 1):
        for index in range(0, len(tokens) - size + 1):
            surface = " ".join(tokens[index : index + size]).strip()
            if surface:
                forms.append(surface)
    return forms


def _canonical_surface(surface: str) -> str:
    return SURFACE_CLEAN_RE.sub("", str(surface or "").lower())


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), quantile))


def _top_by_score(score_map: dict[str, float], keep_quantile: float) -> list[str]:
    if not score_map:
        return []
    threshold = _safe_quantile(list(score_map.values()), keep_quantile)
    return sorted([key for key, value in score_map.items() if value >= threshold], key=score_map.get, reverse=True)


def _load_seed_rules() -> tuple[dict[str, Any], bool, str]:
    if NORMALIZATION_SEED_RULES_JSON.exists():
        try:
            payload = json.loads(NORMALIZATION_SEED_RULES_JSON.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload, False, ""
        except Exception as exc:  # pragma: no cover - defensive
            return {}, True, f"seed_rule_parse_failed:{exc.__class__.__name__}"
    return {}, True, "missing_seed_rules"


@dataclass
class NormalizationResources:
    filler_terms: list[str]
    filler_scores: dict[str, float]
    alias_map: dict[str, str]
    alias_scores: dict[str, float]
    question_markers: list[str]
    spoken_markers: list[str]
    number_words: dict[str, dict[str, int]]
    token_document_frequency: dict[str, float] = field(default_factory=dict, repr=False)
    token_query_frequency: dict[str, float] = field(default_factory=dict, repr=False)
    token_frequency: dict[str, float] = field(default_factory=dict, repr=False)
    average_query_punctuation_ratio: float = 0.0
    average_query_token_count: float = 0.0
    average_transcript_repetition_ratio: float = 0.0
    average_numeric_token_ratio: float = 0.0
    normalization_preference: float = 0.0
    recommended_mode: str = "baseline"
    used_fallback_resources: bool = False
    fallback_reason: str = ""

    def token_idf(self, token: str, document_count: int) -> float:
        df = float(self.token_document_frequency.get(str(token).lower(), 0.0))
        denominator = math.log1p(max(1, document_count) + 1.0)
        if denominator <= 0.0:
            return 0.0
        return _clip01(math.log1p((max(1.0, float(document_count)) + 1.0) / (1.0 + df)) / denominator)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filler_terms"] = self.filler_terms[:48]
        payload["question_markers"] = self.question_markers[:48]
        payload["spoken_markers"] = self.spoken_markers[:48]
        payload["alias_map"] = dict(list(self.alias_map.items())[:64])
        payload["token_document_frequency"] = dict(list(self.token_document_frequency.items())[:64])
        payload["token_query_frequency"] = dict(list(self.token_query_frequency.items())[:64])
        payload["token_frequency"] = dict(list(self.token_frequency.items())[:64])
        return payload


def _iter_corpus_texts(metadata: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    transcript_texts: list[str] = []
    lexical_texts: list[str] = []
    all_texts: list[str] = []
    for row in metadata.to_dict(orient="records"):
        transcript = _normalize_text(
            " ".join(
                part
                for part in [row.get("stt_transcript", ""), row.get("original_transcript", "")]
                if str(part or "").strip()
            )
        )
        lexical = _normalize_text(
            " ".join(
                part
                for part in [
                    row.get("title", ""),
                    row.get("description", ""),
                    row.get("tags", ""),
                    row.get("keywords", ""),
                    row.get("category", ""),
                ]
                if str(part or "").strip()
            )
        )
        transcript_texts.append(transcript)
        lexical_texts.append(lexical)
        all_texts.append(_normalize_text(" ".join(part for part in [lexical, transcript] if part)))
    return transcript_texts, lexical_texts, all_texts


def _token_frequencies(texts: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    frequency = Counter()
    document_frequency = Counter()
    for text in texts:
        tokens = _tokenize(text)
        frequency.update(tokens)
        document_frequency.update(set(tokens))
    return dict(frequency), dict(document_frequency)


def _repetition_ratio(text: str) -> float:
    tokens = _tokenize(text)
    if len(tokens) < 2:
        return 0.0
    repeated = sum(1 for index in range(1, len(tokens)) if tokens[index] == tokens[index - 1])
    return repeated / max(1, len(tokens))


def _numeric_ratio(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if any(char.isdigit() for char in token)) / max(1, len(tokens))


def _dynamic_fillers(
    transcript_texts: list[str],
    query_texts: list[str],
    lexical_texts: list[str],
    seed_fillers: list[str],
) -> tuple[list[str], dict[str, float], float]:
    transcript_token_count = Counter()
    lexical_token_count = Counter()
    query_token_count = Counter()
    transcript_bigram_count = Counter()
    query_bigram_count = Counter()
    repeat_count = Counter()
    neighbor_diversity: dict[str, set[str]] = defaultdict(set)

    for text in transcript_texts:
        tokens = _tokenize(text)
        transcript_token_count.update(tokens)
        transcript_bigram_count.update(" ".join(tokens[index : index + 2]) for index in range(0, max(0, len(tokens) - 1)))
        for index, token in enumerate(tokens):
            if index > 0:
                neighbor_diversity[token].add(tokens[index - 1])
            if index + 1 < len(tokens):
                neighbor_diversity[token].add(tokens[index + 1])
            if index > 0 and tokens[index] == tokens[index - 1]:
                repeat_count[token] += 1

    for text in lexical_texts:
        lexical_token_count.update(_tokenize(text))
    for text in query_texts:
        tokens = _tokenize(text)
        query_token_count.update(tokens)
        query_bigram_count.update(" ".join(tokens[index : index + 2]) for index in range(0, max(0, len(tokens) - 1)))

    scores: dict[str, float] = {}
    total_transcript = float(sum(transcript_token_count.values()) or 1.0)
    total_query = float(sum(query_token_count.values()) or 1.0)
    total_lexical = float(sum(lexical_token_count.values()) or 1.0)

    for token, count in transcript_token_count.items():
        transcript_share = count / total_transcript
        query_share = query_token_count.get(token, 0) / total_query
        lexical_share = lexical_token_count.get(token, 0) / total_lexical
        diversity = len(neighbor_diversity[token]) / max(1.0, count)
        repetition = repeat_count.get(token, 0) / max(1.0, count)
        shortness = 1.0 - (len(token) / max(2.0, float(np.mean([len(item) for item in transcript_token_count]) or 2.0)))
        score = np.mean(
            [
                _clip01(transcript_share / max(transcript_share + lexical_share, 1e-9)),
                _clip01(transcript_share / max(transcript_share + query_share, 1e-9)),
                _clip01(diversity),
                _clip01(repetition * 2.0),
                _clip01(shortness),
            ]
        )
        scores[token] = float(score)

    for filler in seed_fillers:
        normalized = _normalize_text(filler).lower()
        if normalized:
            scores[normalized] = max(scores.get(normalized, 0.0), 1.0)

    selected = dict.fromkeys(_top_by_score(scores, 0.82) + [_normalize_text(item).lower() for item in seed_fillers if item])
    average_repetition = float(np.mean([_repetition_ratio(text) for text in transcript_texts])) if transcript_texts else 0.0
    return list(selected.keys()), scores, average_repetition


def _dynamic_aliases(
    corpus_texts: list[str],
    query_texts: list[str],
    seed_alias_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, float]]:
    surface_counter: dict[str, Counter] = defaultdict(Counter)
    canonical_support = Counter()
    for text in corpus_texts + query_texts:
        for surface in _surface_forms(text):
            canonical = _canonical_surface(surface)
            if len(canonical) < 2:
                continue
            surface_counter[canonical][surface] += 1
            canonical_support[canonical] += 1

    alias_map: dict[str, str] = {}
    alias_scores: dict[str, float] = {}
    family_scores: list[float] = []

    for canonical, surfaces in surface_counter.items():
        if len(surfaces) < 2:
            continue
        ranked = sorted(
            surfaces.items(),
            key=lambda item: (
                item[1],
                -len(item[0].split()),
                -abs(len(item[0]) - len(canonical)),
            ),
            reverse=True,
        )
        target = ranked[0][0]
        family_total = float(sum(surfaces.values()) or 1.0)
        for surface, count in ranked[1:]:
            if surface == target:
                continue
            support = count / family_total
            compact_gain = _clip01((len(surface) - len(target)) / max(1.0, len(surface)))
            score = float(np.mean([support, compact_gain, canonical_support[canonical] / max(1.0, family_total)]))
            alias_map[surface] = target
            alias_scores[surface] = score
            family_scores.append(score)

    if family_scores:
        score_floor = _safe_quantile(family_scores, 0.65)
        alias_map = {source: target for source, target in alias_map.items() if alias_scores.get(source, 0.0) >= score_floor}
        alias_scores = {source: score for source, score in alias_scores.items() if score >= score_floor}

    for source, target in seed_alias_map.items():
        normalized_source = _normalize_text(source).lower()
        normalized_target = _normalize_text(target).lower()
        if normalized_source and normalized_target and normalized_source != normalized_target:
            alias_map[normalized_source] = normalized_target
            alias_scores[normalized_source] = max(alias_scores.get(normalized_source, 0.0), 1.0)
    return alias_map, alias_scores


def _query_markers(
    queryset: pd.DataFrame | None,
    seed_markers: list[str],
    target_type: str,
) -> list[str]:
    if queryset is None or queryset.empty or "query" not in queryset.columns or "query_type" not in queryset.columns:
        return list(dict.fromkeys(_normalize_text(marker).lower() for marker in seed_markers if marker))

    frame = queryset.copy()
    frame["query"] = frame["query"].fillna("").astype(str)
    frame["query_type"] = frame["query_type"].fillna("").astype(str)
    focus = frame.loc[frame["query_type"].eq(target_type), "query"].tolist()
    other = frame.loc[frame["query_type"].ne(target_type), "query"].tolist()

    focus_counter = Counter(token for query in focus for token in _tokenize(query))
    other_counter = Counter(token for query in other for token in _tokenize(query))
    score_map: dict[str, float] = {}
    total_focus = float(sum(focus_counter.values()) or 1.0)
    total_other = float(sum(other_counter.values()) or 1.0)
    for token, count in focus_counter.items():
        focus_share = count / total_focus
        other_share = other_counter.get(token, 0) / total_other
        score_map[token] = _clip01(focus_share / max(focus_share + other_share, 1e-9))

    selected = _top_by_score(score_map, 0.80)
    selected.extend(_normalize_text(marker).lower() for marker in seed_markers if marker)
    return list(dict.fromkeys(token for token in selected if token))


def build_normalization_resources(
    metadata: pd.DataFrame,
    queryset: pd.DataFrame | None = None,
    *,
    stt_quality_score: float | None = None,
    corpus_token_coverage: float | None = None,
) -> NormalizationResources:
    seed_rules, seed_fallback_used, seed_fallback_reason = _load_seed_rules()
    transcript_texts, lexical_texts, corpus_texts = _iter_corpus_texts(metadata)
    query_texts = (
        queryset["query"].fillna("").astype(str).tolist()
        if queryset is not None and not queryset.empty and "query" in queryset.columns
        else []
    )

    token_frequency, token_document_frequency = _token_frequencies(corpus_texts)
    query_frequency, _ = _token_frequencies(query_texts)
    fillers, filler_scores, average_repetition = _dynamic_fillers(
        transcript_texts,
        query_texts,
        lexical_texts,
        list(seed_rules.get("fillers", [])),
    )
    alias_map, alias_scores = _dynamic_aliases(
        corpus_texts,
        query_texts,
        {str(key): str(value) for key, value in dict(seed_rules.get("alias_seed_map", {})).items()},
    )
    question_markers = _query_markers(queryset, list(seed_rules.get("question_markers", [])), "natural_question")
    spoken_markers = _query_markers(queryset, list(seed_rules.get("spoken_markers", [])), "stt_style")

    query_punct_ratios = [
        sum(1 for char in str(query) if not char.isalnum() and not ("\uac00" <= char <= "\ud7a3") and not char.isspace())
        / max(1, len(str(query)))
        for query in query_texts
        if str(query).strip()
    ]
    average_query_punctuation_ratio = float(np.mean(query_punct_ratios)) if query_punct_ratios else 0.0
    average_query_token_count = float(np.mean([len(_tokenize(query)) for query in query_texts if str(query).strip()])) if query_texts else 0.0
    average_numeric_token_ratio = float(np.mean([_numeric_ratio(text) for text in corpus_texts])) if corpus_texts else 0.0

    filler_prevalence = float(np.mean(list(filler_scores.values()))) if filler_scores else 0.0
    alias_prevalence = float(np.mean(list(alias_scores.values()))) if alias_scores else 0.0
    numeric_prevalence = average_numeric_token_ratio
    noise_signal = 1.0 - float(stt_quality_score if stt_quality_score is not None else 0.5)
    coverage_signal = float(corpus_token_coverage if corpus_token_coverage is not None else 0.5)
    adaptive_score = float(np.mean([filler_prevalence, alias_prevalence, numeric_prevalence, noise_signal]))
    baseline_score = float(np.mean([coverage_signal, 1.0 - filler_prevalence, 1.0 - alias_prevalence, 1.0 - noise_signal]))
    recommended_mode = "adaptive_corpus" if adaptive_score >= baseline_score else "baseline"

    return NormalizationResources(
        filler_terms=fillers,
        filler_scores=filler_scores,
        alias_map=alias_map,
        alias_scores=alias_scores,
        question_markers=question_markers,
        spoken_markers=spoken_markers,
        number_words={
            "simple": {str(key): int(value) for key, value in dict(seed_rules.get("number_words", {}).get("simple", {})).items()},
            "tens": {str(key): int(value) for key, value in dict(seed_rules.get("number_words", {}).get("tens", {})).items()},
            "scales": {str(key): int(value) for key, value in dict(seed_rules.get("number_words", {}).get("scales", {})).items()},
            "year_suffix": {
                str(key): int(value)
                for key, value in dict(seed_rules.get("number_words", {}).get("year_suffix", {})).items()
            },
        },
        token_document_frequency={str(key): float(value) for key, value in token_document_frequency.items()},
        token_query_frequency={str(key): float(value) for key, value in query_frequency.items()},
        token_frequency={str(key): float(value) for key, value in token_frequency.items()},
        average_query_punctuation_ratio=average_query_punctuation_ratio,
        average_query_token_count=average_query_token_count,
        average_transcript_repetition_ratio=average_repetition,
        average_numeric_token_ratio=average_numeric_token_ratio,
        normalization_preference=adaptive_score,
        recommended_mode=recommended_mode,
        used_fallback_resources=seed_fallback_used,
        fallback_reason=seed_fallback_reason,
    )
