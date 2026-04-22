from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.adaptive.dataset_profile import DatasetProfile
from src.adaptive.normalization_resources import NormalizationResources
from src.adaptive.performance_profile import PerformanceProfile


TOKEN_RE = re.compile(r"[0-9A-Za-z\uac00-\ud7a3]+")


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _entropy(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    probabilities = [value / len(tokens) for value in counts.values()]
    entropy = -sum(probability * math.log(probability + 1e-9) for probability in probabilities)
    return _clip01(entropy / math.log(len(tokens) + 1e-9))


def _repeat_ratio(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    repeated = sum(1 for index in range(1, len(tokens)) if tokens[index] == tokens[index - 1])
    return repeated / max(1, len(tokens))


def _garble_ratio(text: str) -> float:
    value = str(text or "")
    if not value:
        return 0.0
    weird = sum(1 for char in value if char == "\ufffd" or (ord(char) < 32 and char not in "\r\n\t"))
    punctuation = sum(1 for char in value if not char.isalnum() and not ("\uac00" <= char <= "\ud7a3") and not char.isspace())
    return _clip01((weird + (0.15 * punctuation)) / max(1, len(value)))


@dataclass
class QueryFeatureVector:
    token_count: float
    unique_token_ratio: float
    lexical_entropy: float
    lexical_rarity: float
    punctuation_ratio: float
    numeric_salience: float
    named_entity_score: float
    question_likeness: float
    spoken_style: float
    stt_noise_score: float
    lexical_precision: float
    semantic_need: float
    ambiguity: float
    candidate_pressure: float
    exact_affinity: float
    paraphrase_affinity: float
    natural_affinity: float
    stt_affinity: float
    reranker_value_signal: float
    dominant_bucket: str
    token_salience: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_salience"] = dict(list(self.token_salience.items())[:24])
        return payload


def extract_query_features(
    query: str,
    profile: DatasetProfile,
    normalization: NormalizationResources,
    performance: PerformanceProfile | None = None,
) -> QueryFeatureVector:
    raw_query = str(query or "")
    tokens = _tokenize(raw_query)
    token_count = float(len(tokens))
    unique_token_ratio = len(set(tokens)) / max(1, len(tokens))
    lexical_entropy = _entropy(tokens)
    punctuation_ratio = sum(
        1
        for char in raw_query
        if not char.isalnum() and not ("\uac00" <= char <= "\ud7a3") and not char.isspace()
    ) / max(1, len(raw_query))

    idf_values = [normalization.token_idf(token, profile.document_count) for token in tokens] if tokens else []
    lexical_rarity = float(np.mean(idf_values)) if idf_values else 0.0
    numeric_values = [
        normalization.token_idf(token, profile.document_count)
        for token in tokens
        if any(char.isdigit() for char in token)
    ]
    numeric_density = len(numeric_values) / max(1, len(tokens))
    numeric_salience = float(np.mean(numeric_values)) if numeric_values else 0.0
    numeric_salience = float(np.mean([numeric_density, numeric_salience]))

    token_lengths = [len(token) for token in tokens] or [0]
    average_token_length = float(np.mean(token_lengths))
    relative_length = token_count / max(1.0, profile.avg_query_length)
    query_length_signal = _clip01(relative_length / max(1.0, relative_length + 1.0))
    punctuation_signal = _clip01(
        punctuation_ratio / max(punctuation_ratio + normalization.average_query_punctuation_ratio + 1e-9, 1e-9)
    )

    filler_density = sum(1 for token in tokens if token in set(normalization.filler_terms)) / max(1, len(tokens))
    question_marker_density = sum(1 for token in tokens if token in set(normalization.question_markers)) / max(1, len(tokens))
    spoken_marker_density = sum(1 for token in tokens if token in set(normalization.spoken_markers)) / max(1, len(tokens))

    repeat_ratio = _repeat_ratio(tokens)
    repeat_signal = _clip01(
        repeat_ratio
        / max(repeat_ratio + normalization.average_transcript_repetition_ratio + 1e-9, 1e-9)
    )
    garble_ratio = _garble_ratio(raw_query)
    stt_noise_score = float(np.mean([filler_density, spoken_marker_density, repeat_signal, garble_ratio]))

    named_entity_score = float(
        np.mean(
            [
                lexical_rarity,
                numeric_salience,
                _clip01(average_token_length / max(average_token_length + profile.avg_query_length, 1e-9)),
            ]
        )
    )
    question_likeness = float(
        np.mean(
            [
                punctuation_signal,
                question_marker_density,
                query_length_signal,
                1.0 - filler_density,
            ]
        )
    )
    spoken_style = float(
        np.mean(
            [
                filler_density,
                spoken_marker_density,
                repeat_signal,
                1.0 - punctuation_signal,
                stt_noise_score,
            ]
        )
    )
    lexical_precision = float(
        np.mean(
            [
                lexical_rarity,
                named_entity_score,
                numeric_salience,
                1.0 - question_likeness,
                1.0 - spoken_style,
            ]
        )
    )
    semantic_need = float(
        np.mean(
            [
                question_likeness,
                spoken_style,
                query_length_signal,
                1.0 - lexical_precision,
                profile.semantic_need_score,
            ]
        )
    )
    ambiguity = float(
        np.mean(
            [
                1.0 - lexical_rarity,
                1.0 - named_entity_score,
                lexical_entropy,
                query_length_signal,
                1.0 - unique_token_ratio,
            ]
        )
    )
    token_salience = {
        token: float(np.mean([normalization.token_idf(token, profile.document_count), numeric_salience, named_entity_score]))
        for token in dict.fromkeys(tokens)
    }
    if performance is None:
        performance = PerformanceProfile()

    raw_affinities = {
        "exact_keyword": float(
            np.mean([lexical_precision, named_entity_score, 1.0 - question_likeness, 1.0 - spoken_style, numeric_salience])
        ),
        "paraphrase_semantic": float(
            np.mean([semantic_need, lexical_rarity, lexical_precision, 1.0 - question_likeness, 1.0 - spoken_style])
        ),
        "natural_question": float(
            np.mean([semantic_need, question_likeness, query_length_signal, 1.0 - lexical_precision, 1.0 - numeric_salience])
        ),
        "stt_style": float(np.mean([spoken_style, stt_noise_score, 1.0 - punctuation_signal, 1.0 - named_entity_score])),
    }
    affinity_total = sum(raw_affinities.values()) or 1.0
    affinities = {bucket: value / affinity_total for bucket, value in raw_affinities.items()}

    reranker_value_signal = 0.0
    for bucket, affinity in affinities.items():
        bucket_profile = performance.bucket(bucket)
        reranker_value_signal += affinity * float(
            np.mean(
                [
                    max(0.0, bucket_profile.delta_semantic_success_rate),
                    max(0.0, bucket_profile.delta_mrr),
                    max(0.0, bucket_profile.delta_ndcg),
                    1.0 - bucket_profile.semantic_success_rate,
                ]
            )
        )
    reranker_value_signal = _clip01(reranker_value_signal + performance.reranker_value_prior)
    candidate_pressure = float(
        np.mean(
            [
                semantic_need,
                ambiguity,
                question_likeness,
                spoken_style,
                reranker_value_signal,
                affinities["natural_question"],
                affinities["stt_style"],
            ]
        )
    )
    dominant_bucket = max(affinities.items(), key=lambda item: item[1])[0]
    reasoning = (
        f"bucket={dominant_bucket}, rarity={lexical_rarity:.3f}, question={question_likeness:.3f}, "
        f"spoken={spoken_style:.3f}, semantic_need={semantic_need:.3f}, ambiguity={ambiguity:.3f}, "
        f"reranker_value={reranker_value_signal:.3f}"
    )

    return QueryFeatureVector(
        token_count=token_count,
        unique_token_ratio=unique_token_ratio,
        lexical_entropy=lexical_entropy,
        lexical_rarity=lexical_rarity,
        punctuation_ratio=punctuation_ratio,
        numeric_salience=numeric_salience,
        named_entity_score=named_entity_score,
        question_likeness=question_likeness,
        spoken_style=spoken_style,
        stt_noise_score=stt_noise_score,
        lexical_precision=lexical_precision,
        semantic_need=semantic_need,
        ambiguity=ambiguity,
        candidate_pressure=candidate_pressure,
        exact_affinity=affinities["exact_keyword"],
        paraphrase_affinity=affinities["paraphrase_semantic"],
        natural_affinity=affinities["natural_question"],
        stt_affinity=affinities["stt_style"],
        reranker_value_signal=reranker_value_signal,
        dominant_bucket=dominant_bucket,
        token_salience=token_salience,
        reasoning=reasoning,
    )
