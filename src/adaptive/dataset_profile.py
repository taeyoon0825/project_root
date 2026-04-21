from __future__ import annotations

import math
import re
import subprocess
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.embedding.vector_models import EmbeddingModelWrapper
from src.search.text_source import DEFAULT_TEXT_SOURCE, resolve_primary_text, split_line_into_sentences


TOKEN_RE = re.compile(r"[0-9A-Za-z\uac00-\ud7a3]+")
QUESTION_TOKENS = {
    "what",
    "why",
    "how",
    "who",
    "where",
    "when",
    "which",
    "explain",
    "tell",
    "describe",
    "무엇",
    "왜",
    "어떻게",
    "누가",
    "어디",
    "언제",
    "설명",
    "알려",
    "질문",
}
TEXT_FIELDS = ("title", "tags", "description", "transcript")
DEFAULT_SAMPLE_LIMIT = 48
DEFAULT_QUERY_LIMIT = 24


@dataclass
class AudioProfile:
    sampled_file_count: int = 0
    avg_duration_seconds: float = 0.0
    avg_sample_rate: float = 0.0
    sample_rate_mode: int = 0
    duration_std_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingProfile:
    model_alias: str = ""
    sampled_document_count: int = 0
    sampled_query_count: int = 0
    mean_query_similarity: float = 0.0
    std_query_similarity: float = 0.0
    top1_mean_similarity: float = 0.0
    top5_mean_similarity: float = 0.0
    top1_top5_margin: float = 0.0
    dense_separation_score: float = 0.0
    keyword_preference_score: float = 0.0
    dense_preference_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterProfile:
    candidate_scores: list[dict[str, float | int]] = field(default_factory=list)
    recommended_k: int = 0
    density_label: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetProfile:
    document_count: int
    query_count: int
    category_count: int
    avg_document_length: float
    document_length_std: float
    avg_sentence_count: float
    avg_query_length: float
    query_document_overlap: float
    exact_match_ratio: float
    semantic_need_score: float
    semantic_match_required: bool
    field_missing_rates: dict[str, float]
    field_avg_lengths: dict[str, float]
    field_avg_token_counts: dict[str, float]
    field_information_density: dict[str, float]
    field_query_overlap: dict[str, float]
    field_retrieval_gain: dict[str, float]
    lexical_field_quality: dict[str, float]
    vocabulary_diversity: float
    dominant_language: str
    language_ratios: dict[str, float]
    stt_quality_score: float
    stt_short_ratio: float
    stt_repetition_ratio: float
    stt_garbled_ratio: float
    stt_alignment_score: float
    audio_profile: AudioProfile
    embedding_profile: EmbeddingProfile | None = None
    cluster_profile: ClusterProfile | None = None
    sampled_queries: list[str] = field(default_factory=list)
    sampled_document_ids: list[str] = field(default_factory=list)
    corpus_token_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.audio_profile is not None:
            payload["audio_profile"] = self.audio_profile.to_dict()
        if self.embedding_profile is not None:
            payload["embedding_profile"] = self.embedding_profile.to_dict()
        if self.cluster_profile is not None:
            payload["cluster_profile"] = self.cluster_profile.to_dict()
        return payload


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _sample_indices(size: int, limit: int) -> list[int]:
    if size <= limit:
        return list(range(size))
    return sorted(set(int(index) for index in np.linspace(0, size - 1, num=limit)))


def _sample_frame(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.iloc[_sample_indices(len(frame), limit)].reset_index(drop=True)


def _field_text(row: pd.Series, field: str, text_source: str) -> str:
    if field == "transcript":
        return resolve_primary_text(row, text_source=text_source)
    if field == "tags":
        return str(row.get("tags", "") or row.get("keywords", "")).strip()
    return str(row.get(field, "")).strip()


def _query_frame(metadata: pd.DataFrame, queryset: pd.DataFrame | None, text_source: str) -> pd.DataFrame:
    if queryset is not None and not queryset.empty and "query" in queryset.columns:
        frame = queryset.copy()
        frame["query"] = frame["query"].fillna("").astype(str).str.strip()
        frame = frame.loc[frame["query"].ne("")].reset_index(drop=True)
        if not frame.empty:
            return _sample_frame(frame[["query"]], DEFAULT_QUERY_LIMIT)

    rows: list[dict[str, str]] = []
    for row in _sample_frame(metadata, DEFAULT_QUERY_LIMIT).itertuples(index=False):
        transcript = str(getattr(row, "stt_transcript", "") or getattr(row, "original_transcript", "")).strip()
        snippet = " ".join(transcript.split()[:12]).strip()
        query = " ".join(
            part
            for part in [
                str(getattr(row, "title", "")).strip(),
                str(getattr(row, "keywords", "") or getattr(row, "tags", "")).strip(),
                snippet,
            ]
            if part
        ).strip()
        if query:
            rows.append({"query": query})
    return pd.DataFrame(rows or [{"query": ""}]).iloc[:DEFAULT_QUERY_LIMIT]


def _sentence_count(text: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    lines = [part for part in re.split(r"[\r\n]+", normalized) if part.strip()]
    total = 0
    for line in lines:
        total += max(1, len(split_line_into_sentences(line)))
    return total


def _language_label(text: str) -> str:
    if not text:
        return "unknown"
    hangul = sum(1 for char in text if "\uac00" <= char <= "\ud7a3")
    latin = sum(1 for char in text if ("a" <= char.lower() <= "z"))
    if hangul == 0 and latin == 0:
        return "unknown"
    if hangul > 0 and latin > 0:
        ratio = min(hangul, latin) / max(hangul, latin)
        if ratio >= 0.25:
            return "mixed"
    if hangul >= latin:
        return "ko"
    return "en"


def _language_distribution(texts: list[str]) -> tuple[str, dict[str, float]]:
    counts = {"ko": 0, "en": 0, "mixed": 0, "unknown": 0}
    for text in texts:
        counts[_language_label(text)] += 1
    total = max(1, sum(counts.values()))
    ratios = {key: value / total for key, value in counts.items()}
    dominant = max(["ko", "en", "mixed", "unknown"], key=lambda key: ratios[key])
    return dominant, ratios


def _repetition_ratio(text: str) -> float:
    tokens = _tokenize(text)
    if len(tokens) < 4:
        return 0.0
    repeated = 0
    for index in range(2, len(tokens)):
        if tokens[index] == tokens[index - 1] == tokens[index - 2]:
            repeated += 1
    return repeated / max(1, len(tokens))


def _garbled_ratio(text: str) -> float:
    value = str(text or "")
    if not value:
        return 1.0
    weird_chars = sum(1 for char in value if char == "\ufffd" or (ord(char) < 32 and char not in "\r\n\t"))
    punctuation = sum(1 for char in value if not char.isalnum() and not ("\uac00" <= char <= "\ud7a3") and not char.isspace())
    return (weird_chars + (0.15 * punctuation)) / max(1, len(value))


def _jaccard(tokens_a: list[str], tokens_b: list[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def _stt_quality(metadata: pd.DataFrame) -> tuple[float, float, float, float, float]:
    if metadata.empty:
        return 0.5, 0.0, 0.0, 0.0, 0.0

    short_flags: list[float] = []
    repetition_scores: list[float] = []
    garbled_scores: list[float] = []
    alignment_scores: list[float] = []

    for row in metadata.itertuples(index=False):
        stt_text = str(getattr(row, "stt_transcript", "")).strip()
        original_text = str(getattr(row, "original_transcript", "")).strip()
        stt_tokens = _tokenize(stt_text)
        original_tokens = _tokenize(original_text)
        short_flags.append(1.0 if len(stt_tokens) < 4 else 0.0)
        repetition_scores.append(_repetition_ratio(stt_text))
        garbled_scores.append(_garbled_ratio(stt_text))
        if original_tokens:
            alignment_scores.append(_jaccard(stt_tokens, original_tokens))

    short_ratio = float(np.mean(short_flags)) if short_flags else 0.0
    repetition_ratio = float(np.mean(repetition_scores)) if repetition_scores else 0.0
    garbled_ratio = float(np.mean(garbled_scores)) if garbled_scores else 0.0
    alignment_score = float(np.mean(alignment_scores)) if alignment_scores else 0.5

    quality = 1.0 - (
        (0.35 * short_ratio)
        + (0.25 * _clamp(repetition_ratio * 4.0, 0.0, 1.0))
        + (0.20 * _clamp(garbled_ratio * 3.0, 0.0, 1.0))
        + (0.20 * (1.0 - alignment_score))
    )
    return _clamp(quality, 0.05, 0.99), short_ratio, repetition_ratio, garbled_ratio, alignment_score


def _field_statistics(metadata: pd.DataFrame, text_source: str) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    missing_rates: dict[str, float] = {}
    avg_lengths: dict[str, float] = {}
    avg_token_counts: dict[str, float] = {}
    information_density: dict[str, float] = {}

    for field in TEXT_FIELDS:
        texts = [_field_text(row, field, text_source) for _, row in metadata.iterrows()]
        normalized_texts = [_normalize_text(text) for text in texts]
        missing_rates[field] = float(np.mean([1.0 if not text else 0.0 for text in normalized_texts])) if normalized_texts else 1.0
        lengths = [len(text) for text in normalized_texts if text]
        token_lists = [_tokenize(text) for text in normalized_texts if text]
        avg_lengths[field] = float(np.mean(lengths)) if lengths else 0.0
        avg_token_counts[field] = float(np.mean([len(tokens) for tokens in token_lists])) if token_lists else 0.0
        flat_tokens = [token for tokens in token_lists for token in tokens]
        ttr = (len(set(flat_tokens)) / max(1, len(flat_tokens))) if flat_tokens else 0.0
        information_density[field] = ttr * math.log1p(avg_token_counts[field] + avg_lengths[field] / 32.0)

    max_density = max(information_density.values(), default=1.0) or 1.0
    information_density = {field: value / max_density for field, value in information_density.items()}
    return missing_rates, avg_lengths, avg_token_counts, information_density


def _lexical_query_signals(
    metadata: pd.DataFrame,
    queries: pd.DataFrame,
    text_source: str,
) -> tuple[dict[str, float], dict[str, float], float, float, float]:
    sample_docs = _sample_frame(metadata, DEFAULT_SAMPLE_LIMIT)
    sample_queries = _sample_frame(queries, DEFAULT_QUERY_LIMIT)
    if sample_docs.empty or sample_queries.empty:
        zeroes = {field: 0.0 for field in TEXT_FIELDS}
        return zeroes, zeroes, 0.0, 0.0, 0.0

    field_tokens = {
        field: [set(_tokenize(_field_text(row, field, text_source))) for _, row in sample_docs.iterrows()]
        for field in TEXT_FIELDS
    }
    search_texts = [
        _normalize_text(
            " ".join(_field_text(row, field, text_source) for field in TEXT_FIELDS)
        )
        for _, row in sample_docs.iterrows()
    ]
    corpus_tokens = set(token for text in search_texts for token in _tokenize(text))

    field_overlap_totals = {field: 0.0 for field in TEXT_FIELDS}
    field_gain_totals = {field: 0.0 for field in TEXT_FIELDS}
    exact_matches = 0.0
    overlap_totals = 0.0
    coverage_totals = 0.0

    for _, query_row in sample_queries.iterrows():
        query = _normalize_text(query_row.get("query", ""))
        query_tokens = _tokenize(query)
        if not query_tokens:
            continue
        token_count = len(query_tokens)
        coverage_totals += sum(1 for token in query_tokens if token in corpus_tokens) / max(1, token_count)
        if any(query and query in text for text in search_texts):
            exact_matches += 1.0

        best_search_overlap = 0.0
        for doc_text in search_texts:
            doc_tokens = set(_tokenize(doc_text))
            overlap = sum(1 for token in query_tokens if token in doc_tokens) / max(1, token_count)
            best_search_overlap = max(best_search_overlap, overlap)
        overlap_totals += best_search_overlap

        for field in TEXT_FIELDS:
            best_overlap = 0.0
            exact_field = 0.0
            for doc_index, row in enumerate(sample_docs.itertuples(index=False)):
                field_text = _normalize_text(_field_text(pd.Series(row._asdict()), field, text_source))
                overlap = sum(1 for token in query_tokens if token in field_tokens[field][doc_index]) / max(1, token_count)
                best_overlap = max(best_overlap, overlap)
                if query and query in field_text:
                    exact_field = 1.0
            field_overlap_totals[field] += best_overlap
            field_gain_totals[field] += (0.6 * best_overlap) + (0.4 * exact_field)

    query_count = max(1, len(sample_queries))
    field_query_overlap = {field: field_overlap_totals[field] / query_count for field in TEXT_FIELDS}
    field_retrieval_gain = {field: field_gain_totals[field] / query_count for field in TEXT_FIELDS}
    exact_match_ratio = exact_matches / query_count
    query_doc_overlap = overlap_totals / query_count
    corpus_coverage = coverage_totals / query_count
    return field_query_overlap, field_retrieval_gain, query_doc_overlap, exact_match_ratio, corpus_coverage


def _audio_info(path: Path) -> tuple[float, int] | None:
    try:
        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as handle:
                sample_rate = int(handle.getframerate())
                frame_count = int(handle.getnframes())
                duration = frame_count / max(1, sample_rate)
                return duration, sample_rate
    except Exception:
        return None

    ffprobe = "ffprobe"
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate:format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=8)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    try:
        sample_rate = int(float(lines[0]))
        duration = float(lines[1])
    except ValueError:
        return None
    return duration, sample_rate


def _build_audio_profile(metadata: pd.DataFrame) -> AudioProfile:
    durations: list[float] = []
    sample_rates: list[int] = []
    audio_paths = []
    for _, row in metadata.iterrows():
        candidate = str(row.get("audio_path", "")).strip() or str(row.get("audio_file_path", "")).strip()
        if candidate:
            audio_paths.append(Path(candidate))
    for path in _sample_frame(pd.DataFrame({"path": audio_paths}), 12)["path"].tolist() if audio_paths else []:
        if not isinstance(path, Path) or not path.exists():
            continue
        info = _audio_info(path)
        if info is None:
            continue
        duration, sample_rate = info
        durations.append(duration)
        sample_rates.append(sample_rate)

    if not durations or not sample_rates:
        return AudioProfile()

    mode = int(pd.Series(sample_rates).mode().iloc[0])
    return AudioProfile(
        sampled_file_count=len(durations),
        avg_duration_seconds=float(np.mean(durations)),
        avg_sample_rate=float(np.mean(sample_rates)),
        sample_rate_mode=mode,
        duration_std_seconds=float(np.std(durations)),
    )


def _embedding_profile(
    metadata: pd.DataFrame,
    queries: pd.DataFrame,
    text_source: str,
    embedding_model_alias: str | None,
    embeddings: np.ndarray | None,
    query_document_overlap: float,
    exact_match_ratio: float,
    semantic_need_score: float,
) -> tuple[EmbeddingProfile | None, ClusterProfile | None]:
    if metadata.empty:
        return None, None

    sample_docs = _sample_frame(metadata, 36)
    sample_queries = _sample_frame(queries, 16)
    doc_texts = [resolve_primary_text(row, text_source=text_source) for _, row in sample_docs.iterrows()]
    query_texts = sample_queries["query"].fillna("").astype(str).tolist() if not sample_queries.empty else []
    if not doc_texts:
        return None, None

    doc_embeddings = embeddings
    model_alias = embedding_model_alias or ""
    if doc_embeddings is None and embedding_model_alias:
        wrapper = EmbeddingModelWrapper(embedding_model_alias)
        doc_embeddings = wrapper.encode_documents(doc_texts, batch_size=min(16, len(doc_texts)))
        query_embeddings = wrapper.encode_queries(query_texts) if query_texts else np.zeros((0, doc_embeddings.shape[1]), dtype=np.float32)
    elif doc_embeddings is not None:
        if len(doc_embeddings) != len(doc_texts):
            doc_embeddings = doc_embeddings[: len(doc_texts)]
        if query_texts and embedding_model_alias:
            wrapper = EmbeddingModelWrapper(embedding_model_alias)
            query_embeddings = wrapper.encode_queries(query_texts)
        else:
            query_embeddings = np.zeros((0, doc_embeddings.shape[1]), dtype=np.float32)
    else:
        return None, None

    assert doc_embeddings is not None
    dense_profile = EmbeddingProfile(model_alias=model_alias, sampled_document_count=len(doc_embeddings), sampled_query_count=len(query_texts))
    cluster_profile = ClusterProfile()

    if len(query_embeddings):
        similarity = query_embeddings @ doc_embeddings.T
        dense_profile.mean_query_similarity = float(similarity.mean())
        dense_profile.std_query_similarity = float(similarity.std())
        top_sorted = np.sort(similarity, axis=1)
        dense_profile.top1_mean_similarity = float(top_sorted[:, -1].mean())
        top5_index = max(0, top_sorted.shape[1] - min(5, top_sorted.shape[1]))
        dense_profile.top5_mean_similarity = float(top_sorted[:, top5_index:].mean())
        if top_sorted.shape[1] > 1:
            dense_profile.top1_top5_margin = float(np.mean(top_sorted[:, -1] - top_sorted[:, max(0, top_sorted.shape[1] - min(5, top_sorted.shape[1]))]))
        separation = dense_profile.top1_mean_similarity - dense_profile.mean_query_similarity
        dense_profile.dense_separation_score = _clamp((separation + 1.0) / 2.0, 0.0, 1.0)
        dense_profile.keyword_preference_score = _clamp((0.65 * exact_match_ratio) + (0.35 * query_document_overlap), 0.0, 1.0)
        dense_profile.dense_preference_score = _clamp((0.55 * semantic_need_score) + (0.45 * dense_profile.dense_separation_score), 0.0, 1.0)

    if len(doc_embeddings) >= 4:
        max_k = max(2, min(12, int(math.sqrt(len(doc_embeddings))) + 2))
        candidate_scores: list[dict[str, float | int]] = []
        for k in range(2, max_k + 1):
            try:
                labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(doc_embeddings)
                if len(set(labels)) <= 1:
                    continue
                silhouette = float(silhouette_score(doc_embeddings, labels))
                davies = float(davies_bouldin_score(doc_embeddings, labels))
                inertia = float(KMeans(n_clusters=k, random_state=42, n_init=10).fit(doc_embeddings).inertia_)
                candidate_scores.append(
                    {
                        "k": k,
                        "silhouette": silhouette,
                        "davies_bouldin": davies,
                        "inertia": inertia,
                    }
                )
            except Exception:
                continue
        if candidate_scores:
            cluster_profile.candidate_scores = candidate_scores
            ranked = sorted(candidate_scores, key=lambda item: (-float(item["silhouette"]), float(item["davies_bouldin"])))
            cluster_profile.recommended_k = int(ranked[0]["k"])

        off_diagonal = doc_embeddings @ doc_embeddings.T
        mask = ~np.eye(len(off_diagonal), dtype=bool)
        pair_mean = float(off_diagonal[mask].mean()) if mask.any() else 0.0
        if pair_mean >= 0.65:
            cluster_profile.density_label = "dense"
        elif pair_mean >= 0.35:
            cluster_profile.density_label = "balanced"
        else:
            cluster_profile.density_label = "sparse"

    return dense_profile, cluster_profile


def build_dataset_profile(
    metadata: pd.DataFrame,
    queryset: pd.DataFrame | None = None,
    *,
    text_source: str = DEFAULT_TEXT_SOURCE,
    embedding_model_alias: str | None = None,
    embeddings: np.ndarray | None = None,
) -> DatasetProfile:
    frame = metadata.copy().reset_index(drop=True)
    if frame.empty:
        return DatasetProfile(
            document_count=0,
            query_count=0,
            category_count=0,
            avg_document_length=0.0,
            document_length_std=0.0,
            avg_sentence_count=0.0,
            avg_query_length=0.0,
            query_document_overlap=0.0,
            exact_match_ratio=0.0,
            semantic_need_score=0.5,
            semantic_match_required=True,
            field_missing_rates={field: 1.0 for field in TEXT_FIELDS},
            field_avg_lengths={field: 0.0 for field in TEXT_FIELDS},
            field_avg_token_counts={field: 0.0 for field in TEXT_FIELDS},
            field_information_density={field: 0.0 for field in TEXT_FIELDS},
            field_query_overlap={field: 0.0 for field in TEXT_FIELDS},
            field_retrieval_gain={field: 0.0 for field in TEXT_FIELDS},
            lexical_field_quality={field: 0.0 for field in TEXT_FIELDS},
            vocabulary_diversity=0.0,
            dominant_language="unknown",
            language_ratios={"ko": 0.0, "en": 0.0, "mixed": 0.0, "unknown": 1.0},
            stt_quality_score=0.5,
            stt_short_ratio=0.0,
            stt_repetition_ratio=0.0,
            stt_garbled_ratio=0.0,
            stt_alignment_score=0.0,
            audio_profile=AudioProfile(),
        )

    queries = _query_frame(frame, queryset, text_source=text_source)
    transcripts = [resolve_primary_text(row, text_source=text_source) for _, row in frame.iterrows()]
    token_lists = [_tokenize(text) for text in transcripts if text]
    flat_tokens = [token for tokens in token_lists for token in tokens]
    vocabulary_diversity = (len(set(flat_tokens)) / max(1, len(flat_tokens))) if flat_tokens else 0.0

    document_lengths = [len(_tokenize(text)) for text in transcripts]
    sentence_counts = [_sentence_count(text) for text in transcripts]
    query_lengths = [len(_tokenize(query)) for query in queries["query"].fillna("").astype(str).tolist() if query]
    missing_rates, avg_lengths, avg_token_counts, information_density = _field_statistics(frame, text_source=text_source)
    field_query_overlap, field_retrieval_gain, query_doc_overlap, exact_match_ratio, corpus_coverage = _lexical_query_signals(
        frame,
        queries,
        text_source=text_source,
    )
    dominant_language, language_ratios = _language_distribution(
        transcripts[:DEFAULT_SAMPLE_LIMIT]
        + frame["title"].fillna("").astype(str).head(DEFAULT_SAMPLE_LIMIT).tolist()
    )
    stt_quality, stt_short_ratio, stt_repetition_ratio, stt_garbled_ratio, stt_alignment_score = _stt_quality(frame)
    audio_profile = _build_audio_profile(frame)

    lexical_field_quality = {
        field: _clamp(
            (0.45 * field_query_overlap[field])
            + (0.25 * (1.0 - missing_rates[field]))
            + (0.30 * field_retrieval_gain[field]),
            0.0,
            1.0,
        )
        for field in TEXT_FIELDS
    }
    semantic_need = _clamp(
        (0.45 * (1.0 - exact_match_ratio))
        + (0.30 * (1.0 - query_doc_overlap))
        + (0.25 * (1.0 - stt_quality)),
        0.0,
        1.0,
    )

    embedding_profile, cluster_profile = _embedding_profile(
        frame,
        queries,
        text_source,
        embedding_model_alias,
        embeddings,
        query_doc_overlap,
        exact_match_ratio,
        semantic_need,
    )

    return DatasetProfile(
        document_count=len(frame),
        query_count=int(len(queries)),
        category_count=int(frame["category"].fillna("").astype(str).nunique()),
        avg_document_length=float(np.mean(document_lengths)) if document_lengths else 0.0,
        document_length_std=float(np.std(document_lengths)) if document_lengths else 0.0,
        avg_sentence_count=float(np.mean(sentence_counts)) if sentence_counts else 0.0,
        avg_query_length=float(np.mean(query_lengths)) if query_lengths else 0.0,
        query_document_overlap=query_doc_overlap,
        exact_match_ratio=exact_match_ratio,
        semantic_need_score=semantic_need,
        semantic_match_required=semantic_need >= 0.5,
        field_missing_rates=missing_rates,
        field_avg_lengths=avg_lengths,
        field_avg_token_counts=avg_token_counts,
        field_information_density=information_density,
        field_query_overlap=field_query_overlap,
        field_retrieval_gain=field_retrieval_gain,
        lexical_field_quality=lexical_field_quality,
        vocabulary_diversity=vocabulary_diversity,
        dominant_language=dominant_language,
        language_ratios=language_ratios,
        stt_quality_score=stt_quality,
        stt_short_ratio=stt_short_ratio,
        stt_repetition_ratio=stt_repetition_ratio,
        stt_garbled_ratio=stt_garbled_ratio,
        stt_alignment_score=stt_alignment_score,
        audio_profile=audio_profile,
        embedding_profile=embedding_profile,
        cluster_profile=cluster_profile,
        sampled_queries=queries["query"].fillna("").astype(str).tolist(),
        sampled_document_ids=frame["id"].fillna("").astype(str).head(DEFAULT_SAMPLE_LIMIT).tolist(),
        corpus_token_coverage=corpus_coverage,
    )
