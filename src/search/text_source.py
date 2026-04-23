from __future__ import annotations

"""검색용 텍스트를 정규화하고 조합하는 공통 규칙 모음.

이 모듈은 STT 전사문, 원문 전사문, 메타데이터 필드를 어떤 방식으로 합칠지,
밀집 검색용 정규화를 얼마나 공격적으로 적용할지, 미리보기와 문장 분할을
어떻게 할지를 한곳에서 정의한다. 이 규칙이 여기 모여 있어야 검색기별로
서로 다른 텍스트 해석이 생기지 않는다.
"""

import math
import re
from functools import lru_cache
from typing import Iterable

import pandas as pd

from src.adaptive.normalization_resources import NormalizationResources, build_normalization_resources


SUPPORTED_TEXT_SOURCES = ("stt_transcript", "original_transcript", "combined")
DEFAULT_TEXT_SOURCE = "stt_transcript"
SUPPORTED_DENSE_NORMALIZATION_MODES = ("baseline", "adaptive_corpus")
DEFAULT_DENSE_NORMALIZATION_MODE = "adaptive_corpus"


@lru_cache(maxsize=1)
def _fallback_resources() -> NormalizationResources:
    """정규화 리소스가 아직 계산되지 않은 상황에서 쓸 기본 리소스를 만든다."""
    return build_normalization_resources(pd.DataFrame())


def validate_text_source(text_source: str) -> str:
    """지원하지 않는 텍스트 소스가 조용히 섞이지 않도록 초기에 검증한다."""
    if text_source not in SUPPORTED_TEXT_SOURCES:
        raise ValueError(f"Unsupported text source: {text_source}")
    return text_source


def validate_dense_normalization_mode(mode: str) -> str:
    """밀집 검색 정규화 모드가 허용된 값인지 확인한다."""
    if mode not in SUPPORTED_DENSE_NORMALIZATION_MODES:
        raise ValueError(f"Unsupported dense normalization mode: {mode}")
    return mode


def resolve_dense_normalization_mode(
    mode: str | None = None,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    """명시 설정이 없으면 적응형 리소스가 추천하는 정규화 모드를 고른다."""
    candidate = str(mode or "").strip()
    if candidate:
        return validate_dense_normalization_mode(candidate)
    adaptive_resources = resources or _fallback_resources()
    return validate_dense_normalization_mode(adaptive_resources.recommended_mode or DEFAULT_DENSE_NORMALIZATION_MODE)


def normalize_text_for_search(text: str) -> str:
    """공통 검색 전처리의 가장 바깥 단계로 공백을 단순화한다."""
    return re.sub(r"\s+", " ", str(text).strip())


def _collapse_repeated_characters(value: str) -> str:
    """노이즈성 반복 문자 길이를 줄여 STT 흔들림의 영향을 낮춘다."""
    return re.sub(r"(.)\1{3,}", r"\1\1", value)


def _collapse_repeated_tokens(value: str) -> str:
    """같은 토큰이 불필요하게 반복될 때 하나로 축약한다."""
    return re.sub(r"\b([0-9a-z\uac00-\ud7a3]+)(?:\s+\1){1,}\b", r"\1", value)


def _replace_surface(value: str, surface: str, replacement: str) -> str:
    """경계가 맞는 표면형만 치환해 부분 문자열 오염을 막는다."""
    if not surface or not replacement or surface == replacement:
        return value
    escaped = re.escape(surface)
    pattern = re.compile(rf"(?<![0-9A-Za-z\uac00-\ud7a3]){escaped}(?![0-9A-Za-z\uac00-\ud7a3])", re.IGNORECASE)
    return pattern.sub(replacement, value)


def _apply_alias_map(value: str, resources: NormalizationResources) -> str:
    """별칭 사전을 긴 표면형부터 적용해 검색 대상 표현을 통일한다."""
    alias_items = sorted(resources.alias_map.items(), key=lambda item: len(item[0]), reverse=True)
    for surface, replacement in alias_items:
        value = _replace_surface(value, str(surface).lower(), str(replacement).lower())
    return value


def _strip_filler_terms(value: str, resources: NormalizationResources) -> str:
    """의미 없는 군더더기 발화 표현을 제거해 의미 토큰 밀도를 높인다."""
    result = value
    for filler in sorted(resources.filler_terms, key=len, reverse=True):
        result = _replace_surface(result, str(filler).lower(), " ")
    return result


def _expand_spoken_years(value: str, resources: NormalizationResources) -> str:
    """말로 읽힌 연도를 숫자 표현과 함께 남겨 숫자 질의 대응력을 높인다."""
    year_suffix_words = dict(resources.number_words.get("year_suffix", {}))
    if not year_suffix_words:
        return value

    suffix_choices = "|".join(re.escape(key) for key in year_suffix_words)

    def _replace(match: re.Match[str]) -> str:
        suffix_word = str(match.group(1) or "").lower()
        suffix_value = year_suffix_words.get(suffix_word)
        if suffix_value is None:
            return match.group(0)
        return f"{match.group(0)} 20{int(suffix_value):02d}"

    return re.sub(
        rf"\btwenty\s+twenty\s+({suffix_choices})\b",
        _replace,
        value,
        flags=re.IGNORECASE,
    )


def _parse_spoken_number(
    tokens: list[str],
    start: int,
    resources: NormalizationResources,
) -> tuple[str, str, int] | None:
    """연속된 영어 숫자 어구를 실제 숫자 문자열로 해석한다.

    이 로직이 있어야 'twenty three million' 같은 발화가 숫자 질의와도
    연결될 수 있다.
    """
    simple_words = dict(resources.number_words.get("simple", {}))
    tens_words = dict(resources.number_words.get("tens", {}))
    scale_words = dict(resources.number_words.get("scales", {}))
    total = 0
    current = 0
    decimal_digits: list[str] | None = None
    index = start
    saw_number_word = False

    while index < len(tokens):
        token = re.sub(r"[^a-z]", "", tokens[index].lower())
        if not token:
            break
        if token in simple_words:
            saw_number_word = True
            value = int(simple_words[token])
            if decimal_digits is not None:
                decimal_digits.append(str(value))
            else:
                current += value
            index += 1
            continue
        if token in tens_words:
            saw_number_word = True
            value = int(tens_words[token])
            if decimal_digits is not None:
                decimal_digits.append(str(value))
            else:
                current += value
            index += 1
            continue
        if token == "hundred" and decimal_digits is None:
            saw_number_word = True
            current = max(1, current) * 100
            index += 1
            continue
        if token in scale_words and decimal_digits is None:
            saw_number_word = True
            total += max(1, current) * int(scale_words[token])
            current = 0
            index += 1
            continue
        if token == "point" and decimal_digits is None and saw_number_word:
            decimal_digits = []
            total += current
            current = 0
            index += 1
            continue
        break

    consumed = index - start
    if not saw_number_word or consumed < 2:
        return None

    raw_text = " ".join(tokens[start:index])
    if decimal_digits is None:
        numeric_text = str(int(total + current))
    else:
        numeric_text = f"{int(total)}.{''.join(decimal_digits)}".rstrip("0").rstrip(".")
    return raw_text, numeric_text, consumed


def _expand_spoken_numbers(value: str, resources: NormalizationResources) -> str:
    """말로 풀어 읽힌 수사를 원문+숫자 병기 형태로 확장한다."""
    tokens = value.split()
    if not tokens:
        return value

    normalized_tokens: list[str] = []
    index = 0
    while index < len(tokens):
        parsed = _parse_spoken_number(tokens, index, resources)
        if parsed is None:
            normalized_tokens.append(tokens[index])
            index += 1
            continue

        raw_text, numeric_text, consumed = parsed
        normalized_tokens.append(raw_text)
        if numeric_text and numeric_text != raw_text:
            normalized_tokens.append(numeric_text)
        index += consumed

    expanded = " ".join(normalized_tokens)
    scale_words = dict(resources.number_words.get("scales", {}))
    if not scale_words:
        return expanded
    scale_choices = "|".join(re.escape(key) for key in scale_words)

    def _replace(match: re.Match[str]) -> str:
        number_text = str(match.group(1))
        scale_word = str(match.group(2)).lower()
        factor = int(scale_words.get(scale_word, 1))
        try:
            expanded_number = int(float(number_text) * factor)
        except ValueError:
            return match.group(0)
        return f"{match.group(0)} {expanded_number}"

    return re.sub(
        rf"\b([0-9]+(?:\.[0-9]+)?)\s+({scale_choices})\b",
        _replace,
        expanded,
        flags=re.IGNORECASE,
    )


def _normalize_dense_core(text: str) -> str:
    """밀집 검색 전에 공통적으로 적용할 핵심 정규화 단계만 수행한다."""
    value = normalize_text_for_search(text).lower()
    if not value:
        return value
    value = re.sub(r"([0-9])[, ]+([0-9])", r"\1\2", value)
    value = _collapse_repeated_characters(value)
    value = _collapse_repeated_tokens(value)
    return value


def normalize_text_for_dense(
    text: str,
    mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    """밀집 검색용 문서를 정규화한다.

    adaptive_corpus 모드에서는 별칭, 숫자, filler 정리를 더 공격적으로 수행해
    STT 특유의 표현 흔들림을 줄인다.
    """
    resolved_mode = resolve_dense_normalization_mode(mode, resources=resources)
    adaptive_resources = resources or _fallback_resources()
    value = _normalize_dense_core(text)
    if resolved_mode == "adaptive_corpus":
        value = _apply_alias_map(value, adaptive_resources)
        value = _expand_spoken_years(value, adaptive_resources)
        value = _expand_spoken_numbers(value, adaptive_resources)
        value = _strip_filler_terms(value, adaptive_resources)
        value = _collapse_repeated_tokens(value)
    value = re.sub(r"[^0-9a-z\uac00-\ud7a3\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_text_for_reranker(
    text: str,
    mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    """리랭커 입력용 텍스트를 정규화한다.

    리랭커는 구조 힌트 구분자도 읽기 때문에 dense 정규화와 비슷하되
    허용 문자를 조금 더 넓게 둔다.
    """
    resolved_mode = resolve_dense_normalization_mode(mode, resources=resources)
    adaptive_resources = resources or _fallback_resources()
    value = _normalize_dense_core(text)
    if resolved_mode == "adaptive_corpus":
        value = _apply_alias_map(value, adaptive_resources)
        value = _expand_spoken_years(value, adaptive_resources)
        value = _expand_spoken_numbers(value, adaptive_resources)
        value = _strip_filler_terms(value, adaptive_resources)
    value = re.sub(r"[^0-9a-z\uac00-\ud7a3\s:|/\-]", " ", value)
    return normalize_text_for_search(value)


def prepare_query_for_dense(
    query: str,
    mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    """질의를 밀집 검색용 규칙으로 정규화하는 얇은 래퍼."""
    return normalize_text_for_dense(query, mode=mode, resources=resources)


def prepare_query_for_reranker(
    query: str,
    mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    """질의를 리랭커 입력 규칙으로 정규화하는 얇은 래퍼."""
    return normalize_text_for_reranker(query, mode=mode, resources=resources)


def _safe_value(row: pd.Series, column: str) -> str:
    """행 컬럼 값을 NaN 안전하게 문자열로 읽는다."""
    value = row.get(column, "")
    if pd.isna(value):
        return ""
    return str(value)


def resolve_primary_text(row: pd.Series, text_source: str = DEFAULT_TEXT_SOURCE) -> str:
    """현재 검색 모드에서 대표 본문으로 쓸 텍스트를 선택한다.

    STT와 원문 중 무엇을 대표 본문으로 볼지 한곳에서 결정해야
    검색기, 미리보기, 문장 anchor가 같은 본문을 바라본다.
    """
    validate_text_source(text_source)
    original_text = normalize_text_for_search(_safe_value(row, "original_transcript"))
    stt_text = normalize_text_for_search(_safe_value(row, "stt_transcript"))
    if text_source == "original_transcript":
        return original_text
    if text_source == "combined":
        return normalize_text_for_search(" ".join(part for part in [stt_text, original_text] if part))
    return stt_text or original_text


def _join_non_empty(parts: Iterable[str]) -> str:
    """비어 있지 않은 조각만 이어 붙여 검색 텍스트를 깔끔하게 만든다."""
    return " ".join(part for part in parts if part).strip()


def build_search_text(
    row: pd.Series,
    text_source: str = DEFAULT_TEXT_SOURCE,
    for_dense: bool = False,
    normalization_mode: str = DEFAULT_DENSE_NORMALIZATION_MODE,
    *,
    resources: NormalizationResources | None = None,
) -> str:
    """메타데이터 행을 검색기 입력 문자열로 조립한다.

    어휘 검색은 사람이 읽는 표면형을 최대한 보존하고,
    밀집 검색은 의미 중심 정규화를 거친 필드를 합친다.
    """
    primary_text = resolve_primary_text(row, text_source=text_source)
    if not for_dense:
        return _join_non_empty(
            [
                normalize_text_for_search(_safe_value(row, "title")),
                normalize_text_for_search(_safe_value(row, "description")),
                normalize_text_for_search(_safe_value(row, "tags")),
                normalize_text_for_search(_safe_value(row, "category")),
                normalize_text_for_search(_safe_value(row, "keywords")),
                normalize_text_for_search(primary_text),
            ]
        )

    # 밀집 검색에서는 대표 본문 외에 반대편 전사문도 함께 넣어 두어
    # STT/원문 중 한쪽에만 남은 표현 차이를 완화한다.
    alternate_text_source = "original_transcript" if text_source != "original_transcript" else "stt_transcript"
    alternate_text = resolve_primary_text(row, text_source=alternate_text_source)
    dense_parts = [
        normalize_text_for_dense(_safe_value(row, "title"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(_safe_value(row, "description"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(_safe_value(row, "tags"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(_safe_value(row, "category"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(_safe_value(row, "keywords"), mode=normalization_mode, resources=resources),
        normalize_text_for_dense(primary_text, mode=normalization_mode, resources=resources),
    ]
    normalized_alternate = normalize_text_for_dense(alternate_text, mode=normalization_mode, resources=resources)
    if normalized_alternate:
        dense_parts.append(normalized_alternate)
    return _join_non_empty(dense_parts)


def _adaptive_preview_length(text: str, requested_length: int | None = None) -> int:
    """본문 길이와 문장 구조를 보고 미리보기 길이를 적응적으로 정한다."""
    if requested_length is not None:
        return max(40, int(requested_length))
    normalized = normalize_text_for_search(text)
    if not normalized:
        return 80
    sentence_lengths = [len(part) for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    avg_sentence = sum(sentence_lengths) / max(1, len(sentence_lengths)) if sentence_lengths else len(normalized)
    target = avg_sentence + max(40.0, len(normalized) / max(1.0, len(sentence_lengths) or 1.0))
    return int(max(80, round(target)))


def build_preview_text(row: pd.Series, text_source: str = DEFAULT_TEXT_SOURCE, length: int | None = None) -> str:
    """대표 본문에서 UI용 미리보기 텍스트를 잘라 만든다."""
    preview = resolve_primary_text(row, text_source=text_source)
    resolved_length = _adaptive_preview_length(preview, requested_length=length)
    return preview[:resolved_length] + ("..." if len(preview) > resolved_length else "")


def split_line_into_sentences(text: str) -> list[str]:
    """한 줄 텍스트를 문장 단위로 나눈다."""
    normalized = normalize_text_for_search(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [normalize_text_for_search(part) for part in parts if normalize_text_for_search(part)]
    return sentences or [normalized]


def _logical_line_limits(sentences: list[str]) -> tuple[int, int]:
    """문장 묶음을 논리 줄로 재구성할 때 사용할 길이 한계를 계산한다."""
    if not sentences:
        return 120, 2
    avg_sentence = sum(len(sentence) for sentence in sentences) / max(1, len(sentences))
    max_chars = int(max(120, round(avg_sentence * max(1.5, len(sentences) / max(1.0, math.log1p(len(sentences) + 1.0))))))
    max_sentences = max(2, int(round(len(sentences) / max(1.0, math.sqrt(len(sentences))))))
    return max_chars, max_sentences


def _chunk_sentences_into_logical_lines(sentences: list[str]) -> list[str]:
    """짧은 문장을 적절히 묶어 사람이 읽기 좋은 논리 줄을 만든다."""
    if not sentences:
        return []
    max_chars, max_sentences = _logical_line_limits(sentences)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        projected_length = current_length + sentence_length + (1 if current_sentences else 0)
        if current_sentences and (len(current_sentences) >= max_sentences or projected_length > max_chars):
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_length = 0
        current_sentences.append(sentence)
        current_length += sentence_length + (1 if len(current_sentences) > 1 else 0)
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def split_text_into_lines(text: str) -> list[dict[str, str | int]]:
    """원문 줄과 논리 줄 정보를 함께 보존한 라인 구조를 만든다."""
    rows: list[dict[str, str | int]] = []
    raw_lines = [
        (source_line_number, normalize_text_for_search(raw_line))
        for source_line_number, raw_line in enumerate(str(text).splitlines(), start=1)
    ]
    for source_line_number, line_text in raw_lines:
        if not line_text:
            continue
        logical_lines = _chunk_sentences_into_logical_lines(split_line_into_sentences(line_text)) or [line_text]
        for logical_line in logical_lines:
            rows.append(
                {
                    "line_number": len(rows) + 1,
                    "line_text": logical_line,
                    "source_line_number": source_line_number,
                }
            )
    return rows


def build_sentence_segments(text: str) -> list[dict[str, str | int]]:
    """라인 정보를 유지한 채 문장 단위 세그먼트 목록을 만든다."""
    segments: list[dict[str, str | int]] = []
    for line in split_text_into_lines(text):
        sentences = split_line_into_sentences(str(line["line_text"]))
        for sentence_number, sentence_text in enumerate(sentences, start=1):
            segments.append(
                {
                    "line_number": int(line["line_number"]),
                    "source_line_number": int(line["source_line_number"]),
                    "sentence_number": sentence_number,
                    "line_text": str(line["line_text"]),
                    "sentence_text": sentence_text,
                }
            )
    return segments


def text_source_suffix(text_source: str) -> str:
    """텍스트 소스 이름을 아티팩트 파일명에 안전한 접미어로 바꾼다."""
    validate_text_source(text_source)
    return text_source.replace("/", "_")
