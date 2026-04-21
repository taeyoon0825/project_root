from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.adaptive.parameter_resolver import (
    build_adaptive_context,
    build_static_reference_context,
    resolve_query_search_config,
    resolve_top_k,
)
from src.config import EVALUATION_DIR, ensure_project_dirs
from src.embedding.build_indices import DenseSearchEngine
from src.evaluation.evaluate import evaluate_all
from src.search.keyword_search import KeywordSearchEngine
from src.search.query_preview import extract_dense_preview, extract_keyword_preview


@dataclass
class DatasetBundle:
    name: str
    kind: str
    metadata: pd.DataFrame
    queryset: pd.DataFrame


def _mk_metadata(rows: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "file_name" not in frame.columns:
        frame["file_name"] = frame["id"].astype(str).str.lower() + ".txt"
    if "source_type" not in frame.columns:
        frame["source_type"] = "synthetic_dummy"
    return frame


def _mk_queryset(rows: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "query_id" not in frame.columns:
        frame["query_id"] = [f"q_{i+1:03d}" for i in range(len(frame))]
    return frame


def build_datasets() -> list[DatasetBundle]:
    ko_meta = _mk_metadata(
        [
            {
                "id": "KO-001",
                "category": "ai",
                "title": "한국어 음성 인식 파이프라인",
                "description": "Whisper 기반 한국어 STT 품질 개선 실험",
                "tags": "한국어,whisper,stt,품질",
                "keywords": "한국어 whisper stt",
                "original_transcript": "한국어 음성 인식 모델을 fine tuning 하여 발화 단위 정확도를 높였습니다. 잡음 제거 전처리가 핵심입니다.",
                "stt_transcript": "한국어 음성 인식 모델을 파인 튜닝하여 발화 단위 정확도를 높였습니다. 잡음 제거 전처리가 핵심입니다.",
            },
            {
                "id": "KO-002",
                "category": "search",
                "title": "키워드 검색 가중치 튜닝",
                "description": "title tags description transcript 가중치 실험",
                "tags": "bm25,가중치,랭킹",
                "keywords": "검색 가중치 bm25",
                "original_transcript": "검색 품질은 title 과 tags 신호가 강할 때 향상됩니다. transcript는 긴 문맥 질의에서 중요합니다.",
                "stt_transcript": "검색 품질은 title 과 tags 신호가 강할 때 향상됩니다. transcript 는 긴 문맥 질의에서 중요합니다.",
            },
            {
                "id": "KO-003",
                "category": "audio",
                "title": "오디오 샘플레이트 비교",
                "description": "16000과 22050 샘플레이트 비교",
                "tags": "오디오,sample_rate",
                "keywords": "sample rate 16000",
                "original_transcript": "샘플레이트 16000에서는 음성 정보가 충분하지만 고주파 성분은 제한됩니다.",
                "stt_transcript": "샘플레이트 16000 에서는 음성 정보가 충분하지만 고주파 성분은 제한됩니다.",
            },
            {
                "id": "KO-004",
                "category": "evaluation",
                "title": "환각 탐지 임계값",
                "description": "hallucination threshold 동적 조정",
                "tags": "평가,threshold,hallucination",
                "keywords": "환각 임계값",
                "original_transcript": "점수 분포의 median 과 MAD를 사용하면 임계값을 데이터별로 안정적으로 설정할 수 있습니다.",
                "stt_transcript": "점수 분포의 median 과 mad 를 사용하면 임계값을 데이터별로 안정적으로 설정할 수 있습니다.",
            },
            {
                "id": "KO-005",
                "category": "ui",
                "title": "검색 결과 프리뷰 스니펫",
                "description": "질의 중심 발췌 미리보기",
                "tags": "preview,snippet,ui",
                "keywords": "스니펫 프리뷰",
                "original_transcript": "기존에는 검색어 그대로 프리뷰에 노출됐지만 이제는 원문에서 매칭 구간을 발췌합니다.",
                "stt_transcript": "기존에는 검색어 그대로 프리뷰에 노출됐지만 이제는 원문에서 매칭 구간을 발췌합니다.",
            },
            {
                "id": "KO-006",
                "category": "query",
                "title": "짧은 질의와 긴 질의",
                "description": "query-level alpha 조정 사례",
                "tags": "alpha,keyword,dense",
                "keywords": "query alpha",
                "original_transcript": "짧은 키워드 질의는 lexical exactness가 높아 keyword 비중이 커집니다. 긴 질문은 dense 비중이 증가합니다.",
                "stt_transcript": "짧은 키워드 질의는 lexical exactness 가 높아 keyword 비중이 커집니다. 긴 질문은 dense 비중이 증가합니다.",
            },
        ]
    )
    ko_q = _mk_queryset(
        [
            {"query_id": "ko_q1", "query": "한국어 whisper stt 품질 개선", "relevant_id": "KO-001"},
            {"query_id": "ko_q2", "query": "title tags 가중치 랭킹", "relevant_id": "KO-002"},
            {"query_id": "ko_q3", "query": "sample rate 16000 제한", "relevant_id": "KO-003"},
            {"query_id": "ko_q4", "query": "median MAD 환각 임계값", "relevant_id": "KO-004"},
            {"query_id": "ko_q5", "query": "프리뷰 스니펫 원문 발췌", "relevant_id": "KO-005"},
            {"query_id": "ko_q6", "query": "짧은 질의 긴 질문 alpha", "relevant_id": "KO-006"},
        ]
    )

    en_meta = _mk_metadata(
        [
            {
                "id": "EN-001",
                "category": "nlp",
                "title": "Entity linking with product codes",
                "description": "Handling SKU IDs and proper nouns in retrieval",
                "tags": "entity,sku,retrieval",
                "keywords": "SKU-9912 ZX-5",
                "original_transcript": "Short exact queries containing SKU-9912 should prioritize lexical matching over semantic expansion.",
                "stt_transcript": "Short exact queries containing sku 9912 should prioritize lexical matching over semantic expansion.",
            },
            {
                "id": "EN-002",
                "category": "qa",
                "title": "Long-form question answering",
                "description": "Dense retrieval for explanatory questions",
                "tags": "question,semantic,dense",
                "keywords": "why how explanation",
                "original_transcript": "When users ask why a ranking changed after adaptation, dense semantic signals become more useful than sparse token overlap.",
                "stt_transcript": "When users ask why a ranking changed after adaptation dense semantic signals become more useful than sparse token overlap.",
            },
            {
                "id": "EN-003",
                "category": "metrics",
                "title": "Soft metric weighting",
                "description": "Balancing exact match and overlap",
                "tags": "metrics,precision,recall",
                "keywords": "soft metric weights",
                "original_transcript": "Soft precision and recall should change with dataset exact match ratio and query-document overlap.",
                "stt_transcript": "Soft precision and recall should change with dataset exact match ratio and query document overlap.",
            },
            {
                "id": "EN-004",
                "category": "audio",
                "title": "English TTS and sample rate",
                "description": "Voice defaults for edge provider",
                "tags": "tts,voice,sample-rate",
                "keywords": "en-US-AriaNeural",
                "original_transcript": "For English dominant sets, the voice should resolve to en-US-AriaNeural with appropriate sample rate.",
                "stt_transcript": "For English dominant sets the voice should resolve to en US AriaNeural with appropriate sample rate.",
            },
            {
                "id": "EN-005",
                "category": "clustering",
                "title": "Adaptive cluster sizing",
                "description": "Choosing cluster_k from category and density",
                "tags": "cluster,kmeans,hdbscan",
                "keywords": "cluster k",
                "original_transcript": "Cluster count should scale with document count and category diversity, not stay fixed at six.",
                "stt_transcript": "Cluster count should scale with document count and category diversity not stay fixed at six.",
            },
            {
                "id": "EN-006",
                "category": "ui",
                "title": "Snippet preview extraction",
                "description": "Show source chunk instead of query echo",
                "tags": "preview,snippet",
                "keywords": "query preview",
                "original_transcript": "A useful preview should include the matched sentence fragment from source text rather than repeating the query verbatim.",
                "stt_transcript": "A useful preview should include the matched sentence fragment from source text rather than repeating the query verbatim.",
            },
        ]
    )
    en_q = _mk_queryset(
        [
            {"query_id": "en_q1", "query": "SKU-9912 lexical matching", "relevant_id": "EN-001"},
            {"query_id": "en_q2", "query": "why did ranking change after adaptation", "relevant_id": "EN-002"},
            {"query_id": "en_q3", "query": "soft precision recall exact match ratio", "relevant_id": "EN-003"},
            {"query_id": "en_q4", "query": "English voice en-US-AriaNeural sample rate", "relevant_id": "EN-004"},
            {"query_id": "en_q5", "query": "cluster count not fixed at six", "relevant_id": "EN-005"},
            {"query_id": "en_q6", "query": "preview snippet instead of query echo", "relevant_id": "EN-006"},
        ]
    )

    mixed_meta = _mk_metadata(
        [
            {
                "id": "MX-001",
                "category": "stt",
                "title": "회의록 STT 오류 사례",
                "description": "한국어와 영어 코드스위칭 발화",
                "tags": "stt,error,mixed",
                "keywords": "회의록 error",
                "original_transcript": "회의에서 speaker가 budget forecast를 설명했고, Q3 revenue는 one point two million 이라고 말했다.",
                "stt_transcript": "회이에서 spiker가 badget forcast를 설명했고 q3 rebenue는 one point to milion 이라고 말했다",
            },
            {
                "id": "MX-002",
                "category": "stt",
                "title": "발화형 질의 복원",
                "description": "음절 누락과 철자 오염",
                "tags": "spoken,query,noise",
                "keywords": "발화형 질의",
                "original_transcript": "사용자가 말하듯 입력한 질의는 문법이 불완전해도 의미 매칭으로 복원할 수 있습니다.",
                "stt_transcript": "사용자가 말하듯 입력한 질의는 문법이 불완전해도 의미 매칭으로 복원 할수 잇습니다",
            },
            {
                "id": "MX-003",
                "category": "entity",
                "title": "고유명사 인식 실패",
                "description": "Project Helios 2026",
                "tags": "entity,proper-noun",
                "keywords": "Helios2026",
                "original_transcript": "Project Helios 2026 milestone B requires compliance review before release.",
                "stt_transcript": "projet helios twenty twenty six mile stone b requires complience review before release",
            },
            {
                "id": "MX-004",
                "category": "qa",
                "title": "긴 혼합 언어 질문",
                "description": "why/how 질문 처리",
                "tags": "long-question,dense",
                "keywords": "long why how",
                "original_transcript": "긴 질문에서는 키워드 일치보다 문맥적 유사도가 중요하며, mixed language에서는 dense retrieval의 안정성이 높습니다.",
                "stt_transcript": "긴 질문에서는 키워드 일치보다 문맥적 유사도가 중요하며 mixed language에서는 dense retrieval 안정성이 높습니다",
            },
            {
                "id": "MX-005",
                "category": "metrics",
                "title": "threshold fallback test",
                "description": "score 분포가 평평할 때",
                "tags": "fallback,threshold",
                "keywords": "median mad fallback",
                "original_transcript": "점수 분포가 좁으면 MAD 기반 임계값이 base threshold와 혼합되어 과도한 환각 판정을 막습니다.",
                "stt_transcript": "점수 분포가 좁으면 mad 기반 임계값이 base threshold와 혼합되어 과도한 환각 판정을 막습니다",
            },
            {
                "id": "MX-006",
                "category": "ui",
                "title": "노이즈 환경 preview",
                "description": "원문 발췌 확인",
                "tags": "snippet,noise",
                "keywords": "preview noise",
                "original_transcript": "검색 결과 preview는 오탈자가 많은 STT보다 원문 맥락을 보존하는 발췌가 더 이해하기 쉽습니다.",
                "stt_transcript": "검색 결과 previw는 오탈자가 많은 stt보다 원문 맥락을 보존하는 발췌가 더 이해 하기 쉽습니다",
            },
        ]
    )
    mixed_q = _mk_queryset(
        [
            {"query_id": "mx_q1", "query": "q3 revenue one point two million", "relevant_id": "MX-001"},
            {"query_id": "mx_q2", "query": "말하듯 입력한 질의 복원", "relevant_id": "MX-002"},
            {"query_id": "mx_q3", "query": "Project Helios 2026 milestone B", "relevant_id": "MX-003"},
            {
                "query_id": "mx_q4",
                "query": "왜 mixed language에서는 dense retrieval 안정성이 높아지는가?",
                "relevant_id": "MX-004",
            },
            {"query_id": "mx_q5", "query": "median MAD threshold fallback", "relevant_id": "MX-005"},
            {"query_id": "mx_q6", "query": "preview 발췌가 더 이해하기 쉬운 이유", "relevant_id": "MX-006"},
        ]
    )
    return [
        DatasetBundle(name="ko_centered", kind="한국어 중심", metadata=ko_meta, queryset=ko_q),
        DatasetBundle(name="en_centered", kind="영어 중심", metadata=en_meta, queryset=en_q),
        DatasetBundle(name="mixed_low_stt", kind="혼합 언어 / STT 저품질", metadata=mixed_meta, queryset=mixed_q),
    ]


def _ctx_row(dataset: DatasetBundle, mode: str, context: Any) -> dict[str, Any]:
    return {
        "dataset": dataset.name,
        "dataset_kind": dataset.kind,
        "mode": mode,
        "field_weights_title": context.search.field_weights.get("title", 0.0),
        "field_weights_tags": context.search.field_weights.get("tags", 0.0),
        "field_weights_description": context.search.field_weights.get("description", 0.0),
        "field_weights_transcript": context.search.field_weights.get("transcript", 0.0),
        "keyword_alpha": context.search.keyword_alpha,
        "dense_alpha": context.search.dense_alpha,
        "hallucination_threshold": context.metric.hallucination_threshold,
        "soft_weight_exact_match": context.metric.soft_accuracy_weights.get("exact_match", 0.0),
        "soft_weight_search_score": context.metric.soft_accuracy_weights.get("search_score", 0.0),
        "soft_weight_text_overlap": context.metric.soft_accuracy_weights.get("text_overlap", 0.0),
        "soft_weight_rank_weight": context.metric.soft_accuracy_weights.get("rank_weight", 0.0),
        "soft_precision_exact_weight": context.metric.soft_precision_exact_weight,
        "soft_recall_exact_weight": context.metric.soft_recall_exact_weight,
        "language": context.language.dominant_language,
        "whisper_language": context.language.whisper_language,
        "whisper_model": context.language.whisper_model,
        "tts_provider": context.language.tts_provider,
        "tts_voice": context.language.edge_voice,
        "sample_rate": context.language.sample_rate,
        "cluster_k": context.cluster.n_clusters,
        "preview_length": context.visualization.preview_length,
        "resolved_top_k": resolve_top_k(context.profile),
    }


def _build_param_rows(dataset: DatasetBundle) -> list[dict[str, Any]]:
    adaptive = build_adaptive_context(dataset.metadata, dataset.queryset, text_source="stt_transcript")
    static = build_static_reference_context(dataset.metadata, dataset.queryset, text_source="stt_transcript")
    return [_ctx_row(dataset, "static_reference", static), _ctx_row(dataset, "adaptive", adaptive)]


def _metric_delta_rows(dataset: DatasetBundle, summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[
        [
            "system_name",
            "parameter_mode",
            "accuracy_at_1",
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "mrr_at_k",
            "ndcg_at_k",
            "topk_hit_rate",
        ]
    ].copy()
    static = selected[selected["parameter_mode"] == "static_reference"].drop(columns=["parameter_mode"])
    adaptive = selected[selected["parameter_mode"] == "adaptive"].drop(columns=["parameter_mode"])
    merged = static.merge(adaptive, on="system_name", suffixes=("_static", "_adaptive"))
    for metric in ["accuracy_at_1", "precision_at_k", "recall_at_k", "f1_at_k", "mrr_at_k", "ndcg_at_k", "topk_hit_rate"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_adaptive"] - merged[f"{metric}_static"]
    merged.insert(0, "dataset", dataset.name)
    merged.insert(1, "dataset_kind", dataset.kind)
    return merged


def _preview_examples_for_dataset(dataset: DatasetBundle) -> pd.DataFrame:
    adaptive_context = build_adaptive_context(dataset.metadata, dataset.queryset, text_source="stt_transcript")
    rows: list[dict[str, Any]] = []
    keyword_engine = KeywordSearchEngine(dataset.metadata, text_source="stt_transcript", adaptive_context=adaptive_context)
    dense_minilm = DenseSearchEngine(
        dataset.metadata,
        "paraphrase-multilingual-MiniLM-L12-v2",
        text_source="stt_transcript",
        artifact_namespace=f"val_{dataset.name}",
        adaptive_context=adaptive_context,
    )
    dense_e5 = DenseSearchEngine(
        dataset.metadata,
        "multilingual-e5-base",
        text_source="stt_transcript",
        artifact_namespace=f"val_{dataset.name}",
        adaptive_context=adaptive_context,
    )
    dense_minilm.load()
    dense_e5.load()

    sample_queries = dataset.queryset.head(2).copy()
    for _, q in sample_queries.iterrows():
        query = str(q["query"])
        old_preview = query[:80] + ("..." if len(query) > 80 else "")

        bm25_row = keyword_engine.search(query, top_k=1, method="bm25").iloc[0]
        bm25_payload = {"best_match_line_text": bm25_row.get("best_match_text", ""), "best_match_text": bm25_row.get("best_match_text", "")}
        bm25_new = extract_keyword_preview(
            str(bm25_row.get("stt_transcript", "")),
            query,
            method="bm25",
            length=int(bm25_row.get("adaptive_preview_length", 160)),
            match_payload=bm25_payload,
        )
        rows.append(
            {
                "dataset": dataset.name,
                "query": query,
                "engine": "BM25",
                "matched_document": bm25_row.get("id", ""),
                "old_preview": old_preview,
                "new_preview": bm25_new,
            }
        )

        mini_row = dense_minilm.search(query, top_k=1).iloc[0]
        mini_payload = {"best_match_line_text": mini_row.get("best_match_text", ""), "best_match_text": mini_row.get("best_match_text", "")}
        mini_new = extract_dense_preview(
            str(mini_row.get("stt_transcript", "")),
            query,
            model=None,
            length=int(mini_row.get("adaptive_preview_length", 160)),
            match_payload=mini_payload,
        )
        rows.append(
            {
                "dataset": dataset.name,
                "query": query,
                "engine": "Dense/paraphrase-multilingual-MiniLM-L12-v2",
                "matched_document": mini_row.get("id", ""),
                "old_preview": old_preview,
                "new_preview": mini_new,
            }
        )

        e5_row = dense_e5.search(query, top_k=1).iloc[0]
        e5_payload = {"best_match_line_text": e5_row.get("best_match_text", ""), "best_match_text": e5_row.get("best_match_text", "")}
        e5_new = extract_dense_preview(
            str(e5_row.get("stt_transcript", "")),
            query,
            model=None,
            length=int(e5_row.get("adaptive_preview_length", 160)),
            match_payload=e5_payload,
        )
        rows.append(
            {
                "dataset": dataset.name,
                "query": query,
                "engine": "Dense/multilingual-e5-base",
                "matched_document": e5_row.get("id", ""),
                "old_preview": old_preview,
                "new_preview": e5_new,
            }
        )
    return pd.DataFrame(rows)


def _query_alpha_rows(dataset: DatasetBundle) -> pd.DataFrame:
    adaptive_context = build_adaptive_context(dataset.metadata, dataset.queryset, text_source="stt_transcript")
    query_bank = [
        ("짧은 키워드 질의", "whisper 품질"),
        ("긴 자연어 질문", "왜 mixed language 데이터에서는 dense retrieval이 keyword보다 안정적으로 동작하는가?"),
        ("숫자/고유명사 포함 질의", "Project Helios 2026 milestone B"),
        ("STT 오류 발화형 질의", "회이에서 badget forcast q3 rebenue 얘기한 내용"),
    ]
    rows: list[dict[str, Any]] = []
    for qtype, query in query_bank:
        conf = resolve_query_search_config(query, adaptive_context)
        rows.append(
            {
                "dataset": dataset.name,
                "dataset_kind": dataset.kind,
                "query_type": qtype,
                "query": query,
                "keyword_alpha": conf.keyword_alpha,
                "dense_alpha": conf.dense_alpha,
                "keyword_ranker_weight": conf.keyword_ranker_weight,
                "dense_semantic_weight": conf.dense_semantic_weight,
                "reasoning": conf.reasoning,
            }
        )
    return pd.DataFrame(rows)


def _failure_rows(mode_compare: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy_at_1_delta",
        "precision_at_k_delta",
        "recall_at_k_delta",
        "f1_at_k_delta",
        "mrr_at_k_delta",
        "ndcg_at_k_delta",
        "soft_f1_at_k_delta",
    ]
    metrics = [column for column in metrics if column in mode_compare.columns]
    if not metrics:
        return pd.DataFrame()
    frame = mode_compare.copy()
    frame["worst_delta"] = frame[metrics].min(axis=1)
    return frame[frame["worst_delta"] < 0].sort_values("worst_delta").reset_index(drop=True)


def _json_like(value: Any) -> str:
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return ", ".join(f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in parsed.items())
        except Exception:
            return value
    return str(value)


def _write_markdown_report(
    path: Path,
    dataset_rows: pd.DataFrame,
    metric_rows: pd.DataFrame,
    preview_rows: pd.DataFrame,
    alpha_rows: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Adaptive Validation Report")
    lines.append("")
    lines.append("본 리포트는 static_reference 대비 adaptive 동작을 데이터셋/쿼리 단위로 실증 검증한 결과입니다.")
    lines.append("")
    lines.append("## 1) 데이터셋 구성")
    for dataset_name, frame in dataset_rows.groupby("dataset"):
        kind = frame["dataset_kind"].iloc[0]
        lines.append(f"- `{dataset_name}`: {kind}")
    lines.append("")
    lines.append("## 2) 데이터셋별 adaptive resolved 값")
    lines.append(dataset_rows.to_markdown(index=False))
    lines.append("")
    lines.append("## 3) static vs adaptive 지표 비교")
    lines.append(metric_rows.to_markdown(index=False))
    lines.append("")
    lines.append("## 4) query preview 비교 (old vs new)")
    lines.append(preview_rows.to_markdown(index=False))
    lines.append("")
    lines.append("## 5) query-level adaptive alpha")
    lines.append(alpha_rows.to_markdown(index=False))
    lines.append("")
    lines.append("## 6) 실패 사례")
    if failures.empty:
        lines.append("- 관측된 지표 기준으로 adaptive가 static보다 나빠진 케이스가 없습니다.")
    else:
        lines.append(failures.to_markdown(index=False))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_project_dirs()
    out_dir = EVALUATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets()

    param_rows: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []
    preview_frames: list[pd.DataFrame] = []
    alpha_frames: list[pd.DataFrame] = []
    mode_compare_frames: list[pd.DataFrame] = []

    for dataset in datasets:
        namespace = f"validation_{dataset.name}"
        param_rows.extend(_build_param_rows(dataset))
        outputs = evaluate_all(
            metadata=dataset.metadata,
            queryset=dataset.queryset,
            text_sources=("stt_transcript",),
            include_optional=False,
            artifact_namespace=namespace,
            include_static_reference=True,
        )
        summary = outputs["summary"].loc[outputs["summary"]["text_source"] == "stt_transcript"].reset_index(drop=True)
        metric_frames.append(_metric_delta_rows(dataset, summary))
        mode_compare = outputs["mode_comparison"].copy()
        mode_compare.insert(0, "dataset", dataset.name)
        mode_compare.insert(1, "dataset_kind", dataset.kind)
        mode_compare_frames.append(mode_compare)
        preview_frames.append(_preview_examples_for_dataset(dataset))
        alpha_frames.append(_query_alpha_rows(dataset))

    param_df = pd.DataFrame(param_rows)
    metric_df = pd.concat(metric_frames, ignore_index=True)
    preview_df = pd.concat(preview_frames, ignore_index=True)
    alpha_df = pd.concat(alpha_frames, ignore_index=True)
    mode_compare_df = pd.concat(mode_compare_frames, ignore_index=True)
    failures = _failure_rows(mode_compare_df)

    param_df.to_csv(out_dir / "adaptive_param_comparison.csv", index=False, encoding="utf-8-sig")
    metric_df.to_csv(out_dir / "static_vs_adaptive_metrics.csv", index=False, encoding="utf-8-sig")
    preview_df.to_csv(out_dir / "query_preview_comparison.csv", index=False, encoding="utf-8-sig")
    alpha_df.to_csv(out_dir / "query_level_alpha_analysis.csv", index=False, encoding="utf-8-sig")
    mode_compare_df.to_csv(out_dir / "static_vs_adaptive_mode_comparison_full.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(out_dir / "adaptive_failure_cases.csv", index=False, encoding="utf-8-sig")

    report_path = out_dir / "adaptive_validation_report.md"
    _write_markdown_report(report_path, param_df, metric_df, preview_df, alpha_df, failures)

    print(f"[DONE] report={report_path}")
    print(f"[DONE] params={out_dir / 'adaptive_param_comparison.csv'}")
    print(f"[DONE] metrics={out_dir / 'static_vs_adaptive_metrics.csv'}")
    print(f"[DONE] previews={out_dir / 'query_preview_comparison.csv'}")
    print(f"[DONE] alpha={out_dir / 'query_level_alpha_analysis.csv'}")
    print(f"[DONE] failures={out_dir / 'adaptive_failure_cases.csv'}")
    if not failures.empty:
        print("[INFO] failure case sample:")
        print(failures[["dataset", "system_name", "worst_delta"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
