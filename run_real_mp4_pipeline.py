from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.audio.extract_audio_from_mp4 import extract_audio_from_metadata
from src.config import (
    COMBINED_METADATA_CSV,
    DEFAULT_METADATA_CSV,
    INDICES_DIR,
    INCREMENTAL_RUN_SUMMARY_JSON,
    PROCESSED_REGISTRY_CSV,
    REALDATA_METADATA_CSV,
    YOUTUBE_MP4_INPUT_DIR,
)
from src.data.build_realdata_metadata import build_realdata_metadata
from src.data.merge_metadata import merge_metadata_files
from src.data.metadata_schema import empty_metadata_frame, load_metadata_frame
from src.embedding.build_indices import artifact_stem, build_all_indices
from src.embedding.vector_models import list_available_models
from src.evaluation.evaluate import compare_summary_frames, evaluate_all, evaluation_artifact_path
from src.evaluation.metrics_report import (
    DEFAULT_HALLUCINATION_THRESHOLD,
    build_incremental_probe_queryset,
    format_model_weight_lines,
)
from src.ingest.incremental_registry import (
    build_run_summary_payload,
    finalize_registry,
    plan_incremental_update,
    write_incremental_run_summary,
)
from src.search.keyword_search import KeywordSearchEngine
from src.search.load_realdata_dataset import dataset_artifact_namespace
from src.search.text_source import text_source_suffix
from src.stt.transcribe_mp4_batch import transcribe_mp4_batch
from src.utils.io_utils import save_dataframe
from src.visualize.clustering import cluster_embeddings
from src.visualize.pca_plot import build_projection_artifacts


def _normalized_paths(paths: list[Path]) -> set[str]:
    return {str(path.resolve()).casefold() for path in paths}


def _resolve_target_ids(metadata: pd.DataFrame, target_paths: list[Path]) -> set[str]:
    if metadata.empty or not target_paths:
        return set()
    normalized_paths = _normalized_paths(target_paths)
    filtered = metadata.loc[
        metadata["file_path"].fillna("").astype(str).str.casefold().isin(normalized_paths)
    ]
    return set(filtered["id"].fillna("").astype(str))


def _print_incremental_summary(plan: dict) -> None:
    summary = plan["summary"]
    print(f"[INFO] 전체 유튜브 파일 {summary['total_files']}개 확인")
    print(
        f"[INFO] 신규 파일 {summary['new_files']}개 / "
        f"변경 파일 {summary['changed_files']}개 / "
        f"skip {summary['skipped_files']}개"
    )
    if summary["missing_files"]:
        print(f"[INFO] 현재 입력 경로에서 사라진 기존 파일 {summary['missing_files']}개 감지")

    target_names = [record.file_name for record in plan["target_records"]]
    if target_names:
        print(f"[INFO] 이번 실행 처리 대상: {', '.join(target_names)}")
    else:
        print("[INFO] 신규/변경 파일이 없어 원본 처리 단계는 모두 skip 합니다.")


def _index_artifacts_exist(
    artifact_namespace: str,
    text_sources: tuple[str, ...],
    include_optional_models: bool,
) -> bool:
    model_aliases = list(list_available_models(include_optional=include_optional_models).keys())
    for text_source in text_sources:
        keyword_index = INDICES_DIR / f"{artifact_namespace}__keyword_index_metadata__{text_source_suffix(text_source)}.json"
        if not keyword_index.exists():
            return False
        for model_alias in model_aliases:
            dense_summary = INDICES_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_index_summary.json"
            if not dense_summary.exists():
                return False
    return True


def _evaluation_artifacts_exist(artifact_namespace: str | None) -> bool:
    required = [
        evaluation_artifact_path("retrieval_eval_summary.csv", artifact_namespace),
        evaluation_artifact_path("retrieval_eval_detail.csv", artifact_namespace),
        evaluation_artifact_path("retrieval_eval_source_comparison.csv", artifact_namespace),
    ]
    return all(path.exists() for path in required)


def _print_before_after_delta(delta: pd.DataFrame, top_k: int) -> None:
    if delta.empty:
        print("[EVAL] before/after 비교 대상이 없습니다.")
        return
    for row in delta.itertuples(index=False):
        print(
            f"[EVAL][DELTA] {row.system_name} / {row.text_source} | "
            f"Recall@{top_k}: {float(row.recall_at_k_delta):+.4f}, "
            f"Precision@{top_k}: {float(row.precision_at_k_delta):+.4f}, "
            f"Accuracy@1: {float(row.accuracy_at_1_delta):+.4f}, "
            f"F1@{top_k}: {float(row.f1_at_k_delta):+.4f}"
        )


def run_real_mp4_pipeline(
    input_dir: Path = YOUTUBE_MP4_INPUT_DIR,
    real_metadata_path: Path = REALDATA_METADATA_CSV,
    combined_metadata_path: Path = COMBINED_METADATA_CSV,
    whisper_model: str = "base",
    language: str | None = "ko",
    recursive: bool = True,
    limit: int | None = None,
    sample_rate: int = 16000,
    overwrite_audio: bool = False,
    overwrite_stt: bool = False,
    merge_with_dummy: bool = False,
    build_indices_for_search: bool = True,
    include_optional_models: bool = False,
    n_clusters: int = 6,
    text_sources: tuple[str, ...] = ("stt_transcript",),
    optional_projection_methods: tuple[str, ...] = ("tsne",),
    run_evaluation: bool = True,
    incremental: bool = True,
    registry_path: Path = PROCESSED_REGISTRY_CSV,
    compute_hash: bool = False,
    preserve_missing: bool = True,
    top_k_eval: int = 3,
    hallucination_threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
    compare_before_after: bool = True,
    show_weights: bool = True,
    show_metrics: bool = True,
) -> Path:
    before_summary = pd.DataFrame()
    before_metadata_path = combined_metadata_path if merge_with_dummy else real_metadata_path
    if compare_before_after and run_evaluation and before_metadata_path.exists():
        try:
            before_outputs = evaluate_all(
                metadata_path=before_metadata_path,
                text_sources=text_sources,
                include_optional=include_optional_models,
                artifact_namespace=f"{dataset_artifact_namespace(before_metadata_path)}__before_snapshot",
                top_k=top_k_eval,
                hallucination_threshold=hallucination_threshold,
                print_report=False,
                show_weights=False,
            )
            before_summary = before_outputs["summary"]
        except Exception as exc:
            print(f"[EVAL] before snapshot 생성을 건너뜁니다: {exc}")

    print("[1/8] 입력 파일 스캔 및 증분 계획 수립")
    plan = plan_incremental_update(
        input_dir=input_dir,
        registry_path=registry_path,
        recursive=recursive,
        limit=limit,
        compute_hash=compute_hash,
    )
    _print_incremental_summary(plan)

    all_records = plan["discovered_records"]
    target_records = plan["target_records"] if incremental else all_records
    target_paths = [Path(record.file_path) for record in target_records]

    should_refresh_full_metadata = not real_metadata_path.exists() and bool(all_records)
    should_refresh_metadata = bool(target_paths) or should_refresh_full_metadata

    if should_refresh_metadata:
        print("[2/8] metadata 증분 갱신")
        metadata_files = (
            target_paths
            if incremental and target_paths and not should_refresh_full_metadata
            else [Path(record.file_path) for record in all_records]
        )
        real_frame = build_realdata_metadata(
            input_dir=input_dir,
            metadata_path=real_metadata_path,
            recursive=recursive,
            limit=limit,
            media_files=metadata_files,
            preserve_missing=preserve_missing,
            compute_hash=compute_hash,
        )
        print("[INFO] metadata 갱신 완료")
    elif real_metadata_path.exists():
        print("[2/8] metadata 갱신 skip - 변경 파일 없음")
        real_frame = load_metadata_frame(real_metadata_path)
    else:
        print("[2/8] metadata 갱신 skip - 입력 파일 없음")
        real_frame = empty_metadata_frame()

    target_ids = _resolve_target_ids(real_frame, target_paths)

    if target_ids:
        print("[3/8] 오디오 추출 또는 wav 재사용")
        extract_audio_from_metadata(
            metadata_path=real_metadata_path,
            source_type=None,
            sample_rate=sample_rate,
            overwrite=overwrite_audio,
            skip_errors=True,
            target_ids=target_ids,
        )
        print("[INFO] 오디오 준비 완료")
    else:
        print("[3/8] 오디오 단계 skip - 신규/변경 파일 없음")

    if target_ids:
        print("[4/8] Whisper STT 증분 처리")
        transcribe_mp4_batch(
            metadata_path=real_metadata_path,
            model_name=whisper_model,
            language=language,
            overwrite=overwrite_stt,
            skip_errors=True,
            target_ids=target_ids,
        )
        print("[INFO] STT 완료")
    else:
        print("[4/8] STT 단계 skip - 신규/변경 파일 없음")

    if target_ids:
        print("[5/8] transcript 반영 및 metadata 동기화")
        real_frame = build_realdata_metadata(
            input_dir=input_dir,
            metadata_path=real_metadata_path,
            recursive=recursive,
            limit=limit,
            media_files=target_paths,
            preserve_missing=preserve_missing,
            compute_hash=compute_hash,
        )
        print("[INFO] transcript/metadata 반영 완료")
    else:
        print("[5/8] transcript/metadata 단계 skip - 신규/변경 파일 없음")

    if merge_with_dummy:
        merge_metadata_files(
            metadata_paths=[DEFAULT_METADATA_CSV, real_metadata_path],
            output_path=combined_metadata_path,
            skip_missing=True,
        )
        search_metadata_path = combined_metadata_path
        print("[6/8] dummy + youtube metadata 병합 완료")
    else:
        search_metadata_path = real_metadata_path
        print("[6/8] youtube metadata 단독 사용")

    artifact_namespace = dataset_artifact_namespace(search_metadata_path)
    search_metadata = load_metadata_frame(search_metadata_path) if search_metadata_path.exists() else empty_metadata_frame()

    index_artifacts_present = _index_artifacts_exist(artifact_namespace, text_sources, include_optional_models)
    should_rebuild_indices = build_indices_for_search and (bool(target_ids) or not index_artifacts_present)

    built_models: list[tuple[str, str]] = []
    indices_rebuilt = False
    if should_rebuild_indices:
        print("[7/8] 검색 인덱스 및 시각화 산출물 갱신")
        if target_ids:
            print(
                "[INFO] 원본 변환과 STT는 신규/변경 파일만 처리합니다. "
                "dense index는 append를 우선 시도하고, projection/clustering은 전역 좌표 일관성을 위해 안전하게 재계산합니다."
            )
        for text_source in text_sources:
            KeywordSearchEngine(search_metadata, text_source=text_source).export_index_metadata(
                artifact_namespace=artifact_namespace
            )
        built_models = build_all_indices(
            metadata_path=search_metadata_path,
            include_optional=include_optional_models,
            text_sources=text_sources,
            artifact_namespace=artifact_namespace,
            incremental=incremental,
        )
        indices_rebuilt = True
        for model_alias, text_source in built_models:
            print(f"[INFO] dense index 갱신: {model_alias} / {text_source}")
            build_projection_artifacts(
                search_metadata,
                model_alias,
                text_source=text_source,
                optional_methods=optional_projection_methods,
                artifact_namespace=artifact_namespace,
            )
            cluster_embeddings(
                search_metadata,
                model_alias,
                method="kmeans",
                n_clusters=n_clusters,
                text_source=text_source,
                artifact_namespace=artifact_namespace,
            )
            if include_optional_models:
                try:
                    cluster_embeddings(
                        search_metadata,
                        model_alias,
                        method="hdbscan",
                        n_clusters=n_clusters,
                        text_source=text_source,
                        artifact_namespace=artifact_namespace,
                    )
                except Exception as exc:
                    print(f"[INFO] HDBSCAN skip: {model_alias} / {text_source} / {exc}")
        print("[INFO] embedding/index 최소 갱신 완료")
    else:
        print("[7/8] 검색 인덱스 단계 skip - 변경 파일 없음")

    evaluation_ran = False
    after_outputs: dict[str, pd.DataFrame] | None = None
    if run_evaluation:
        evaluation_needed = bool(target_ids) or not _evaluation_artifacts_exist(artifact_namespace) or compare_before_after
        if evaluation_needed:
            print("[8/8] 평가 및 리포트 생성")
            after_outputs = evaluate_all(
                metadata_path=search_metadata_path,
                text_sources=text_sources,
                include_optional=include_optional_models,
                artifact_namespace=artifact_namespace,
                top_k=top_k_eval,
                hallucination_threshold=hallucination_threshold,
                print_report=show_metrics,
                show_weights=show_weights,
            )
            evaluation_ran = True
            if target_ids:
                probe_queryset = build_incremental_probe_queryset(search_metadata, target_ids)
                if not probe_queryset.empty:
                    print("[EVAL] 신규/변경 파일 Recall 우선 probe 평가")
                    evaluate_all(
                        metadata=search_metadata,
                        queryset=probe_queryset,
                        text_sources=text_sources,
                        include_optional=include_optional_models,
                        artifact_namespace=f"{artifact_namespace}__incremental_probe",
                        top_k=top_k_eval,
                        hallucination_threshold=hallucination_threshold,
                        print_report=True,
                        show_weights=False,
                    )
            if compare_before_after and not before_summary.empty:
                delta = compare_summary_frames(before_summary, after_outputs["summary"])
                delta_path = evaluation_artifact_path("retrieval_eval_delta.csv", artifact_namespace)
                save_dataframe(delta_path, delta)
                _print_before_after_delta(delta, top_k=top_k_eval)
        else:
            print("[8/8] 평가 skip - 변경 파일 없고 기존 결과 재사용")
            if show_weights:
                for line in format_model_weight_lines(include_optional=include_optional_models):
                    print(line)
    else:
        print("[8/8] 평가 skip")
        if show_weights:
            for line in format_model_weight_lines(include_optional=include_optional_models):
                print(line)

    final_real_frame = load_metadata_frame(real_metadata_path) if real_metadata_path.exists() else empty_metadata_frame()
    finalize_registry(
        plan=plan,
        metadata=final_real_frame,
        registry_path=registry_path,
        embedding_built=indices_rebuilt if target_ids else None,
    )
    processed_files = [record.file_name for record in target_records]
    run_summary = build_run_summary_payload(
        plan=plan,
        processed_files=processed_files,
        indices_rebuilt=indices_rebuilt,
        evaluation_ran=evaluation_ran,
        artifact_namespace=artifact_namespace,
        registry_path=registry_path,
    )
    write_incremental_run_summary(run_summary, INCREMENTAL_RUN_SUMMARY_JSON)

    print(f"[INFO] Search metadata ready: {search_metadata_path}")
    print(f"[INFO] Artifact namespace: {artifact_namespace}")
    print(f"[INFO] Registry path: {registry_path}")
    return search_metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run youtube mp4/wav -> STT -> metadata -> search pipeline.")
    parser.add_argument("--input-dir", type=Path, default=YOUTUBE_MP4_INPUT_DIR)
    parser.add_argument("--real-metadata-path", type=Path, default=REALDATA_METADATA_CSV)
    parser.add_argument("--combined-metadata-path", type=Path, default=COMBINED_METADATA_CSV)
    parser.add_argument("--registry-path", type=Path, default=PROCESSED_REGISTRY_CSV)
    parser.add_argument("--whisper-model", type=str, default="base")
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--overwrite-stt", action="store_true")
    parser.add_argument("--compute-hash", action="store_true")
    parser.add_argument("--preserve-missing", dest="preserve_missing", action="store_true")
    parser.add_argument("--remove-missing", dest="preserve_missing", action="store_false")
    parser.set_defaults(preserve_missing=True)
    parser.add_argument("--incremental", dest="incremental", action="store_true")
    parser.add_argument("--no-incremental", dest="incremental", action="store_false")
    parser.set_defaults(incremental=True)
    parser.add_argument("--merge-with-dummy", dest="merge_with_dummy", action="store_true")
    parser.add_argument("--real-only", dest="merge_with_dummy", action="store_false")
    parser.set_defaults(merge_with_dummy=False)
    parser.add_argument("--build-indices", dest="build_indices_for_search", action="store_true")
    parser.add_argument("--skip-indices", dest="build_indices_for_search", action="store_false")
    parser.set_defaults(build_indices_for_search=True)
    parser.add_argument("--include-optional-models", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--top-k-eval", type=int, default=3)
    parser.add_argument("--hallucination-threshold", type=float, default=DEFAULT_HALLUCINATION_THRESHOLD)
    parser.add_argument("--recursive", dest="recursive", action="store_true")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)
    parser.add_argument("--run-evaluation", dest="run_evaluation", action="store_true")
    parser.add_argument("--skip-evaluation", dest="run_evaluation", action="store_false")
    parser.set_defaults(run_evaluation=True)
    parser.add_argument("--compare-before-after", dest="compare_before_after", action="store_true")
    parser.add_argument("--no-compare-before-after", dest="compare_before_after", action="store_false")
    parser.set_defaults(compare_before_after=True)
    parser.add_argument("--show-weights", dest="show_weights", action="store_true")
    parser.add_argument("--hide-weights", dest="show_weights", action="store_false")
    parser.set_defaults(show_weights=True)
    parser.add_argument("--show-metrics", dest="show_metrics", action="store_true")
    parser.add_argument("--hide-metrics", dest="show_metrics", action="store_false")
    parser.set_defaults(show_metrics=True)
    parser.add_argument(
        "--text-sources",
        nargs="+",
        default=["stt_transcript"],
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    parser.add_argument(
        "--optional-projection-methods",
        nargs="*",
        default=["tsne"],
        choices=["tsne", "umap"],
    )
    args = parser.parse_args()

    run_real_mp4_pipeline(
        input_dir=args.input_dir,
        real_metadata_path=args.real_metadata_path,
        combined_metadata_path=args.combined_metadata_path,
        whisper_model=args.whisper_model,
        language=args.language,
        recursive=args.recursive,
        limit=args.limit,
        sample_rate=args.sample_rate,
        overwrite_audio=args.overwrite_audio,
        overwrite_stt=args.overwrite_stt,
        merge_with_dummy=args.merge_with_dummy,
        build_indices_for_search=args.build_indices_for_search,
        include_optional_models=args.include_optional_models,
        n_clusters=args.n_clusters,
        text_sources=tuple(args.text_sources),
        optional_projection_methods=tuple(args.optional_projection_methods),
        run_evaluation=args.run_evaluation,
        incremental=args.incremental,
        registry_path=args.registry_path,
        compute_hash=args.compute_hash,
        preserve_missing=args.preserve_missing,
        top_k_eval=args.top_k_eval,
        hallucination_threshold=args.hallucination_threshold,
        compare_before_after=args.compare_before_after,
        show_weights=args.show_weights,
        show_metrics=args.show_metrics,
    )


if __name__ == "__main__":
    main()
