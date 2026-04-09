from __future__ import annotations

import argparse
from pathlib import Path

from src.audio.extract_audio_from_mp4 import extract_audio_from_metadata
from src.config import COMBINED_METADATA_CSV, DEFAULT_METADATA_CSV, REALDATA_METADATA_CSV, YOUTUBE_MP4_INPUT_DIR
from src.data.build_realdata_metadata import build_realdata_metadata
from src.data.merge_metadata import merge_metadata_files
from src.data.metadata_schema import load_metadata_frame
from src.embedding.build_indices import build_all_indices
from src.evaluation.evaluate import evaluate_all
from src.search.keyword_search import KeywordSearchEngine
from src.search.load_realdata_dataset import dataset_artifact_namespace
from src.stt.transcribe_mp4_batch import transcribe_mp4_batch
from src.visualize.clustering import cluster_embeddings
from src.visualize.pca_plot import build_projection_artifacts


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
    merge_with_dummy: bool = True,
    build_indices_for_search: bool = True,
    include_optional_models: bool = False,
    n_clusters: int = 6,
    text_sources: tuple[str, ...] = ("stt_transcript",),
    optional_projection_methods: tuple[str, ...] = ("tsne",),
    run_evaluation: bool = True,
) -> Path:
    print("[1/8] Discover mp4 files and build/update youtube metadata")
    real_frame = build_realdata_metadata(
        input_dir=input_dir,
        metadata_path=real_metadata_path,
        recursive=recursive,
        limit=limit,
    )
    print(f"  - discovered {len(real_frame)} mp4 files")

    print("[2/8] Extract audio from mp4")
    extract_audio_from_metadata(
        metadata_path=real_metadata_path,
        source_type="youtube_mp4",
        sample_rate=sample_rate,
        overwrite=overwrite_audio,
        skip_errors=True,
    )

    print("[3/8] Transcribe extracted audio with Whisper")
    transcribe_mp4_batch(
        metadata_path=real_metadata_path,
        model_name=whisper_model,
        language=language,
        overwrite=overwrite_stt,
        skip_errors=True,
    )

    print("[4/8] Refresh youtube metadata with transcript txt and inferred keywords")
    build_realdata_metadata(
        input_dir=input_dir,
        metadata_path=real_metadata_path,
        recursive=recursive,
        limit=limit,
    )

    if merge_with_dummy:
        print("[5/8] Merge dummy metadata and youtube metadata")
        merge_metadata_files(
            metadata_paths=[DEFAULT_METADATA_CSV, real_metadata_path],
            output_path=combined_metadata_path,
            skip_missing=True,
        )
        search_metadata_path = combined_metadata_path
    else:
        print("[5/8] Skip dummy merge and search only on youtube metadata")
        search_metadata_path = real_metadata_path

    artifact_namespace = dataset_artifact_namespace(search_metadata_path)
    metadata = load_metadata_frame(search_metadata_path)

    if build_indices_for_search:
        print("[6/8] Build keyword and dense retrieval artifacts")
        for text_source in text_sources:
            KeywordSearchEngine(metadata, text_source=text_source).export_index_metadata(
                artifact_namespace=artifact_namespace
            )
        built_models = build_all_indices(
            metadata_path=search_metadata_path,
            include_optional=include_optional_models,
            text_sources=text_sources,
            artifact_namespace=artifact_namespace,
        )
        for model_alias, text_source in built_models:
            print(f"  - dense index: {model_alias} / {text_source} / namespace={artifact_namespace}")

        print("[7/8] Build projection and clustering artifacts")
        for model_alias, text_source in built_models:
            build_projection_artifacts(
                metadata,
                model_alias,
                text_source=text_source,
                optional_methods=optional_projection_methods,
                artifact_namespace=artifact_namespace,
            )
            cluster_embeddings(
                metadata,
                model_alias,
                method="kmeans",
                n_clusters=n_clusters,
                text_source=text_source,
                artifact_namespace=artifact_namespace,
            )
            if include_optional_models:
                try:
                    cluster_embeddings(
                        metadata,
                        model_alias,
                        method="hdbscan",
                        n_clusters=n_clusters,
                        text_source=text_source,
                        artifact_namespace=artifact_namespace,
                    )
                except Exception as exc:
                    print(f"  - skip HDBSCAN for {model_alias} / {text_source}: {exc}")
    else:
        built_models = []
        print("[6/8] Skip index build")
        print("[7/8] Skip projection and clustering")

    print("[8/8] Evaluation")
    if run_evaluation and DEFAULT_METADATA_CSV.exists():
        try:
            evaluate_all(
                metadata_path=search_metadata_path,
                text_sources=text_sources,
                include_optional=include_optional_models,
                artifact_namespace=artifact_namespace,
            )
        except Exception as exc:
            print(f"  - evaluation skipped: {exc}")
    else:
        print("  - evaluation skipped")

    print(f"Search metadata ready: {search_metadata_path}")
    print(f"Artifact namespace: {artifact_namespace}")
    return search_metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real youtube mp4 -> STT -> metadata -> search pipeline.")
    parser.add_argument("--input-dir", type=Path, default=YOUTUBE_MP4_INPUT_DIR)
    parser.add_argument("--real-metadata-path", type=Path, default=REALDATA_METADATA_CSV)
    parser.add_argument("--combined-metadata-path", type=Path, default=COMBINED_METADATA_CSV)
    parser.add_argument("--whisper-model", type=str, default="base")
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--overwrite-stt", action="store_true")
    parser.add_argument("--merge-with-dummy", dest="merge_with_dummy", action="store_true")
    parser.add_argument("--real-only", dest="merge_with_dummy", action="store_false")
    parser.set_defaults(merge_with_dummy=True)
    parser.add_argument("--build-indices", dest="build_indices_for_search", action="store_true")
    parser.add_argument("--skip-indices", dest="build_indices_for_search", action="store_false")
    parser.set_defaults(build_indices_for_search=True)
    parser.add_argument("--include-optional-models", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--recursive", dest="recursive", action="store_true")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)
    parser.add_argument("--run-evaluation", dest="run_evaluation", action="store_true")
    parser.add_argument("--skip-evaluation", dest="run_evaluation", action="store_false")
    parser.set_defaults(run_evaluation=True)
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

    search_metadata_path = run_real_mp4_pipeline(
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
    )
    print("Pipeline completed.")
    print(f"Use this metadata file for search: {search_metadata_path}")
    print("Main app: streamlit run experiment_app.py")


if __name__ == "__main__":
    main()
