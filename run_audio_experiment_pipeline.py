from __future__ import annotations

import argparse

from src.config import DEFAULT_METADATA_CSV, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame
from src.data.generate_dataset import generate_dataset
from src.audio.generate_tts_audio import generate_tts_audio_batch
from src.embedding.build_indices import build_all_indices
from src.evaluation.evaluate import evaluate_all
from src.search.keyword_search import KeywordSearchEngine
from src.stt.batch_transcribe import transcribe_audio_batch
from src.visualize.clustering import cluster_embeddings
from src.visualize.pca_plot import build_projection_artifacts


def run_audio_pipeline(
    total_items: int = 100,
    tts_provider: str | None = None,
    edge_voice: str | None = None,
    whisper_model: str | None = None,
    include_optional: bool = False,
    n_clusters: int | None = None,
    text_sources: tuple[str, ...] = ("stt_transcript", "original_transcript"),
    regenerate_dataset: bool = False,
    overwrite_audio: bool = False,
    overwrite_stt: bool = False,
) -> None:
    ensure_project_dirs()

    if regenerate_dataset or not DEFAULT_METADATA_CSV.exists():
        print("[1/8] Data generation")
        generate_dataset(total_items=total_items)
    else:
        print("[1/8] Data generation skipped - existing metadata reused")

    print("[2/8] TTS audio generation")
    generate_tts_audio_batch(
        metadata_path=DEFAULT_METADATA_CSV,
        provider=tts_provider,
        edge_voice=edge_voice,
        overwrite=overwrite_audio,
    )

    print("[3/8] Whisper STT")
    transcribe_audio_batch(
        metadata_path=DEFAULT_METADATA_CSV,
        model_name=whisper_model,
        overwrite=overwrite_stt,
    )

    metadata = load_metadata_frame(DEFAULT_METADATA_CSV)

    print("[4/8] Keyword indexing")
    for text_source in text_sources:
        KeywordSearchEngine(metadata, text_source=text_source).export_index_metadata()

    print("[5/8] Dense embedding / vector indexing")
    built_models = build_all_indices(
        DEFAULT_METADATA_CSV,
        include_optional=include_optional,
        text_sources=text_sources,
    )

    print("[6/8] Dimensionality reduction")
    for model_alias, text_source in built_models:
        build_projection_artifacts(metadata, model_alias, text_source=text_source)

    print("[7/8] Clustering")
    for model_alias, text_source in built_models:
        cluster_embeddings(
            metadata,
            model_alias,
            method="kmeans",
            n_clusters=n_clusters,
            text_source=text_source,
        )
        if include_optional:
            try:
                cluster_embeddings(
                    metadata,
                    model_alias,
                    method="hdbscan",
                    n_clusters=n_clusters,
                    text_source=text_source,
                )
            except Exception as exc:
                print(f"Skipping HDBSCAN for {model_alias} / {text_source}: {exc}")

    print("[8/8] Evaluation")
    evaluate_all(DEFAULT_METADATA_CSV, text_sources=text_sources, include_optional=include_optional)

    print("Audio-first experiment pipeline completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the audio -> STT -> retrieval experiment pipeline.")
    parser.add_argument("--total-items", type=int, default=100)
    parser.add_argument("--tts-provider", type=str, default=None)
    parser.add_argument("--edge-voice", type=str, default=None)
    parser.add_argument("--whisper-model", type=str, default=None)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument(
        "--text-sources",
        nargs="+",
        default=["stt_transcript", "original_transcript"],
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    parser.add_argument("--regenerate-dataset", action="store_true")
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--overwrite-stt", action="store_true")
    args = parser.parse_args()

    run_audio_pipeline(
        total_items=args.total_items,
        tts_provider=args.tts_provider,
        edge_voice=args.edge_voice,
        whisper_model=args.whisper_model,
        include_optional=args.include_optional,
        n_clusters=args.n_clusters,
        text_sources=tuple(args.text_sources),
        regenerate_dataset=args.regenerate_dataset,
        overwrite_audio=args.overwrite_audio,
        overwrite_stt=args.overwrite_stt,
    )


if __name__ == "__main__":
    main()
