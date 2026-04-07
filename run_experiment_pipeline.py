from __future__ import annotations

import argparse

from src.config import DEFAULT_METADATA_CSV, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame
from src.data.generate_dataset import generate_dataset
from src.embedding.build_indices import build_all_indices
from src.evaluation.evaluate import evaluate_all
from src.search.keyword_search import KeywordSearchEngine
from src.visualize.clustering import cluster_embeddings
from src.visualize.pca_plot import build_projection_artifacts


def run_pipeline(
    total_items: int = 100,
    include_optional: bool = False,
    n_clusters: int = 6,
    text_sources: tuple[str, ...] = ("stt_transcript", "original_transcript"),
) -> None:
    ensure_project_dirs()

    print("[1/6] Data Generation")
    metadata = generate_dataset(total_items=total_items)

    print("[2/6] Keyword Indexing")
    for text_source in text_sources:
        keyword_engine = KeywordSearchEngine(metadata, text_source=text_source)
        keyword_engine.export_index_metadata()

    print("[3/6] Dense Embedding / Vector Indexing")
    built_models = build_all_indices(
        DEFAULT_METADATA_CSV,
        include_optional=include_optional,
        text_sources=text_sources,
    )

    print("[4/6] Dimensionality Reduction")
    metadata = load_metadata_frame(DEFAULT_METADATA_CSV)
    for model_alias, text_source in built_models:
        build_projection_artifacts(metadata, model_alias, text_source=text_source)

    print("[5/6] Clustering")
    for model_alias, text_source in built_models:
        cluster_embeddings(metadata, model_alias, method="kmeans", n_clusters=n_clusters, text_source=text_source)
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

    print("[6/6] Evaluation")
    evaluate_all(DEFAULT_METADATA_CSV, text_sources=text_sources, include_optional=include_optional)

    print("Pipeline completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline.")
    parser.add_argument("--total-items", type=int, default=100)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument(
        "--text-sources",
        nargs="+",
        default=["stt_transcript", "original_transcript"],
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    args = parser.parse_args()
    run_pipeline(
        total_items=args.total_items,
        include_optional=args.include_optional,
        n_clusters=args.n_clusters,
        text_sources=tuple(args.text_sources),
    )


if __name__ == "__main__":
    main()
