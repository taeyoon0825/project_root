from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.adaptive.parameter_resolver import AdaptiveContext, build_adaptive_context, resolve_cluster_config
from src.config import CLUSTERS_DIR, DEFAULT_METADATA_CSV, EMBEDDINGS_DIR, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame
from src.embedding.build_indices import DenseSearchEngine, artifact_stem
from src.search.text_source import DEFAULT_TEXT_SOURCE, build_preview_text
from src.utils.io_utils import save_dataframe, save_json

try:
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    hdbscan = None


def load_embeddings(
    model_alias: str,
    text_source: str = DEFAULT_TEXT_SOURCE,
    artifact_namespace: str | None = None,
) -> np.ndarray:
    path = EMBEDDINGS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings: {path}")
    return np.load(path)


def cluster_embeddings(
    metadata: pd.DataFrame,
    model_alias: str,
    method: str = "kmeans",
    n_clusters: int | None = None,
    text_source: str = DEFAULT_TEXT_SOURCE,
    artifact_namespace: str | None = None,
    adaptive_context: AdaptiveContext | None = None,
) -> tuple[pd.DataFrame, dict]:
    ensure_project_dirs()
    dense_engine = DenseSearchEngine(
        metadata,
        model_alias,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
        adaptive_context=adaptive_context,
    )
    dense_engine.load()
    assert dense_engine.embeddings is not None
    metadata = dense_engine.metadata.copy()
    embeddings = dense_engine.embeddings

    if len(metadata) == 0:
        raise ValueError("Cannot cluster an empty dataset.")

    context = dense_engine.adaptive_context or build_adaptive_context(
        metadata,
        text_source=text_source,
        embedding_model_alias=model_alias,
        embeddings=embeddings,
        artifact_namespace=artifact_namespace,
    )
    cluster_config = resolve_cluster_config(context.profile, embeddings=embeddings)
    requested_clusters = n_clusters if n_clusters is not None else cluster_config.n_clusters

    effective_method = method
    if effective_method == "hdbscan" and hdbscan is None:
        effective_method = "kmeans"

    if effective_method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cluster_config.min_cluster_size,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
        effective_requested_clusters = requested_clusters
    else:
        effective_requested_clusters = max(1, min(int(requested_clusters), len(metadata)))
        if effective_requested_clusters == 1:
            labels = np.zeros(len(metadata), dtype=int)
        else:
            clusterer = KMeans(
                n_clusters=effective_requested_clusters,
                random_state=context.visualization.random_seed,
                n_init=20,
            )
            labels = clusterer.fit_predict(embeddings)

    frame = metadata.copy()
    frame["cluster_id"] = labels
    frame["text_source"] = text_source
    frame["preview"] = frame.apply(
        lambda row: build_preview_text(
            row,
            text_source=text_source,
            length=context.visualization.preview_length,
        ),
        axis=1,
    )

    valid_mask = labels >= 0
    if len(set(labels[valid_mask])) > 1 and valid_mask.sum() > 1:
        silhouette = float(silhouette_score(embeddings[valid_mask], labels[valid_mask]))
    else:
        silhouette = float("nan")

    representatives = (
        frame.groupby("cluster_id")
        .agg(
            size=("id", "count"),
            categories=("category", lambda items: ", ".join(sorted(set(map(str, items))))),
            sample_id=("id", "first"),
            sample_title=("title", "first"),
            sample_preview=("preview", "first"),
        )
        .reset_index()
    )

    stem = artifact_stem(model_alias, text_source, artifact_namespace)
    cluster_csv = CLUSTERS_DIR / f"{stem}_{effective_method}_clusters.csv"
    reps_csv = CLUSTERS_DIR / f"{stem}_{effective_method}_representatives.csv"
    summary_json = CLUSTERS_DIR / f"{stem}_{effective_method}_summary.json"
    save_dataframe(cluster_csv, frame)
    save_dataframe(reps_csv, representatives)
    save_json(
        summary_json,
        {
            "model_alias": model_alias,
            "text_source": text_source,
            "artifact_namespace": artifact_namespace,
            "method_requested": method,
            "method_effective": effective_method,
            "n_clusters_requested": requested_clusters,
            "n_clusters_effective": effective_requested_clusters,
            "n_clusters_found": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "silhouette_score": silhouette,
            "adaptive_cluster": cluster_config.to_dict(),
            "adaptive_profile": context.profile.to_dict(),
            "adaptive_visualization": context.visualization.to_dict(),
            "reasoning": cluster_config.reasoning,
        },
    )
    return frame, load_summary(summary_json)


def load_cluster_frame(
    model_alias: str,
    text_source: str = DEFAULT_TEXT_SOURCE,
    method: str = "kmeans",
    artifact_namespace: str | None = None,
) -> pd.DataFrame:
    return pd.read_csv(CLUSTERS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_{method}_clusters.csv")


def load_representatives(
    model_alias: str,
    text_source: str = DEFAULT_TEXT_SOURCE,
    method: str = "kmeans",
    artifact_namespace: str | None = None,
) -> pd.DataFrame:
    return pd.read_csv(
        CLUSTERS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_{method}_representatives.csv"
    )


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_cluster_summary(
    model_alias: str,
    text_source: str = DEFAULT_TEXT_SOURCE,
    method: str = "kmeans",
    artifact_namespace: str | None = None,
) -> dict:
    return load_summary(CLUSTERS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_{method}_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster embeddings and save outputs.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--model-alias", type=str, required=True)
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "hdbscan"])
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--artifact-namespace", type=str, default=None)
    parser.add_argument(
        "--text-source",
        type=str,
        default=DEFAULT_TEXT_SOURCE,
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    args = parser.parse_args()

    metadata = load_metadata_frame(args.metadata_path)
    _, summary = cluster_embeddings(
        metadata,
        args.model_alias,
        method=args.method,
        n_clusters=args.n_clusters,
        text_source=args.text_source,
        artifact_namespace=args.artifact_namespace,
    )
    print(summary)


if __name__ == "__main__":
    main()
