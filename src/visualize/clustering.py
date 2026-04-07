from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import CLUSTERS_DIR, DEFAULT_METADATA_CSV, EMBEDDINGS_DIR, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame
from src.embedding.build_indices import artifact_stem
from src.search.text_source import DEFAULT_TEXT_SOURCE, build_preview_text
from src.utils.io_utils import save_dataframe, save_json

try:
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    hdbscan = None


def load_embeddings(model_alias: str, text_source: str = DEFAULT_TEXT_SOURCE) -> np.ndarray:
    path = EMBEDDINGS_DIR / f"{artifact_stem(model_alias, text_source)}_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings: {path}")
    return np.load(path)


def cluster_embeddings(
    metadata: pd.DataFrame,
    model_alias: str,
    method: str = "kmeans",
    n_clusters: int = 6,
    text_source: str = DEFAULT_TEXT_SOURCE,
) -> tuple[pd.DataFrame, dict]:
    ensure_project_dirs()
    embeddings = load_embeddings(model_alias, text_source=text_source)

    if method == "hdbscan":
        if hdbscan is None:
            raise ImportError("hdbscan is not installed.")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)
    else:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = clusterer.fit_predict(embeddings)

    frame = metadata.copy()
    frame["cluster_id"] = labels
    frame["text_source"] = text_source
    frame["preview"] = frame.apply(lambda row: build_preview_text(row, text_source=text_source), axis=1)

    valid_mask = labels >= 0
    if len(set(labels[valid_mask])) > 1 and valid_mask.sum() > 1:
        silhouette = float(silhouette_score(embeddings[valid_mask], labels[valid_mask]))
    else:
        silhouette = float("nan")

    representatives = (
        frame.groupby("cluster_id")
        .agg(
            size=("id", "count"),
            categories=("category", lambda items: ", ".join(sorted(set(items)))),
            sample_id=("id", "first"),
            sample_title=("title", "first"),
            sample_preview=("preview", "first"),
        )
        .reset_index()
    )

    stem = artifact_stem(model_alias, text_source)
    cluster_csv = CLUSTERS_DIR / f"{stem}_{method}_clusters.csv"
    reps_csv = CLUSTERS_DIR / f"{stem}_{method}_representatives.csv"
    summary_json = CLUSTERS_DIR / f"{stem}_{method}_summary.json"
    save_dataframe(cluster_csv, frame)
    save_dataframe(reps_csv, representatives)
    save_json(
        summary_json,
        {
            "model_alias": model_alias,
            "text_source": text_source,
            "method": method,
            "n_clusters_requested": n_clusters,
            "n_clusters_found": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "silhouette_score": silhouette,
        },
    )
    return frame, load_summary(summary_json)


def load_cluster_frame(model_alias: str, text_source: str = DEFAULT_TEXT_SOURCE, method: str = "kmeans") -> pd.DataFrame:
    return pd.read_csv(CLUSTERS_DIR / f"{artifact_stem(model_alias, text_source)}_{method}_clusters.csv")


def load_representatives(model_alias: str, text_source: str = DEFAULT_TEXT_SOURCE, method: str = "kmeans") -> pd.DataFrame:
    return pd.read_csv(CLUSTERS_DIR / f"{artifact_stem(model_alias, text_source)}_{method}_representatives.csv")


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_cluster_summary(model_alias: str, text_source: str = DEFAULT_TEXT_SOURCE, method: str = "kmeans") -> dict:
    return load_summary(CLUSTERS_DIR / f"{artifact_stem(model_alias, text_source)}_{method}_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster embeddings and save outputs.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--model-alias", type=str, required=True)
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "hdbscan"])
    parser.add_argument("--n-clusters", type=int, default=6)
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
    )
    print(summary)


if __name__ == "__main__":
    main()
