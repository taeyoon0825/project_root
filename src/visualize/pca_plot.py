from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import DEFAULT_METADATA_CSV, EMBEDDINGS_DIR, PLOTS_DIR, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame
from src.embedding.build_indices import DenseSearchEngine, artifact_stem
from src.search.text_source import DEFAULT_TEXT_SOURCE
from src.utils.io_utils import save_dataframe, save_json


UMAP_TIMEOUT_SECONDS = 45
OPTIONAL_PROJECTION_METHODS = ("umap", "tsne")


class ZeroProjectionReducer:
    def __init__(self, output_dimensions: int = 3):
        self.output_dimensions = output_dimensions
        self.components_ = np.zeros((0, 0), dtype=np.float32)
        self.explained_variance_ratio_ = np.zeros(output_dimensions, dtype=np.float32)

    def transform(self, vectors) -> np.ndarray:
        array = np.asarray(vectors)
        return np.zeros((len(array), self.output_dimensions), dtype=np.float32)


def _configure_projection_env() -> None:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _umap_projection_worker(embeddings: np.ndarray, dimensions: int, output_queue) -> None:
    _configure_projection_env()
    try:
        import umap as umap_module  # type: ignore

        reducer = umap_module.UMAP(
            n_components=dimensions,
            random_state=42,
            transform_seed=42,
            n_jobs=1,
            low_memory=True,
            init="random",
        )
        projected = reducer.fit_transform(embeddings)
        output_queue.put({"ok": True, "projected": projected.tolist()})
    except Exception as exc:  # pragma: no cover
        output_queue.put({"ok": False, "error": str(exc)})


def projection_columns(method: str, dimensions: int) -> list[str]:
    if method == "pca":
        return [f"PC{i}" for i in range(1, dimensions + 1)]
    prefix = method.upper()
    return [f"{prefix}{i}" for i in range(1, dimensions + 1)]


def projection_csv_path(
    model_alias: str,
    text_source: str,
    method: str,
    dimensions: int,
    artifact_namespace: str | None = None,
) -> Path:
    return PLOTS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_{method}_{dimensions}d_projection.csv"


def reducer_model_path(
    model_alias: str,
    text_source: str,
    method: str,
    dimensions: int,
    artifact_namespace: str | None = None,
) -> Path:
    return PLOTS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_{method}_{dimensions}d_model.joblib"


def load_embeddings(
    model_alias: str,
    text_source: str = DEFAULT_TEXT_SOURCE,
    artifact_namespace: str | None = None,
) -> np.ndarray:
    path = EMBEDDINGS_DIR / f"{artifact_stem(model_alias, text_source, artifact_namespace)}_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings: {path}")
    return np.load(path)


def _pad_projection(projected: np.ndarray, target_dimensions: int) -> np.ndarray:
    if projected.shape[1] >= target_dimensions:
        return projected[:, :target_dimensions]
    padding = np.zeros((projected.shape[0], target_dimensions - projected.shape[1]), dtype=projected.dtype)
    return np.hstack([projected, padding])


def compute_pca_projection(embeddings: np.ndarray) -> tuple[pd.DataFrame, object]:
    if embeddings.shape[0] < 2:
        projected = np.zeros((embeddings.shape[0], 3), dtype=np.float32)
        frame = pd.DataFrame(projected, columns=projection_columns("pca", 3))
        return frame, ZeroProjectionReducer(output_dimensions=3)

    n_components = max(1, min(3, embeddings.shape[0], embeddings.shape[1]))
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(embeddings)
    projected = _pad_projection(projected, 3)
    frame = pd.DataFrame(projected, columns=projection_columns("pca", 3))
    return frame, pca


def compute_optional_projection(
    embeddings: np.ndarray,
    method: str,
    dimensions: int,
) -> tuple[pd.DataFrame, object | None]:
    if embeddings.shape[0] <= dimensions:
        raise ValueError(f"{method.upper()} requires more than {dimensions} samples.")

    if method == "tsne":
        perplexity = min(30, max(2, embeddings.shape[0] - 1))
        reducer = TSNE(
            n_components=dimensions,
            random_state=42,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        projected = reducer.fit_transform(embeddings)
        return pd.DataFrame(projected, columns=projection_columns(method, dimensions)), None

    if method == "umap":
        try:
            __import__("umap")
        except ImportError as exc:
            raise ImportError("umap-learn is not installed.") from exc
        except Exception as exc:
            raise RuntimeError(f"Unable to initialize umap: {exc}") from exc
        context = mp.get_context("spawn")
        output_queue = context.Queue()
        process = context.Process(target=_umap_projection_worker, args=(embeddings, dimensions, output_queue))
        process.start()
        process.join(timeout=UMAP_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError(f"UMAP projection timed out after {UMAP_TIMEOUT_SECONDS} seconds")
        if output_queue.empty():
            raise RuntimeError("UMAP projection did not return a result")
        result = output_queue.get()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "UMAP projection failed"))
        projected = np.asarray(result["projected"], dtype=np.float32)
        return pd.DataFrame(projected, columns=projection_columns(method, dimensions)), None

    raise ValueError(f"Unsupported projection method: {method}")


def build_projection_artifacts(
    metadata: pd.DataFrame,
    model_alias: str,
    text_source: str = DEFAULT_TEXT_SOURCE,
    optional_methods: list[str] | tuple[str, ...] = ("tsne",),
    artifact_namespace: str | None = None,
) -> dict[str, Path]:
    _configure_projection_env()
    ensure_project_dirs()
    dense_engine = DenseSearchEngine(
        metadata,
        model_alias,
        text_source=text_source,
        artifact_namespace=artifact_namespace,
    )
    dense_engine.load()
    assert dense_engine.embeddings is not None
    metadata = dense_engine.metadata.copy()
    embeddings = dense_engine.embeddings
    stem = artifact_stem(model_alias, text_source, artifact_namespace)
    enabled_methods = set(optional_methods)

    outputs: dict[str, Path] = {}

    pca_frame, pca_model = compute_pca_projection(embeddings)
    pca_combined = pd.concat([metadata.reset_index(drop=True), pca_frame], axis=1)
    pca_combined["text_source"] = text_source
    pca_csv = projection_csv_path(model_alias, text_source, "pca", 3, artifact_namespace)
    save_dataframe(pca_csv, pca_combined)
    outputs["pca_3d_csv"] = pca_csv

    pca_model_path = reducer_model_path(model_alias, text_source, "pca", 3, artifact_namespace)
    joblib.dump(pca_model, pca_model_path)
    outputs["pca_model"] = pca_model_path

    explained = np.asarray(getattr(pca_model, "explained_variance_ratio_", []), dtype=np.float32)
    explained = np.pad(explained, (0, max(0, 3 - len(explained))), constant_values=0.0)
    variance_payload = {
        "model_alias": model_alias,
        "text_source": text_source,
        "artifact_namespace": artifact_namespace,
        "explained_variance_ratio": explained.tolist(),
        "components_shape": list(getattr(pca_model, "components_", np.empty((0, 0))).shape),
        "pc_contribution_summary": {
            "PC1": float(explained[0]),
            "PC2": float(explained[1]),
            "PC3": float(explained[2]),
        },
        "cumulative_first3": float(explained[:3].sum()),
    }
    variance_json_path = PLOTS_DIR / f"{stem}_pca_variance.json"
    save_json(variance_json_path, variance_payload)
    outputs["pca_variance_json"] = variance_json_path

    for method in OPTIONAL_PROJECTION_METHODS:
        for dimensions in [2, 3]:
            csv_path = projection_csv_path(model_alias, text_source, method, dimensions, artifact_namespace)
            model_path = reducer_model_path(model_alias, text_source, method, dimensions, artifact_namespace)
            if method not in enabled_methods:
                csv_path.unlink(missing_ok=True)
                model_path.unlink(missing_ok=True)
                continue
            try:
                projection_frame, reducer = compute_optional_projection(embeddings, method=method, dimensions=dimensions)
                combined = pd.concat([metadata.reset_index(drop=True), projection_frame], axis=1)
                combined["text_source"] = text_source
                save_dataframe(csv_path, combined)
                outputs[f"{method}_{dimensions}d_csv"] = csv_path

                if reducer is not None:
                    joblib.dump(reducer, model_path)
                    outputs[f"{method}_{dimensions}d_model"] = model_path
            except Exception:
                csv_path.unlink(missing_ok=True)
                model_path.unlink(missing_ok=True)
                continue

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PCA/UMAP/t-SNE projection artifacts.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--model-alias", type=str, required=True)
    parser.add_argument("--artifact-namespace", type=str, default=None)
    parser.add_argument(
        "--text-source",
        type=str,
        default=DEFAULT_TEXT_SOURCE,
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    parser.add_argument(
        "--optional-methods",
        nargs="*",
        default=["tsne"],
        choices=["tsne", "umap"],
        help="Optional projections to build in addition to PCA. Defaults to tsne only.",
    )
    args = parser.parse_args()

    metadata = load_metadata_frame(args.metadata_path)
    output = build_projection_artifacts(
        metadata,
        args.model_alias,
        text_source=args.text_source,
        optional_methods=args.optional_methods,
        artifact_namespace=args.artifact_namespace,
    )
    for name, path in output.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
