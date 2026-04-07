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
from src.embedding.build_indices import artifact_stem
from src.search.text_source import DEFAULT_TEXT_SOURCE
from src.utils.io_utils import save_dataframe, save_json


UMAP_TIMEOUT_SECONDS = 45
OPTIONAL_PROJECTION_METHODS = ("umap", "tsne")


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


def projection_csv_path(model_alias: str, text_source: str, method: str, dimensions: int) -> Path:
    return PLOTS_DIR / f"{artifact_stem(model_alias, text_source)}_{method}_{dimensions}d_projection.csv"


def reducer_model_path(model_alias: str, text_source: str, method: str, dimensions: int) -> Path:
    return PLOTS_DIR / f"{artifact_stem(model_alias, text_source)}_{method}_{dimensions}d_model.joblib"


def load_embeddings(model_alias: str, text_source: str = DEFAULT_TEXT_SOURCE) -> np.ndarray:
    path = EMBEDDINGS_DIR / f"{artifact_stem(model_alias, text_source)}_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings: {path}")
    return np.load(path)


def compute_pca_projection(embeddings: np.ndarray) -> tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=3, random_state=42)
    projected = pca.fit_transform(embeddings)
    frame = pd.DataFrame(projected, columns=projection_columns("pca", 3))
    return frame, pca


def compute_optional_projection(
    embeddings: np.ndarray,
    method: str,
    dimensions: int,
) -> tuple[pd.DataFrame, object | None]:
    if method == "tsne":
        reducer = TSNE(n_components=dimensions, random_state=42, init="pca", learning_rate="auto")
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
) -> dict[str, Path]:
    _configure_projection_env()
    ensure_project_dirs()
    embeddings = load_embeddings(model_alias, text_source=text_source)
    stem = artifact_stem(model_alias, text_source)
    enabled_methods = set(optional_methods)

    outputs: dict[str, Path] = {}

    pca_frame, pca_model = compute_pca_projection(embeddings)
    pca_combined = pd.concat([metadata.reset_index(drop=True), pca_frame], axis=1)
    pca_combined["text_source"] = text_source
    pca_csv_path = projection_csv_path(model_alias, text_source, "pca", 3)
    save_dataframe(pca_csv_path, pca_combined)
    outputs["pca_3d_csv"] = pca_csv_path

    joblib.dump(pca_model, reducer_model_path(model_alias, text_source, "pca", 3))
    outputs["pca_model"] = reducer_model_path(model_alias, text_source, "pca", 3)

    variance_payload = {
        "model_alias": model_alias,
        "text_source": text_source,
        "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "components_shape": list(pca_model.components_.shape),
        "pc_contribution_summary": {
            "PC1": float(pca_model.explained_variance_ratio_[0]),
            "PC2": float(pca_model.explained_variance_ratio_[1]),
            "PC3": float(pca_model.explained_variance_ratio_[2]),
        },
        "cumulative_first3": float(sum(pca_model.explained_variance_ratio_[:3])),
    }
    variance_json_path = PLOTS_DIR / f"{stem}_pca_variance.json"
    save_json(variance_json_path, variance_payload)
    outputs["pca_variance_json"] = variance_json_path

    for method in OPTIONAL_PROJECTION_METHODS:
        for dimensions in [2, 3]:
            if method not in enabled_methods:
                projection_csv_path(model_alias, text_source, method, dimensions).unlink(missing_ok=True)
                reducer_model_path(model_alias, text_source, method, dimensions).unlink(missing_ok=True)
                continue
            try:
                projection_frame, reducer = compute_optional_projection(embeddings, method=method, dimensions=dimensions)
                combined = pd.concat([metadata.reset_index(drop=True), projection_frame], axis=1)
                combined["text_source"] = text_source
                csv_path = projection_csv_path(model_alias, text_source, method, dimensions)
                save_dataframe(csv_path, combined)
                outputs[f"{method}_{dimensions}d_csv"] = csv_path

                if reducer is not None:
                    model_path = reducer_model_path(model_alias, text_source, method, dimensions)
                    joblib.dump(reducer, model_path)
                    outputs[f"{method}_{dimensions}d_model"] = model_path
            except Exception:
                projection_csv_path(model_alias, text_source, method, dimensions).unlink(missing_ok=True)
                reducer_model_path(model_alias, text_source, method, dimensions).unlink(missing_ok=True)
                continue

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PCA/UMAP/t-SNE projection artifacts.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--model-alias", type=str, required=True)
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
    )
    for name, path in output.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
