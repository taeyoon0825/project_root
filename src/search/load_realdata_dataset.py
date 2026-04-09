from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import COMBINED_METADATA_CSV, DEFAULT_METADATA_CSV, REALDATA_METADATA_CSV
from src.data.metadata_schema import load_metadata_frame


DATASET_PATHS = {
    "dummy": DEFAULT_METADATA_CSV,
    "youtube_mp4": REALDATA_METADATA_CSV,
    "combined": COMBINED_METADATA_CSV,
}


def available_dataset_options() -> list[tuple[str, Path]]:
    options: list[tuple[str, Path]] = []
    for key, path in DATASET_PATHS.items():
        if path.exists():
            options.append((key, path))
    if not options:
        options.append(("dummy", DEFAULT_METADATA_CSV))
    return options


def resolve_dataset_path(dataset_key_or_path: str | Path) -> Path:
    if isinstance(dataset_key_or_path, Path):
        return dataset_key_or_path
    if dataset_key_or_path in DATASET_PATHS:
        return DATASET_PATHS[dataset_key_or_path]
    return Path(dataset_key_or_path)


def default_search_metadata_path() -> Path:
    if COMBINED_METADATA_CSV.exists():
        return COMBINED_METADATA_CSV
    if REALDATA_METADATA_CSV.exists():
        return REALDATA_METADATA_CSV
    return DEFAULT_METADATA_CSV


def dataset_artifact_namespace(metadata_path: Path, source_types: tuple[str, ...] | None = None) -> str:
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", metadata_path.stem).strip("._") or "dataset"
    if not source_types:
        return stem
    source_token = "_".join(sorted(re.sub(r"[^0-9A-Za-z._-]+", "_", item) for item in source_types))
    source_token = source_token.strip("._")
    if not source_token:
        return stem
    return f"{stem}__{source_token}"


def load_search_metadata(
    dataset_key_or_path: str | Path,
    source_types: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, Path]:
    metadata_path = resolve_dataset_path(dataset_key_or_path)
    frame = load_metadata_frame(metadata_path)
    if source_types:
        frame = frame.loc[frame["source_type"].isin(source_types)].reset_index(drop=True)
    return frame, metadata_path
