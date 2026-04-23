from __future__ import annotations

"""검색 대상 메타데이터셋을 선택하고 로드하는 도우미 모듈.

UI와 실험 코드가 같은 데이터셋 선택 규칙을 공유하도록 경로 해석, 기본 선택,
아티팩트 네임스페이스 계산을 여기서 담당한다.
"""

import re
from pathlib import Path

import pandas as pd

from src.config import COMBINED_METADATA_CSV, DEFAULT_METADATA_CSV, REALDATA_METADATA_CSV
from src.data.metadata_schema import empty_metadata_frame, load_metadata_frame


DATASET_PATHS = {
    "youtube_mp4": REALDATA_METADATA_CSV,
    "combined": COMBINED_METADATA_CSV,
    "dummy": DEFAULT_METADATA_CSV,
}


def available_dataset_options() -> list[tuple[str, Path]]:
    """실제로 디스크에 존재하는 데이터셋 선택지만 반환한다."""
    options: list[tuple[str, Path]] = []
    for key in ["youtube_mp4", "combined", "dummy"]:
        path = DATASET_PATHS[key]
        if path.exists():
            options.append((key, path))
    if not options:
        options.append(("youtube_mp4", REALDATA_METADATA_CSV))
    return options


def resolve_dataset_path(dataset_key_or_path: str | Path) -> Path:
    """미리 정의된 데이터셋 키와 직접 지정한 메타데이터 경로를 모두 허용한다."""
    if isinstance(dataset_key_or_path, Path):
        return dataset_key_or_path
    if dataset_key_or_path in DATASET_PATHS:
        return DATASET_PATHS[dataset_key_or_path]
    return Path(dataset_key_or_path)


def default_search_metadata_path() -> Path:
    """UI가 가능한 한 실제 운영에 가까운 데이터셋으로 시작하도록 기본 경로를 고른다."""
    if REALDATA_METADATA_CSV.exists():
        return REALDATA_METADATA_CSV
    if COMBINED_METADATA_CSV.exists():
        return COMBINED_METADATA_CSV
    return DEFAULT_METADATA_CSV


def dataset_artifact_namespace(metadata_path: Path, source_types: tuple[str, ...] | None = None) -> str:
    """데이터셋 식별자와 필터 조합으로 안정적인 캐시 네임스페이스를 만든다."""
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
    """메타데이터를 읽고, 해석된 실제 경로와 필터링된 프레임을 함께 반환한다."""
    metadata_path = resolve_dataset_path(dataset_key_or_path)
    frame = load_metadata_frame(metadata_path) if metadata_path.exists() else empty_metadata_frame()
    if source_types:
        frame = frame.loc[frame["source_type"].isin(source_types)].reset_index(drop=True)
    return frame, metadata_path
