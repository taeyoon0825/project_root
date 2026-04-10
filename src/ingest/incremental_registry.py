from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import INCREMENTAL_RUN_SUMMARY_JSON, PROCESSED_REGISTRY_CSV, ensure_project_dirs
from src.data.metadata_schema import ensure_metadata_columns
from src.utils.io_utils import load_json, save_dataframe, save_json


SUPPORTED_MEDIA_EXTENSIONS = (".mp4", ".wav")
BOOLEAN_REGISTRY_COLUMNS = (
    "audio_extracted",
    "stt_done",
    "metadata_written",
    "embedding_built",
)
REGISTRY_DEFAULTS: dict[str, Any] = {
    "source_type": "",
    "file_name": "",
    "file_path": "",
    "relative_path": "",
    "media_extension": "",
    "mtime": 0.0,
    "size": 0,
    "file_hash": "",
    "metadata_id": "",
    "category": "",
    "audio_path": "",
    "stt_txt_path": "",
    "audio_extracted": False,
    "stt_done": False,
    "metadata_written": False,
    "embedding_built": False,
    "current_status": "",
    "last_seen_at": "",
    "last_processed_at": "",
    "error_message": "",
}


@dataclass(frozen=True)
class ScanRecord:
    source_type: str
    file_name: str
    file_path: str
    relative_path: str
    media_extension: str
    mtime: float
    size: int
    file_hash: str = ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _normalize_path(value: str | Path) -> str:
    return str(Path(value).resolve()).casefold()


def _record_to_dict(record: ScanRecord) -> dict[str, Any]:
    return {
        "source_type": record.source_type,
        "file_name": record.file_name,
        "file_path": record.file_path,
        "relative_path": record.relative_path,
        "media_extension": record.media_extension,
        "mtime": record.mtime,
        "size": record.size,
        "file_hash": record.file_hash,
    }


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    text = series.where(series.notna(), "").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y"})


def _bool_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in BOOLEAN_REGISTRY_COLUMNS:
        normalized[column] = _coerce_bool_series(normalized[column])
    return normalized


def _source_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "youtube_wav"
    return "youtube_mp4"


def _build_file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def discover_media_files(
    input_dir: Path,
    recursive: bool = True,
    limit: int | None = None,
    extensions: tuple[str, ...] = SUPPORTED_MEDIA_EXTENSIONS,
    compute_hash: bool = False,
) -> list[ScanRecord]:
    ensure_project_dirs()
    input_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    normalized_extensions = {extension.lower() for extension in extensions}
    discovered: list[ScanRecord] = []
    for path in sorted(candidate for candidate in input_dir.glob(pattern) if candidate.is_file()):
        if path.suffix.lower() not in normalized_extensions:
            continue
        resolved = path.resolve()
        stat = resolved.stat()
        discovered.append(
            ScanRecord(
                source_type=_source_type_for_path(resolved),
                file_name=resolved.name,
                file_path=str(resolved),
                relative_path=resolved.relative_to(input_dir.resolve()).as_posix(),
                media_extension=resolved.suffix.lower(),
                mtime=float(stat.st_mtime),
                size=int(stat.st_size),
                file_hash=_build_file_hash(resolved) if compute_hash else "",
            )
        )

    if limit is not None:
        return discovered[:limit]
    return discovered


def load_registry(registry_path: Path = PROCESSED_REGISTRY_CSV) -> pd.DataFrame:
    if not registry_path.exists():
        return _bool_columns(pd.DataFrame(columns=list(REGISTRY_DEFAULTS.keys())))

    frame = pd.read_csv(registry_path)
    normalized = frame.copy()
    for column, default_value in REGISTRY_DEFAULTS.items():
        if column not in normalized.columns:
            normalized[column] = default_value
        if isinstance(default_value, bool):
            normalized[column] = _coerce_bool_series(normalized[column])
        elif isinstance(default_value, float):
            normalized[column] = normalized[column].fillna(default_value).astype(float)
        elif isinstance(default_value, int):
            normalized[column] = normalized[column].fillna(default_value).astype(int)
        else:
            normalized[column] = normalized[column].fillna(default_value).astype(str)
    return normalized[list(REGISTRY_DEFAULTS.keys())]


def save_registry(frame: pd.DataFrame, registry_path: Path = PROCESSED_REGISTRY_CSV) -> None:
    normalized = frame.copy()
    for column, default_value in REGISTRY_DEFAULTS.items():
        if column not in normalized.columns:
            normalized[column] = default_value
    normalized = _bool_columns(normalized[list(REGISTRY_DEFAULTS.keys())])
    normalized = normalized.sort_values(["file_path", "source_type"], kind="stable").reset_index(drop=True)
    save_dataframe(registry_path, normalized)


def _signatures_match(existing_row: pd.Series, record: ScanRecord) -> bool:
    existing_mtime = float(existing_row.get("mtime", 0.0))
    existing_size = int(existing_row.get("size", 0))
    if not math.isclose(existing_mtime, record.mtime, abs_tol=1e-6):
        existing_hash = str(existing_row.get("file_hash", "")).strip()
        if existing_hash and record.file_hash and existing_hash == record.file_hash:
            return True
        return False
    if existing_size != record.size:
        existing_hash = str(existing_row.get("file_hash", "")).strip()
        if existing_hash and record.file_hash and existing_hash == record.file_hash:
            return True
        return False
    existing_hash = str(existing_row.get("file_hash", "")).strip()
    if existing_hash and record.file_hash and existing_hash != record.file_hash:
        return False
    return True


def plan_incremental_update(
    input_dir: Path,
    registry_path: Path = PROCESSED_REGISTRY_CSV,
    recursive: bool = True,
    limit: int | None = None,
    compute_hash: bool = False,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    discovered_records = discover_media_files(
        input_dir=input_dir,
        recursive=recursive,
        limit=limit,
        compute_hash=compute_hash,
    )

    registry_lookup = {
        _normalize_path(row["file_path"]): row
        for _, row in registry.iterrows()
        if str(row.get("file_path", "")).strip()
    }

    new_records: list[ScanRecord] = []
    changed_records: list[ScanRecord] = []
    skipped_records: list[ScanRecord] = []
    status_map: dict[str, str] = {}

    for record in discovered_records:
        key = _normalize_path(record.file_path)
        existing = registry_lookup.get(key)
        if existing is None:
            new_records.append(record)
            status_map[key] = "new"
            continue
        if _signatures_match(existing, record):
            skipped_records.append(record)
            status_map[key] = "skipped"
            continue
        changed_records.append(record)
        status_map[key] = "changed"

    current_paths = {_normalize_path(record.file_path) for record in discovered_records}
    missing_records = []
    for _, row in registry.iterrows():
        file_path = str(row.get("file_path", "")).strip()
        if not file_path:
            continue
        normalized_path = _normalize_path(file_path)
        if normalized_path in current_paths:
            continue
        missing_records.append(row.to_dict())

    return {
        "registry_path": registry_path,
        "input_dir": input_dir,
        "discovered_records": discovered_records,
        "new_records": new_records,
        "changed_records": changed_records,
        "skipped_records": skipped_records,
        "target_records": [*new_records, *changed_records],
        "missing_records": missing_records,
        "status_map": status_map,
        "summary": {
            "total_files": len(discovered_records),
            "new_files": len(new_records),
            "changed_files": len(changed_records),
            "skipped_files": len(skipped_records),
            "missing_files": len(missing_records),
        },
    }


def _metadata_lookup(metadata: pd.DataFrame) -> dict[str, pd.Series]:
    normalized = ensure_metadata_columns(metadata)
    lookup: dict[str, pd.Series] = {}
    for _, row in normalized.iterrows():
        file_path = str(row.get("file_path", "")).strip()
        if not file_path:
            continue
        lookup[_normalize_path(file_path)] = row
    return lookup


def finalize_registry(
    plan: dict[str, Any],
    metadata: pd.DataFrame,
    registry_path: Path = PROCESSED_REGISTRY_CSV,
    embedding_built: bool | None = None,
) -> pd.DataFrame:
    ensure_project_dirs()
    previous_registry = load_registry(registry_path)
    previous_lookup = {
        _normalize_path(row["file_path"]): row.to_dict()
        for _, row in previous_registry.iterrows()
        if str(row.get("file_path", "")).strip()
    }
    metadata_lookup = _metadata_lookup(metadata)
    status_map = plan.get("status_map", {})
    target_paths = {_normalize_path(record.file_path) for record in plan.get("target_records", [])}
    current_records = plan.get("discovered_records", [])
    seen_paths = set()
    now = _utc_now_iso()
    rows: list[dict[str, Any]] = []

    for record in current_records:
        path_key = _normalize_path(record.file_path)
        seen_paths.add(path_key)
        previous = previous_lookup.get(path_key, {}).copy()
        row = {key: previous.get(key, default_value) for key, default_value in REGISTRY_DEFAULTS.items()}
        row.update(_record_to_dict(record))
        if not str(record.file_hash).strip() and str(previous.get("file_hash", "")).strip():
            row["file_hash"] = str(previous.get("file_hash", "")).strip()
        row["last_seen_at"] = now
        row["current_status"] = status_map.get(path_key, "skipped")

        metadata_row = metadata_lookup.get(path_key)
        if metadata_row is not None:
            audio_path = str(metadata_row.get("audio_path", "")).strip() or str(metadata_row.get("audio_file_path", "")).strip()
            transcript_path = str(metadata_row.get("stt_txt_path", "")).strip() or str(metadata_row.get("processed_txt_path", "")).strip()
            row["source_type"] = str(metadata_row.get("source_type", "")).strip() or row["source_type"]
            row["metadata_id"] = str(metadata_row.get("id", "")).strip()
            row["category"] = str(metadata_row.get("category", "")).strip()
            row["audio_path"] = audio_path
            row["stt_txt_path"] = transcript_path
            row["metadata_written"] = bool(row["metadata_id"])
            row["audio_extracted"] = bool(audio_path) and Path(audio_path).exists()
            row["stt_done"] = bool(transcript_path) and Path(transcript_path).exists()
            row["error_message"] = str(metadata_row.get("error_message", "")).strip()
        else:
            row["metadata_written"] = bool(previous.get("metadata_written", False))
            row["audio_extracted"] = bool(previous.get("audio_extracted", False))
            row["stt_done"] = bool(previous.get("stt_done", False))
            row["error_message"] = str(previous.get("error_message", "")).strip()

        if embedding_built is not None and path_key in target_paths:
            row["embedding_built"] = bool(embedding_built)
        else:
            row["embedding_built"] = bool(previous.get("embedding_built", False))

        if path_key in target_paths:
            row["last_processed_at"] = now

        rows.append(row)

    for path_key, previous in previous_lookup.items():
        if path_key in seen_paths:
            continue
        row = {key: previous.get(key, default_value) for key, default_value in REGISTRY_DEFAULTS.items()}
        row["current_status"] = "missing"
        rows.append(row)

    registry = pd.DataFrame(rows, columns=list(REGISTRY_DEFAULTS.keys()))
    save_registry(registry, registry_path)
    return registry


def write_incremental_run_summary(
    payload: dict[str, Any],
    summary_path: Path = INCREMENTAL_RUN_SUMMARY_JSON,
) -> Path:
    ensure_project_dirs()
    save_json(summary_path, payload)
    return summary_path


def load_incremental_run_summary(summary_path: Path = INCREMENTAL_RUN_SUMMARY_JSON) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    return load_json(summary_path)


def build_run_summary_payload(
    plan: dict[str, Any],
    processed_files: list[str],
    indices_rebuilt: bool,
    evaluation_ran: bool,
    artifact_namespace: str | None,
    registry_path: Path = PROCESSED_REGISTRY_CSV,
) -> dict[str, Any]:
    return {
        "run_at": _utc_now_iso(),
        "registry_path": str(registry_path.resolve()),
        "artifact_namespace": artifact_namespace,
        "summary": plan.get("summary", {}),
        "processed_files": processed_files,
        "skipped_files": [record.file_name for record in plan.get("skipped_records", [])],
        "new_files": [record.file_name for record in plan.get("new_records", [])],
        "changed_files": [record.file_name for record in plan.get("changed_records", [])],
        "missing_files": [str(item.get("file_name", "")) for item in plan.get("missing_records", [])],
        "indices_rebuilt": indices_rebuilt,
        "evaluation_ran": evaluation_ran,
    }
