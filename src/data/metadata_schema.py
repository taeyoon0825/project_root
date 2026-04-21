from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_METADATA_CSV


METADATA_DEFAULTS = {
    "id": "",
    "source_type": "synthetic_dummy",
    "category": "",
    "title": "",
    "description": "",
    "file_name": "",
    "file_path": "",
    "audio_path": "",
    "processed_txt_path": "",
    "original_transcript": "",
    "stt_transcript": "",
    "tags": "",
    "keywords": "",
    "tts_text": "",
    "audio_file_name": "",
    "audio_file_path": "",
    "stt_txt_path": "",
    "stt_csv_path": "",
    "tts_provider": "",
    "stt_model_name": "",
    "stt_device": "",
    "processing_status": "",
    "error_message": "",
    "input_kind": "",
    "source_mtime": "",
    "source_size": "",
    "source_hash": "",
    "last_ingested_at": "",
}


def default_value_for(column: str) -> str:
    return str(METADATA_DEFAULTS[column])


def _fill_string_column(frame: pd.DataFrame, column: str) -> None:
    frame[column] = frame[column].fillna(default_value_for(column)).astype(str)


def _sync_alias_columns(frame: pd.DataFrame) -> None:
    frame["audio_path"] = frame["audio_path"].where(frame["audio_path"].str.len() > 0, frame["audio_file_path"])
    frame["audio_file_path"] = frame["audio_file_path"].where(
        frame["audio_file_path"].str.len() > 0,
        frame["audio_path"],
    )
    frame["processed_txt_path"] = frame["processed_txt_path"].where(
        frame["processed_txt_path"].str.len() > 0,
        frame["stt_txt_path"],
    )
    frame["stt_txt_path"] = frame["stt_txt_path"].where(
        frame["stt_txt_path"].str.len() > 0,
        frame["processed_txt_path"],
    )
    frame["audio_file_name"] = frame["audio_file_name"].where(
        frame["audio_file_name"].str.len() > 0,
        frame["audio_path"].apply(lambda value: Path(value).name if str(value).strip() else ""),
    )
    frame["title"] = frame["title"].where(
        frame["title"].str.len() > 0,
        frame["file_name"].apply(lambda value: Path(value).stem if str(value).strip() else ""),
    )
    frame["tags"] = frame["tags"].where(frame["tags"].str.len() > 0, frame["keywords"])
    frame["keywords"] = frame["keywords"].where(frame["keywords"].str.len() > 0, frame["tags"])


def ensure_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column, default_value in METADATA_DEFAULTS.items():
        if column not in normalized.columns:
            normalized[column] = default_value
    for column in METADATA_DEFAULTS:
        _fill_string_column(normalized, column)
    _sync_alias_columns(normalized)
    return normalized


def empty_metadata_frame() -> pd.DataFrame:
    return ensure_metadata_columns(pd.DataFrame(columns=list(METADATA_DEFAULTS.keys())))


def load_metadata_frame(metadata_path: Path = DEFAULT_METADATA_CSV) -> pd.DataFrame:
    return ensure_metadata_columns(pd.read_csv(metadata_path))


def save_metadata_frame(frame: pd.DataFrame, metadata_path: Path = DEFAULT_METADATA_CSV) -> None:
    normalized = ensure_metadata_columns(frame)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(metadata_path, index=False, encoding="utf-8-sig")
