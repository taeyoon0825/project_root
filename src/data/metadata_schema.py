from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_METADATA_CSV


METADATA_DEFAULTS = {
    "id": "",
    "category": "",
    "title": "",
    "file_name": "",
    "file_path": "",
    "processed_txt_path": "",
    "original_transcript": "",
    "stt_transcript": "",
    "keywords": "",
    "tts_text": "",
    "audio_file_name": "",
    "audio_file_path": "",
    "stt_txt_path": "",
    "tts_provider": "",
    "stt_model_name": "",
}


def ensure_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column, default_value in METADATA_DEFAULTS.items():
        if column not in normalized.columns:
            normalized[column] = default_value
    for column in METADATA_DEFAULTS:
        normalized[column] = normalized[column].fillna(default_value_for(column)).astype(str)
    return normalized


def default_value_for(column: str) -> str:
    return str(METADATA_DEFAULTS[column])


def load_metadata_frame(metadata_path: Path = DEFAULT_METADATA_CSV) -> pd.DataFrame:
    return ensure_metadata_columns(pd.read_csv(metadata_path))


def save_metadata_frame(frame: pd.DataFrame, metadata_path: Path = DEFAULT_METADATA_CSV) -> None:
    normalized = ensure_metadata_columns(frame)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(metadata_path, index=False, encoding="utf-8-sig")
