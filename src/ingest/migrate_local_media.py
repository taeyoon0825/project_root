from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.audio.audio_utils import convert_media_to_wav
from src.config import (
    DATA_DIR,
    HTML_UPLOAD_MEDIA_DIR,
    HTML_UPLOAD_TRANSCRIPTS_DIR,
    HTML_UPLOAD_WAV_DIR,
    REALDATA_METADATA_CSV,
    ensure_project_dirs,
)
from src.data.metadata_schema import ensure_metadata_columns, load_metadata_frame, save_metadata_frame
from src.stt.batch_transcribe import transcribe_audio_batch


SUPPORTED_MEDIA_EXTENSIONS = {".mp4", ".wav"}
UNIFIED_SOURCE_TYPE = "local_media"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", Path(value).name).strip("._")
    return cleaned or "media"


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _discover_media_files(root: Path = DATA_DIR) -> list[Path]:
    upload_root = HTML_UPLOAD_MEDIA_DIR.parent.resolve()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
            continue
        resolved = path.resolve()
        if upload_root == resolved or upload_root in resolved.parents:
            continue
        files.append(resolved)
    return sorted(files)


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _current_transcript_path(file_name: str) -> Path | None:
    candidate = DATA_DIR / "transcripts" / "youtube_mp4" / f"{Path(file_name).stem}.txt"
    return candidate.resolve() if candidate.exists() else None


def _repair_existing_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    repaired = ensure_metadata_columns(metadata)
    for index, row in repaired.iterrows():
        file_name = str(row.get("file_name", "")).strip()
        transcript_path = Path(str(row.get("stt_txt_path", "")).strip())
        if not transcript_path.exists() and file_name:
            current_path = _current_transcript_path(file_name)
            if current_path:
                repaired.at[index, "stt_txt_path"] = str(current_path)
                repaired.at[index, "processed_txt_path"] = str(current_path)
                if not str(row.get("stt_transcript", "")).strip():
                    repaired.at[index, "stt_transcript"] = _read_text_if_exists(current_path)
        repaired.at[index, "source_type"] = UNIFIED_SOURCE_TYPE
    return repaired


def _metadata_row_for_media(path: Path, file_hash: str) -> dict[str, Any]:
    doc_id = f"LOCAL-{file_hash[:12]}"
    safe = _safe_name(path.name)
    source_path = HTML_UPLOAD_MEDIA_DIR / f"{doc_id}_{safe}"
    wav_path = HTML_UPLOAD_WAV_DIR / f"{doc_id}.wav"
    transcript_path = HTML_UPLOAD_TRANSCRIPTS_DIR / f"{doc_id}.txt"
    category = path.parent.name if path.parent != DATA_DIR else UNIFIED_SOURCE_TYPE
    return {
        "id": doc_id,
        "source_type": UNIFIED_SOURCE_TYPE,
        "category": category,
        "title": path.stem,
        "description": "",
        "file_name": path.name,
        "file_path": str(source_path.resolve()),
        "audio_path": str(wav_path.resolve()),
        "processed_txt_path": str(transcript_path.resolve()),
        "original_transcript": "",
        "stt_transcript": _read_text_if_exists(transcript_path),
        "tags": category,
        "keywords": category,
        "tts_text": "",
        "audio_file_name": wav_path.name,
        "audio_file_path": str(wav_path.resolve()),
        "stt_txt_path": str(transcript_path.resolve()),
        "tts_provider": "",
        "stt_model_name": "",
        "processing_status": "transcribed" if transcript_path.exists() else "migrated",
        "error_message": "",
        "input_kind": path.suffix.lower().lstrip("."),
        "source_mtime": str(path.stat().st_mtime_ns),
        "source_size": str(path.stat().st_size),
        "source_hash": file_hash,
        "last_ingested_at": "",
    }


def _copy_and_convert(source: Path, row: dict[str, Any], overwrite: bool) -> None:
    target = Path(str(row["file_path"]))
    wav_path = Path(str(row["audio_path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not target.exists():
        shutil.copy2(source, target)
    if overwrite or not wav_path.exists():
        convert_media_to_wav(target, wav_path, overwrite=True)


def migrate_local_media(
    *,
    metadata_path: Path = REALDATA_METADATA_CSV,
    overwrite_media: bool = False,
    transcribe: bool = True,
    whisper_model: str = "base",
    language: str = "ko",
) -> dict[str, Any]:
    ensure_project_dirs()
    HTML_UPLOAD_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    HTML_UPLOAD_WAV_DIR.mkdir(parents=True, exist_ok=True)
    HTML_UPLOAD_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata_frame(metadata_path) if metadata_path.exists() else ensure_metadata_columns(pd.DataFrame())
    metadata = _repair_existing_rows(metadata)

    existing_hashes = set(metadata["source_hash"].fillna("").astype(str))
    imported_rows: list[dict[str, Any]] = []
    copied_count = 0
    for source in _discover_media_files(DATA_DIR):
        file_hash = _sha1_file(source)
        row = _metadata_row_for_media(source, file_hash)
        doc_id = str(row["id"])
        should_copy = overwrite_media or file_hash not in existing_hashes or not Path(str(row["file_path"])).exists()
        if should_copy:
            _copy_and_convert(source, row, overwrite=overwrite_media)
            copied_count += 1

        existing = metadata["id"].astype(str) == doc_id
        if existing.any():
            for key, value in row.items():
                if key == "stt_transcript" and str(metadata.loc[existing, key].iloc[0]).strip():
                    continue
                metadata.loc[existing, key] = value
        else:
            imported_rows.append(row)
            existing_hashes.add(file_hash)

    if imported_rows:
        metadata = pd.concat([metadata, ensure_metadata_columns(pd.DataFrame(imported_rows))], ignore_index=True)

    metadata["source_type"] = UNIFIED_SOURCE_TYPE
    save_metadata_frame(metadata, metadata_path)

    target_ids: set[str] = set()
    refreshed = load_metadata_frame(metadata_path)
    for _, row in refreshed.iterrows():
        doc_id = str(row.get("id", "")).strip()
        transcript = str(row.get("stt_transcript", "")).strip()
        transcript_path = Path(str(row.get("stt_txt_path", "")).strip())
        audio_path = Path(str(row.get("audio_path", "")).strip())
        if doc_id.startswith("LOCAL-") and audio_path.exists() and (not transcript or not transcript_path.exists()):
            target_ids.add(doc_id)

    if transcribe and target_ids:
        transcribe_audio_batch(
            metadata_path=metadata_path,
            model_name=whisper_model,
            language=language,
            overwrite=False,
            target_ids=target_ids,
            skip_errors=False,
        )

    final_metadata = load_metadata_frame(metadata_path)
    final_metadata["source_type"] = UNIFIED_SOURCE_TYPE
    save_metadata_frame(final_metadata, metadata_path)

    return {
        "metadata_path": str(metadata_path),
        "discovered_media_count": len(_discover_media_files(DATA_DIR)),
        "copied_or_converted_count": copied_count,
        "new_rows_count": len(imported_rows),
        "transcribed_count": len(target_ids),
        "row_count": int(len(final_metadata)),
        "source_types": sorted(final_metadata["source_type"].dropna().astype(str).unique().tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate local MP4/WAV files into the unified HTML upload dataset.")
    parser.add_argument("--metadata-path", type=Path, default=REALDATA_METADATA_CSV)
    parser.add_argument("--overwrite-media", action="store_true")
    parser.add_argument("--no-transcribe", action="store_true")
    parser.add_argument("--whisper-model", type=str, default="base")
    parser.add_argument("--language", type=str, default="ko")
    args = parser.parse_args()
    result = migrate_local_media(
        metadata_path=args.metadata_path,
        overwrite_media=args.overwrite_media,
        transcribe=not args.no_transcribe,
        whisper_model=args.whisper_model,
        language=args.language,
    )
    print(result)


if __name__ == "__main__":
    main()
