from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    REALDATA_METADATA_CSV,
    YOUTUBE_MP4_INPUT_DIR,
    YOUTUBE_TRANSCRIPTS_DIR,
    YOUTUBE_WAV_DIR,
    ensure_project_dirs,
)
from src.data.metadata_schema import empty_metadata_frame, load_metadata_frame, save_metadata_frame
from src.ingest.incremental_registry import SUPPORTED_MEDIA_EXTENSIONS, discover_media_files


FILENAME_STOPWORDS = {
    "ytdown",
    "com",
    "youtube",
    "media",
    "mp4",
    "wav",
    "001",
    "1080p",
    "720p",
    "480p",
}


def discover_mp4_files(input_dir: Path, recursive: bool = True, limit: int | None = None) -> list[Path]:
    records = discover_media_files(
        input_dir=input_dir,
        recursive=recursive,
        limit=limit,
        extensions=(".mp4",),
        compute_hash=False,
    )
    return [Path(record.file_path) for record in records]


def discover_media_input_files(
    input_dir: Path,
    recursive: bool = True,
    limit: int | None = None,
) -> list[Path]:
    records = discover_media_files(
        input_dir=input_dir,
        recursive=recursive,
        limit=limit,
        extensions=SUPPORTED_MEDIA_EXTENSIONS,
        compute_hash=False,
    )
    return [Path(record.file_path) for record in records]


def stable_media_id(relative_path: Path) -> str:
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:12].upper()
    prefix = "YTWAV" if relative_path.suffix.lower() == ".wav" else "YTMP4"
    return f"{prefix}-{digest}"


def stable_mp4_id(relative_path: Path) -> str:
    return stable_media_id(relative_path)


def infer_source_type(relative_path: Path) -> str:
    return "youtube_wav" if relative_path.suffix.lower() == ".wav" else "youtube_mp4"


def infer_category(relative_path: Path, source_type: str) -> str:
    if relative_path.parent == Path("."):
        return source_type
    return relative_path.parent.name or source_type


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z\uac00-\ud7a3]{2,}", str(text))


def merge_keywords(*keyword_values: str) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in keyword_values:
        for token in tokenize_text(value):
            lowered = token.lower()
            if lowered in FILENAME_STOPWORDS or lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(token)
    return ", ".join(ordered)


def infer_transcript_keywords(transcript: str, max_keywords: int = 12) -> str:
    tokens = [token for token in tokenize_text(transcript) if len(token) >= 2]
    counts: dict[str, int] = {}
    originals: dict[str, str] = {}
    for token in tokens:
        lowered = token.lower()
        if lowered in FILENAME_STOPWORDS:
            continue
        counts[lowered] = counts.get(lowered, 0) + 1
        originals.setdefault(lowered, token)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(originals[key] for key, _ in ranked[:max_keywords])


def _load_existing_frame(metadata_path: Path) -> pd.DataFrame:
    if metadata_path.exists():
        return load_metadata_frame(metadata_path)
    return empty_metadata_frame()


def _existing_row_lookup(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    if frame.empty:
        return {}
    return {
        str(row["id"]): {key: str(value) for key, value in row.items()}
        for _, row in frame.iterrows()
        if str(row.get("id", "")).strip()
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _transcript_relative_path(relative_path: Path) -> Path:
    if relative_path.suffix.lower() == ".wav":
        filename = f"{relative_path.stem}__wav.txt"
        return relative_path.with_name(filename)
    return relative_path.with_suffix(".txt")


def _resolve_audio_output_path(
    source_path: Path,
    relative_path: Path,
    audio_output_dir: Path,
) -> Path:
    if source_path.suffix.lower() == ".wav":
        return source_path
    return audio_output_dir / relative_path.with_suffix(".wav")


def _preserve_existing_rows(existing_frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {
        str(row["id"]): {key: str(value) for key, value in row.items()}
        for _, row in existing_frame.iterrows()
        if str(row.get("id", "")).strip()
    }


def build_realdata_metadata(
    input_dir: Path = YOUTUBE_MP4_INPUT_DIR,
    metadata_path: Path = REALDATA_METADATA_CSV,
    audio_output_dir: Path = YOUTUBE_WAV_DIR,
    transcript_output_dir: Path = YOUTUBE_TRANSCRIPTS_DIR,
    recursive: bool = True,
    limit: int | None = None,
    media_files: list[Path] | None = None,
    preserve_missing: bool = True,
    compute_hash: bool = False,
) -> pd.DataFrame:
    ensure_project_dirs()
    input_dir.mkdir(parents=True, exist_ok=True)

    discovered_files = media_files or discover_media_input_files(
        input_dir=input_dir,
        recursive=recursive,
        limit=limit,
    )
    existing_frame = _load_existing_frame(metadata_path)
    existing_lookup = _existing_row_lookup(existing_frame)
    records_by_id: dict[str, dict[str, str]] = _preserve_existing_rows(existing_frame) if preserve_missing else {}
    input_dir_resolved = input_dir.resolve()
    run_timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    for source_path in discovered_files:
        resolved_source = source_path.resolve()
        relative_path = resolved_source.relative_to(input_dir_resolved)
        record_id = stable_media_id(relative_path)
        existing = existing_lookup.get(record_id, {})
        source_type = existing.get("source_type", "").strip() or infer_source_type(relative_path)

        audio_path = _resolve_audio_output_path(resolved_source, relative_path, audio_output_dir)
        transcript_path = transcript_output_dir / _transcript_relative_path(relative_path)

        transcript_text = existing.get("stt_transcript", "").strip()
        if transcript_path.exists():
            transcript_text = transcript_path.read_text(encoding="utf-8").strip()

        filename_keywords = merge_keywords(relative_path.stem)
        transcript_keywords = infer_transcript_keywords(transcript_text) if transcript_text else ""
        merged_keywords = merge_keywords(existing.get("keywords", ""), filename_keywords, transcript_keywords)

        if transcript_text and transcript_path.exists():
            processing_status = "transcribed"
        elif audio_path.exists():
            processing_status = "audio_extracted"
        else:
            processing_status = existing.get("processing_status", "").strip() or "discovered"

        stat = resolved_source.stat()
        source_hash = _file_hash(resolved_source) if compute_hash else existing.get("source_hash", "").strip()
        record = {
            "id": record_id,
            "source_type": source_type,
            "category": existing.get("category", "").strip() or infer_category(relative_path, source_type),
            "title": existing.get("title", "").strip() or resolved_source.stem,
            "file_name": resolved_source.name,
            "file_path": str(resolved_source),
            "audio_path": str(audio_path.resolve()),
            "processed_txt_path": str(transcript_path.resolve()),
            "original_transcript": existing.get("original_transcript", "").strip(),
            "stt_transcript": transcript_text,
            "keywords": merged_keywords,
            "tts_text": existing.get("tts_text", "").strip(),
            "audio_file_name": audio_path.name,
            "audio_file_path": str(audio_path.resolve()),
            "stt_txt_path": str(transcript_path.resolve()),
            "tts_provider": existing.get("tts_provider", "").strip(),
            "stt_model_name": existing.get("stt_model_name", "").strip(),
            "processing_status": processing_status,
            "error_message": existing.get("error_message", "").strip(),
            "input_kind": resolved_source.suffix.lower().lstrip("."),
            "source_mtime": str(float(stat.st_mtime)),
            "source_size": str(int(stat.st_size)),
            "source_hash": source_hash,
            "last_ingested_at": run_timestamp,
        }
        records_by_id[record_id] = record

    metadata = pd.DataFrame(records_by_id.values())
    if not metadata.empty:
        metadata = metadata.reset_index(drop=True)
    save_metadata_frame(metadata, metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover youtube mp4/wav files and build metadata CSV.")
    parser.add_argument("--input-dir", type=Path, default=YOUTUBE_MP4_INPUT_DIR)
    parser.add_argument("--metadata-path", type=Path, default=REALDATA_METADATA_CSV)
    parser.add_argument("--audio-output-dir", type=Path, default=YOUTUBE_WAV_DIR)
    parser.add_argument("--transcript-output-dir", type=Path, default=YOUTUBE_TRANSCRIPTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compute-hash", action="store_true")
    parser.add_argument("--preserve-missing", dest="preserve_missing", action="store_true")
    parser.add_argument("--remove-missing", dest="preserve_missing", action="store_false")
    parser.set_defaults(preserve_missing=True)
    parser.add_argument("--recursive", dest="recursive", action="store_true")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)
    args = parser.parse_args()

    frame = build_realdata_metadata(
        input_dir=args.input_dir,
        metadata_path=args.metadata_path,
        audio_output_dir=args.audio_output_dir,
        transcript_output_dir=args.transcript_output_dir,
        recursive=args.recursive,
        limit=args.limit,
        preserve_missing=args.preserve_missing,
        compute_hash=args.compute_hash,
    )
    print(f"Discovered {len(frame)} media files")
    print(f"Metadata CSV: {args.metadata_path}")


if __name__ == "__main__":
    main()
