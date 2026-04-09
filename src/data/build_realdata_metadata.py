from __future__ import annotations

import argparse
import hashlib
import re
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


FILENAME_STOPWORDS = {
    "ytdown",
    "com",
    "youtube",
    "media",
    "mp4",
    "001",
    "1080p",
    "720p",
    "480p",
}


def discover_mp4_files(input_dir: Path, recursive: bool = True, limit: int | None = None) -> list[Path]:
    pattern = "**/*.mp4" if recursive else "*.mp4"
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if limit is not None:
        return files[:limit]
    return files


def stable_mp4_id(relative_path: Path) -> str:
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:12].upper()
    return f"YTMP4-{digest}"


def infer_category(relative_path: Path) -> str:
    if relative_path.parent == Path("."):
        return "youtube_mp4"
    return relative_path.parent.name or "youtube_mp4"


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]{2,}", str(text))


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


def build_realdata_metadata(
    input_dir: Path = YOUTUBE_MP4_INPUT_DIR,
    metadata_path: Path = REALDATA_METADATA_CSV,
    audio_output_dir: Path = YOUTUBE_WAV_DIR,
    transcript_output_dir: Path = YOUTUBE_TRANSCRIPTS_DIR,
    recursive: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    ensure_project_dirs()
    input_dir.mkdir(parents=True, exist_ok=True)

    discovered_files = discover_mp4_files(input_dir=input_dir, recursive=recursive, limit=limit)
    existing_frame = _load_existing_frame(metadata_path)
    existing_lookup = _existing_row_lookup(existing_frame)

    records: list[dict[str, str]] = []
    discovered_ids: set[str] = set()
    for mp4_path in discovered_files:
        relative_path = mp4_path.relative_to(input_dir)
        record_id = stable_mp4_id(relative_path)
        discovered_ids.add(record_id)
        existing = existing_lookup.get(record_id, {})

        audio_path = audio_output_dir / relative_path.with_suffix(".wav")
        transcript_path = transcript_output_dir / relative_path.with_suffix(".txt")

        transcript_text = existing.get("stt_transcript", "").strip()
        if transcript_path.exists():
            transcript_text = transcript_path.read_text(encoding="utf-8").strip()

        filename_keywords = merge_keywords(relative_path.stem)
        transcript_keywords = infer_transcript_keywords(transcript_text) if transcript_text else ""
        merged_keywords = merge_keywords(existing.get("keywords", ""), filename_keywords, transcript_keywords)

        processing_status = existing.get("processing_status", "")
        if transcript_text and transcript_path.exists():
            processing_status = processing_status or "transcribed"
        elif audio_path.exists():
            processing_status = processing_status or "audio_extracted"
        else:
            processing_status = processing_status or "discovered"

        record = {
            "id": record_id,
            "source_type": "youtube_mp4",
            "category": existing.get("category", "").strip() or infer_category(relative_path),
            "title": existing.get("title", "").strip() or relative_path.stem,
            "file_name": mp4_path.name,
            "file_path": str(mp4_path.resolve()),
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
        }
        records.append(record)

    # Do not shrink the main metadata file when a debug run uses --limit.
    # Preserve previously known youtube rows that were not rediscovered only
    # because the caller intentionally limited the scan size.
    if limit is not None and not existing_frame.empty:
        for _, row in existing_frame.iterrows():
            existing_id = str(row.get("id", "")).strip()
            if not existing_id or existing_id in discovered_ids:
                continue
            if str(row.get("source_type", "")).strip() != "youtube_mp4":
                continue
            records.append({key: str(value) for key, value in row.items()})

    metadata = pd.DataFrame(records)
    if not metadata.empty and "file_path" in metadata.columns:
        metadata = metadata.sort_values(["file_path", "id"]).reset_index(drop=True)
    save_metadata_frame(metadata, metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover youtube mp4 files and build metadata CSV.")
    parser.add_argument("--input-dir", type=Path, default=YOUTUBE_MP4_INPUT_DIR)
    parser.add_argument("--metadata-path", type=Path, default=REALDATA_METADATA_CSV)
    parser.add_argument("--audio-output-dir", type=Path, default=YOUTUBE_WAV_DIR)
    parser.add_argument("--transcript-output-dir", type=Path, default=YOUTUBE_TRANSCRIPTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
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
    )
    print(f"Discovered {len(frame)} mp4 files")
    print(f"Metadata CSV: {args.metadata_path}")


if __name__ == "__main__":
    main()
