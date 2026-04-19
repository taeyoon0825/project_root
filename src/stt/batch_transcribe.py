from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from src.config import AUDIO_STT_DIR, DEFAULT_METADATA_CSV, WHISPER_CACHE_DIR, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame, save_metadata_frame
from src.utils.io_utils import save_json

try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", value.strip())
    return cleaned.strip("._") or "transcript"


def _safe_print(message: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(str(message).encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _row_source_type(row) -> str:
    return str(row.get("source_type", "")).strip()


def _resolve_audio_path(row) -> Path:
    value = str(row.get("audio_path", "")).strip() or str(row.get("audio_file_path", "")).strip()
    return Path(value)


def _resolve_stt_output_path(row, row_index: int) -> Path:
    explicit_stt_path = str(row.get("stt_txt_path", "")).strip()
    if explicit_stt_path:
        return Path(explicit_stt_path)

    source_type = _row_source_type(row)
    if source_type in {"youtube_mp4", "youtube_wav"}:
        explicit_processed_path = str(row.get("processed_txt_path", "")).strip()
        if explicit_processed_path:
            return Path(explicit_processed_path)

    doc_id = str(row.get("id", "")).strip() or f"row_{row_index + 1:04d}"
    return AUDIO_STT_DIR / f"{_safe_stem(doc_id)}.txt"


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = str(language).strip().lower()
    if not normalized or normalized == "auto":
        return None
    return normalized


def _segments_output_path(stt_txt_path: Path) -> Path:
    return stt_txt_path.with_suffix(".segments.json")


def transcribe_audio_batch(
    metadata_path: Path = DEFAULT_METADATA_CSV,
    model_name: str = "base",
    language: str | None = "ko",
    overwrite: bool = False,
    source_type: str | None = None,
    skip_errors: bool = True,
    target_ids: set[str] | None = None,
) -> Path:
    if whisper is None:
        raise ImportError("openai-whisper is not installed. Run `pip install openai-whisper`.")

    ensure_project_dirs()
    metadata = load_metadata_frame(metadata_path)
    candidate_indices: list[int] = []
    for row_index, row in metadata.iterrows():
        row_id = str(row.get("id", "")).strip()
        row_source_type = _row_source_type(row)
        if source_type and row_source_type != source_type and not (row_source_type == "youtube_wav" and source_type == "youtube_mp4"):
            continue
        if target_ids and row_id not in target_ids:
            continue
        candidate_indices.append(row_index)

    if not candidate_indices:
        _safe_print("[STT] No target files selected. STT skipped.")
        return metadata_path

    model = whisper.load_model(model_name, download_root=str(WHISPER_CACHE_DIR))
    whisper_language = _normalize_language(language)
    total = len(candidate_indices)

    for progress_index, row_index in enumerate(candidate_indices, start=1):
        row = metadata.iloc[row_index]
        row_source_type = _row_source_type(row)
        audio_path = _resolve_audio_path(row)
        stt_txt_path = _resolve_stt_output_path(row, row_index)
        stt_txt_path.parent.mkdir(parents=True, exist_ok=True)

        _safe_print(f"[stt {progress_index}/{total}] {row.get('file_name', row.get('id', 'unknown'))}")
        if not audio_path.exists():
            message = f"Audio file does not exist: {audio_path}"
            metadata.at[row_index, "processing_status"] = "stt_audio_missing"
            metadata.at[row_index, "error_message"] = message
            if skip_errors:
                _safe_print(f"  - skipped: {message}")
                continue
            raise FileNotFoundError(message)

        try:
            if overwrite or not stt_txt_path.exists():
                result = model.transcribe(
                    str(audio_path),
                    language=whisper_language,
                    fp16=False,
                    condition_on_previous_text=False,
                    verbose=False,
                )
                transcript = str(result.get("text", "")).strip()
                stt_txt_path.write_text(transcript, encoding="utf-8")
                save_json(
                    _segments_output_path(stt_txt_path),
                    {
                        "text": transcript,
                        "language": result.get("language"),
                        "segments": result.get("segments", []),
                    },
                )
                _safe_print(f"  - transcribed to {stt_txt_path}")
            else:
                transcript = stt_txt_path.read_text(encoding="utf-8").strip()
                segments_path = _segments_output_path(stt_txt_path)
                if not segments_path.exists():
                    save_json(
                        segments_path,
                        {
                            "text": transcript,
                            "language": whisper_language,
                            "segments": [],
                        },
                    )
                _safe_print(f"  - reused existing transcript {stt_txt_path}")

            metadata.at[row_index, "stt_transcript"] = transcript
            metadata.at[row_index, "stt_txt_path"] = str(stt_txt_path.resolve())
            metadata.at[row_index, "stt_model_name"] = model_name
            metadata.at[row_index, "error_message"] = ""

            if row_source_type in {"youtube_mp4", "youtube_wav"}:
                metadata.at[row_index, "processed_txt_path"] = str(stt_txt_path.resolve())
                metadata.at[row_index, "processing_status"] = "transcribed"
            else:
                metadata.at[row_index, "processing_status"] = "stt_completed"
        except Exception as exc:
            metadata.at[row_index, "processing_status"] = "stt_error"
            metadata.at[row_index, "error_message"] = str(exc)
            if skip_errors:
                _safe_print(f"  - failed: {exc}")
                continue
            raise

    save_metadata_frame(metadata, metadata_path)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--model-name", type=str, default="base")
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-type", type=str, default=None)
    parser.add_argument("--skip-errors", dest="skip_errors", action="store_true")
    parser.add_argument("--no-skip-errors", dest="skip_errors", action="store_false")
    parser.set_defaults(skip_errors=True)
    args = parser.parse_args()

    output_path = transcribe_audio_batch(
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        language=args.language,
        overwrite=args.overwrite,
        source_type=args.source_type,
        skip_errors=args.skip_errors,
    )
    _safe_print(f"STT transcription completed. Metadata updated at {output_path}")


if __name__ == "__main__":
    main()
