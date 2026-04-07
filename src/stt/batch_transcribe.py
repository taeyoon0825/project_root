from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AUDIO_STT_DIR, DEFAULT_METADATA_CSV, WHISPER_CACHE_DIR, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame, save_metadata_frame

try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None


def transcribe_audio_batch(
    metadata_path: Path = DEFAULT_METADATA_CSV,
    model_name: str = "base",
    language: str = "ko",
    overwrite: bool = False,
) -> Path:
    if whisper is None:
        raise ImportError("openai-whisper is not installed. Run `pip install openai-whisper`.")

    ensure_project_dirs()
    metadata = load_metadata_frame(metadata_path)
    model = whisper.load_model(model_name, download_root=str(WHISPER_CACHE_DIR))

    for row_index, row in metadata.iterrows():
        audio_path = Path(row["audio_file_path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        doc_number = row["id"].split("-")[-1] if row["id"] else f"{row_index + 1:03d}"
        stt_txt_path = AUDIO_STT_DIR / f"stt_{doc_number}.txt"

        if overwrite or not stt_txt_path.exists():
            result = model.transcribe(
                str(audio_path),
                language=language,
                fp16=False,
                condition_on_previous_text=False,
                verbose=False,
            )
            transcript = result.get("text", "").strip()
            stt_txt_path.write_text(transcript, encoding="utf-8")
        else:
            transcript = stt_txt_path.read_text(encoding="utf-8").strip()

        metadata.at[row_index, "stt_transcript"] = transcript
        metadata.at[row_index, "stt_txt_path"] = str(stt_txt_path.resolve())
        metadata.at[row_index, "stt_model_name"] = model_name

    save_metadata_frame(metadata, metadata_path)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe WAV files with Whisper.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--model-name", type=str, default="base")
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = transcribe_audio_batch(
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        language=args.language,
        overwrite=args.overwrite,
    )
    print(f"STT transcription completed. Metadata updated at {output_path}")


if __name__ == "__main__":
    main()
