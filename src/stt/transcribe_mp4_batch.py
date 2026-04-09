from __future__ import annotations

import argparse
from pathlib import Path

from src.config import REALDATA_METADATA_CSV
from src.stt.batch_transcribe import transcribe_audio_batch


def transcribe_mp4_batch(
    metadata_path: Path = REALDATA_METADATA_CSV,
    model_name: str = "base",
    language: str | None = "ko",
    overwrite: bool = False,
    skip_errors: bool = True,
) -> Path:
    return transcribe_audio_batch(
        metadata_path=metadata_path,
        model_name=model_name,
        language=language,
        overwrite=overwrite,
        source_type="youtube_mp4",
        skip_errors=skip_errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe extracted youtube mp4 audio with Whisper.")
    parser.add_argument("--metadata-path", type=Path, default=REALDATA_METADATA_CSV)
    parser.add_argument("--model-name", type=str, default="base")
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", dest="skip_errors", action="store_true")
    parser.add_argument("--no-skip-errors", dest="skip_errors", action="store_false")
    parser.set_defaults(skip_errors=True)
    args = parser.parse_args()

    output_path = transcribe_mp4_batch(
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        language=args.language,
        overwrite=args.overwrite,
        skip_errors=args.skip_errors,
    )
    print(f"youtube mp4 STT completed. Metadata updated at {output_path}")


if __name__ == "__main__":
    main()
