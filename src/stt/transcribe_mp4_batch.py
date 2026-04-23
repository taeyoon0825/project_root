from __future__ import annotations

"""유튜브 mp4 계열 오디오만 대상으로 STT를 실행하는 얇은 래퍼.

일반 배치 전사 함수 위에 source_type 필터만 고정한 파일이다.
이 파일이 있으면 자주 쓰는 mp4 전사 흐름을 CLI에서 짧게 호출할 수 있다.
"""

import argparse
from pathlib import Path

from src.config import REALDATA_METADATA_CSV
from src.stt.batch_transcribe import transcribe_audio_batch


def transcribe_mp4_batch(
    metadata_path: Path = REALDATA_METADATA_CSV,
    model_name: str | None = None,
    language: str | None = None,
    overwrite: bool = False,
    skip_errors: bool = True,
    target_ids: set[str] | None = None,
) -> Path:
    """일반 STT 일괄 처리 함수를 유튜브 mp4 전용 흐름으로 고정한다.

    이 래퍼가 있으면 호출부가 메타데이터 스키마의 정확한 source_type 필터를
    매번 기억하지 않아도 된다.
    """
    return transcribe_audio_batch(
        metadata_path=metadata_path,
        model_name=model_name,
        language=language,
        overwrite=overwrite,
        source_type="youtube_mp4",
        skip_errors=skip_errors,
        target_ids=target_ids,
    )


def main() -> None:
    """가장 자주 쓰는 mp4 전사 흐름을 위한 전용 CLI 진입점."""
    parser = argparse.ArgumentParser(description="Transcribe extracted youtube mp4 audio with Whisper.")
    parser.add_argument("--metadata-path", type=Path, default=REALDATA_METADATA_CSV)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)
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
