from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> None:
    if not ffmpeg_exists():
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install ffmpeg first because WAV conversion and Whisper decoding rely on it."
        )


def _run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        message = stderr or stdout or "Unknown ffmpeg error"
        raise RuntimeError(message)


def convert_media_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    overwrite: bool = True,
) -> None:
    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    _run_ffmpeg(command)


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> None:
    convert_media_to_wav(
        input_path=input_path,
        output_path=output_path,
        sample_rate=sample_rate,
        overwrite=True,
    )
