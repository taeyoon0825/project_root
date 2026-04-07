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


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> None:
    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
