from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.audio.audio_utils import convert_to_wav
from src.config import AUDIO_TMP_DIR, AUDIO_WAV_DIR, DEFAULT_METADATA_CSV, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame, save_metadata_frame

try:
    import edge_tts
except ImportError:  # pragma: no cover
    edge_tts = None

try:
    from gtts import gTTS
except ImportError:  # pragma: no cover
    gTTS = None


RECOMMENDED_PROVIDER = "edge"
SUPPORTED_PROVIDERS = ("edge", "gtts")


async def _save_edge_tts_mp3(text: str, output_path: Path, voice: str) -> None:
    if edge_tts is None:
        raise ImportError("edge-tts is not installed. Run `pip install edge-tts`.")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_path))


def _save_gtts_mp3(text: str, output_path: Path) -> None:
    if gTTS is None:
        raise ImportError("gTTS is not installed. Run `pip install gTTS`.")
    gTTS(text=text, lang="ko").save(str(output_path))


def synthesize_one(
    text: str,
    wav_path: Path,
    tmp_mp3_path: Path,
    provider: str = RECOMMENDED_PROVIDER,
    edge_voice: str = "ko-KR-SunHiNeural",
) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3_path.parent.mkdir(parents=True, exist_ok=True)
    if provider == "edge":
        asyncio.run(_save_edge_tts_mp3(text, tmp_mp3_path, edge_voice))
    elif provider == "gtts":
        _save_gtts_mp3(text, tmp_mp3_path)
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")
    convert_to_wav(tmp_mp3_path, wav_path)


def generate_tts_audio_batch(
    metadata_path: Path = DEFAULT_METADATA_CSV,
    provider: str = RECOMMENDED_PROVIDER,
    edge_voice: str = "ko-KR-SunHiNeural",
    overwrite: bool = False,
) -> Path:
    ensure_project_dirs()
    metadata = load_metadata_frame(metadata_path)

    for row_index, row in metadata.iterrows():
        doc_number = row["id"].split("-")[-1] if row["id"] else f"{row_index + 1:03d}"
        audio_file_name = f"audio_{doc_number}.wav"
        wav_path = AUDIO_WAV_DIR / audio_file_name
        tmp_mp3_path = AUDIO_TMP_DIR / f"audio_{doc_number}.mp3"

        if overwrite or not wav_path.exists():
            source_text = row["tts_text"] or row["original_transcript"]
            synthesize_one(source_text, wav_path, tmp_mp3_path, provider=provider, edge_voice=edge_voice)

        metadata.at[row_index, "audio_file_name"] = audio_file_name
        metadata.at[row_index, "audio_file_path"] = str(wav_path.resolve())
        metadata.at[row_index, "tts_provider"] = provider

    save_metadata_frame(metadata, metadata_path)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WAV audio files from transcript text.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--provider", type=str, default=RECOMMENDED_PROVIDER, choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--edge-voice", type=str, default="ko-KR-SunHiNeural")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = generate_tts_audio_batch(
        metadata_path=args.metadata_path,
        provider=args.provider,
        edge_voice=args.edge_voice,
        overwrite=args.overwrite,
    )
    print(f"TTS audio generation completed. Metadata updated at {output_path}")
    print("Recommended provider: edge-tts")
    print("Alternative provider: gTTS")


if __name__ == "__main__":
    main()
