from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.adaptive.parameter_resolver import build_adaptive_context
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


def _save_gtts_mp3(text: str, output_path: Path, lang: str) -> None:
    if gTTS is None:
        raise ImportError("gTTS is not installed. Run `pip install gTTS`.")
    gTTS(text=text, lang=lang).save(str(output_path))


def synthesize_one(
    text: str,
    wav_path: Path,
    tmp_mp3_path: Path,
    *,
    provider: str = RECOMMENDED_PROVIDER,
    edge_voice: str | None = None,
    gtts_lang: str | None = None,
    sample_rate: int | None = None,
) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3_path.parent.mkdir(parents=True, exist_ok=True)
    if provider == "edge":
        asyncio.run(_save_edge_tts_mp3(text, tmp_mp3_path, edge_voice or "en-US-AriaNeural"))
    elif provider == "gtts":
        _save_gtts_mp3(text, tmp_mp3_path, gtts_lang or "en")
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")
    convert_to_wav(tmp_mp3_path, wav_path, sample_rate=sample_rate)


def generate_tts_audio_batch(
    metadata_path: Path = DEFAULT_METADATA_CSV,
    provider: str | None = None,
    edge_voice: str | None = None,
    overwrite: bool = False,
    sample_rate: int | None = None,
) -> Path:
    ensure_project_dirs()
    metadata = load_metadata_frame(metadata_path)
    adaptive_context = build_adaptive_context(metadata, text_source="original_transcript")
    resolved_provider = str(provider or adaptive_context.language.tts_provider or RECOMMENDED_PROVIDER).strip().lower()
    if resolved_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported TTS provider: {resolved_provider}")
    resolved_edge_voice = edge_voice or adaptive_context.language.edge_voice
    resolved_gtts_lang = adaptive_context.language.gtts_lang
    resolved_sample_rate = sample_rate if sample_rate is not None else adaptive_context.language.sample_rate

    for row_index, row in metadata.iterrows():
        doc_number = row["id"].split("-")[-1] if row["id"] else f"{row_index + 1:03d}"
        audio_file_name = f"audio_{doc_number}.wav"
        wav_path = AUDIO_WAV_DIR / audio_file_name
        tmp_mp3_path = AUDIO_TMP_DIR / f"audio_{doc_number}.mp3"

        if overwrite or not wav_path.exists():
            source_text = str(row.get("tts_text") or row.get("original_transcript") or row.get("stt_transcript") or "").strip()
            synthesize_one(
                source_text,
                wav_path,
                tmp_mp3_path,
                provider=resolved_provider,
                edge_voice=resolved_edge_voice,
                gtts_lang=resolved_gtts_lang,
                sample_rate=resolved_sample_rate,
            )

        metadata.at[row_index, "audio_file_name"] = audio_file_name
        metadata.at[row_index, "audio_file_path"] = str(wav_path.resolve())
        metadata.at[row_index, "tts_provider"] = resolved_provider
        metadata.at[row_index, "tts_voice"] = resolved_edge_voice if resolved_provider == "edge" else ""
        metadata.at[row_index, "tts_language"] = resolved_gtts_lang
        metadata.at[row_index, "tts_sample_rate"] = resolved_sample_rate or ""
        metadata.at[row_index, "adaptive_language_reason"] = adaptive_context.language.reasoning

    save_metadata_frame(metadata, metadata_path)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WAV audio files from transcript text.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--edge-voice", type=str, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = generate_tts_audio_batch(
        metadata_path=args.metadata_path,
        provider=args.provider,
        edge_voice=args.edge_voice,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
    )
    print(f"TTS audio generation completed. Metadata updated at {output_path}")


if __name__ == "__main__":
    main()
