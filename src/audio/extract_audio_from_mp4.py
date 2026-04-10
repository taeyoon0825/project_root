from __future__ import annotations

import argparse
from pathlib import Path

from src.audio.audio_utils import convert_media_to_wav
from src.config import REALDATA_METADATA_CSV, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame, save_metadata_frame


def extract_audio_from_metadata(
    metadata_path: Path = REALDATA_METADATA_CSV,
    source_type: str | None = "youtube_mp4",
    sample_rate: int = 16000,
    overwrite: bool = False,
    skip_errors: bool = True,
    target_ids: set[str] | None = None,
) -> Path:
    ensure_project_dirs()
    metadata = load_metadata_frame(metadata_path)
    total_targets = (
        metadata["id"].astype(str).isin(target_ids).sum()
        if target_ids
        else len(metadata)
    )
    current_index = 0

    for row_index, row in metadata.iterrows():
        row_id = str(row.get("id", "")).strip()
        row_source_type = str(row.get("source_type", "")).strip()
        if source_type and row_source_type != source_type and not (row_source_type == "youtube_wav" and source_type == "youtube_mp4"):
            continue
        if target_ids and row_id not in target_ids:
            continue

        current_index += 1
        input_path = Path(str(row["file_path"]))
        output_value = str(row["audio_path"] or row["audio_file_path"]).strip()
        output_path = Path(output_value) if output_value else input_path.with_suffix(".wav")

        print(f"[audio {current_index}/{total_targets}] {input_path.name}")
        if not input_path.exists():
            message = f"Input media file does not exist: {input_path}"
            metadata.at[row_index, "processing_status"] = "audio_missing"
            metadata.at[row_index, "error_message"] = message
            if skip_errors:
                print(f"  - skipped: {message}")
                continue
            raise FileNotFoundError(message)

        try:
            if input_path.suffix.lower() == ".wav":
                output_path = input_path
                print(f"  - source wav reused {output_path}")
            elif overwrite or not output_path.exists():
                convert_media_to_wav(
                    input_path=input_path,
                    output_path=output_path,
                    sample_rate=sample_rate,
                    overwrite=True,
                )
                print(f"  - extracted to {output_path}")
            else:
                print(f"  - reused existing wav {output_path}")

            metadata.at[row_index, "audio_path"] = str(output_path.resolve())
            metadata.at[row_index, "audio_file_path"] = str(output_path.resolve())
            metadata.at[row_index, "audio_file_name"] = output_path.name
            metadata.at[row_index, "processing_status"] = "audio_extracted"
            metadata.at[row_index, "error_message"] = ""
        except Exception as exc:
            metadata.at[row_index, "processing_status"] = "audio_error"
            metadata.at[row_index, "error_message"] = str(exc)
            if skip_errors:
                print(f"  - failed: {exc}")
                continue
            raise

    save_metadata_frame(metadata, metadata_path)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WAV audio from discovered youtube mp4 files.")
    parser.add_argument("--metadata-path", type=Path, default=REALDATA_METADATA_CSV)
    parser.add_argument("--source-type", type=str, default="youtube_mp4")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", dest="skip_errors", action="store_true")
    parser.add_argument("--no-skip-errors", dest="skip_errors", action="store_false")
    parser.set_defaults(skip_errors=True)
    args = parser.parse_args()

    output_path = extract_audio_from_metadata(
        metadata_path=args.metadata_path,
        source_type=args.source_type,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
        skip_errors=args.skip_errors,
    )
    print(f"Audio extraction completed. Metadata updated at {output_path}")


if __name__ == "__main__":
    main()

