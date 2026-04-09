from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import COMBINED_METADATA_CSV, DEFAULT_METADATA_CSV, REALDATA_METADATA_CSV
from src.data.metadata_schema import empty_metadata_frame, load_metadata_frame, save_metadata_frame


def merge_metadata_files(
    metadata_paths: list[Path],
    output_path: Path = COMBINED_METADATA_CSV,
    skip_missing: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metadata_path in metadata_paths:
        if metadata_path.exists():
            frames.append(load_metadata_frame(metadata_path))
        elif not skip_missing:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not frames:
        merged = empty_metadata_frame()
    else:
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)

    save_metadata_frame(merged, output_path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge dummy metadata and youtube mp4 metadata into one CSV.")
    parser.add_argument(
        "--metadata-paths",
        nargs="+",
        type=Path,
        default=[DEFAULT_METADATA_CSV, REALDATA_METADATA_CSV],
    )
    parser.add_argument("--output-path", type=Path, default=COMBINED_METADATA_CSV)
    parser.add_argument("--skip-missing", dest="skip_missing", action="store_true")
    parser.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    parser.set_defaults(skip_missing=True)
    args = parser.parse_args()

    merged = merge_metadata_files(
        metadata_paths=args.metadata_paths,
        output_path=args.output_path,
        skip_missing=args.skip_missing,
    )
    print(f"Merged metadata rows: {len(merged)}")
    print(f"Combined metadata CSV: {args.output_path}")


if __name__ == "__main__":
    main()
