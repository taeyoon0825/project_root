from __future__ import annotations

from pathlib import Path

from src.config import HF_CACHE_DIR


def resolve_local_hf_snapshot(model_name: str) -> str | None:
    repo_dir = HF_CACHE_DIR / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_path = repo_dir / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / revision
        if snapshot_dir.exists():
            return str(snapshot_dir)

    snapshot_dirs = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.name)
    if snapshot_dirs:
        return str(snapshot_dirs[-1])
    return None
