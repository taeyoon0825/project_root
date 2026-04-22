from __future__ import annotations

import os
from typing import Any

import torch


def stt_gpu_required() -> bool:
    return str(os.getenv("REQUIRE_GPU_FOR_STT", "true")).strip().lower() not in {"0", "false", "no"}


def _cuda_index_from_device(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Not a CUDA device string: {device}")


def resolve_torch_device() -> str:
    forced = str(os.getenv("APP_DEVICE", "")).strip().lower()
    if forced in {"cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}:
        if forced.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return forced
    gpu_index = str(os.getenv("APP_CUDA_INDEX", "0")).strip()
    if torch.cuda.is_available():
        return f"cuda:{gpu_index}"
    return "cpu"


def resolve_stt_device() -> str:
    require_gpu = stt_gpu_required()
    forced = str(os.getenv("APP_STT_DEVICE", "")).strip().lower()
    index = str(os.getenv("APP_STT_CUDA_INDEX", os.getenv("APP_CUDA_INDEX", "0"))).strip()
    target = forced or f"cuda:{index}"

    if not require_gpu:
        if forced == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return target
        return "cpu"

    if not torch.cuda.is_available():
        raise RuntimeError(
            "STT requires a dedicated CUDA GPU, but CUDA is unavailable. "
            "Check CUDA-enabled PyTorch installation, GPU driver, and CUDA runtime."
        )
    if not target.startswith("cuda"):
        raise RuntimeError(f"STT requires CUDA device, but got APP_STT_DEVICE={target!r}.")
    try:
        requested_idx = _cuda_index_from_device(target)
    except Exception as exc:
        raise RuntimeError(f"Invalid STT CUDA device format: {target!r}") from exc
    gpu_count = torch.cuda.device_count()
    if requested_idx < 0 or requested_idx >= gpu_count:
        raise RuntimeError(
            f"Invalid STT GPU index {requested_idx}. "
            f"Available CUDA device count={gpu_count}. Set APP_STT_DEVICE or APP_STT_CUDA_INDEX correctly."
        )
    return f"cuda:{requested_idx}"


def device_payload() -> dict[str, Any]:
    retrieval_device = resolve_torch_device()
    stt_device = ""
    stt_error = ""
    try:
        stt_device = resolve_stt_device()
    except Exception as exc:
        stt_error = str(exc)
    payload: dict[str, Any] = {
        "device": retrieval_device,
        "retrieval": retrieval_device,
        "stt": stt_device,
        "stt_error": stt_error,
        "stt_gpu_required": stt_gpu_required(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
    }
    if torch.cuda.is_available() and retrieval_device.startswith("cuda"):
        try:
            idx = int(retrieval_device.split(":")[1]) if ":" in retrieval_device else 0
            payload["gpu_name"] = torch.cuda.get_device_name(idx)
        except Exception:
            payload["gpu_name"] = "unknown"
    else:
        payload["gpu_name"] = ""
    return payload

