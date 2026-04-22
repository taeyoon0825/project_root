from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src.config import ADAPTIVE_TUNING_JSON


@lru_cache(maxsize=1)
def load_tuning_config() -> dict[str, Any]:
    if not ADAPTIVE_TUNING_JSON.exists():
        return {}
    try:
        payload = json.loads(ADAPTIVE_TUNING_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def load_tuning_status() -> dict[str, Any]:
    if not ADAPTIVE_TUNING_JSON.exists():
        return {
            "config_loaded": False,
            "used_safe_fallback": True,
            "fallback_reason": "missing_adaptive_tuning_json",
        }
    try:
        payload = json.loads(ADAPTIVE_TUNING_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "config_loaded": False,
            "used_safe_fallback": True,
            "fallback_reason": f"adaptive_tuning_parse_failed:{exc.__class__.__name__}",
        }
    if not isinstance(payload, dict):
        return {
            "config_loaded": False,
            "used_safe_fallback": True,
            "fallback_reason": "adaptive_tuning_not_dict",
        }
    return {
        "config_loaded": True,
        "used_safe_fallback": False,
        "fallback_reason": "",
    }
