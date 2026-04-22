from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from src.config import HF_CACHE_DIR
from src.utils.hf_cache import resolve_local_hf_snapshot


@dataclass
class CrossEncoderReranker:
    model_name: str

    _CACHE: ClassVar[dict[str, object]] = {}

    def __post_init__(self) -> None:
        return

    @staticmethod
    def _resolve_cross_encoder():
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder
        except Exception:  # pragma: no cover
            return None

    def _model(self):
        cross_encoder_cls = self._resolve_cross_encoder()
        if cross_encoder_cls is None:
            return None
        if self.model_name not in self._CACHE:
            local_snapshot = resolve_local_hf_snapshot(self.model_name)
            try:
                self._CACHE[self.model_name] = cross_encoder_cls(
                    local_snapshot or self.model_name,
                    cache_folder=str(HF_CACHE_DIR),
                    local_files_only=True,
                )
            except Exception:
                self._CACHE[self.model_name] = cross_encoder_cls(self.model_name, cache_folder=str(HF_CACHE_DIR))
        return self._CACHE[self.model_name]

    def score(self, query: str, candidates: list[str]) -> np.ndarray:
        model = self._model()
        if model is None or not candidates:
            return np.zeros(len(candidates), dtype=np.float32)
        pairs = [[query, candidate] for candidate in candidates]
        raw = np.asarray(model.predict(pairs, show_progress_bar=False), dtype=np.float32)
        if raw.size == 0:
            return raw
        lo = float(raw.min())
        hi = float(raw.max())
        if abs(hi - lo) < 1e-12:
            return np.zeros_like(raw)
        return ((raw - lo) / (hi - lo)).astype(np.float32)
