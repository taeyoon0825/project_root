from __future__ import annotations

"""융합 검색에서 쓰는 CrossEncoder 리랭커 래퍼.

리랭커는 선택적으로만 켜질 수 있고 비용도 크기 때문에, 실제 모델 import와
로딩을 최대한 늦추고 결과는 다른 점수와 결합하기 쉬운 범위로 정규화한다.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from src.config import HF_CACHE_DIR
from src.utils.hf_cache import resolve_local_hf_snapshot


@dataclass
class CrossEncoderReranker:
    """SentenceTransformer CrossEncoder를 캐시와 함께 감싼 래퍼."""

    model_name: str

    _CACHE: ClassVar[dict[str, object]] = {}

    def __post_init__(self) -> None:
        """객체 생성 시에는 가볍게 두고 실제 모델 로딩은 첫 사용 시점으로 미룬다."""
        return

    @staticmethod
    def _resolve_cross_encoder():
        """의존성이 없는 환경에서도 검색이 죽지 않도록 지연 import를 사용한다."""
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder
        except Exception:  # pragma: no cover
            return None

    def _model(self):
        """처음 필요할 때만 CrossEncoder를 로딩하고 이후에는 재사용한다."""
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
        """리랭커 원점수를 0..1 범위로 정규화해 다른 신호와 결합 가능하게 만든다."""
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
