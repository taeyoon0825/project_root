from __future__ import annotations

"""밀집 인덱싱과 밀집 검색에서 공통으로 쓰는 임베딩 모델 래퍼 모음.

이 모듈은 모델 로딩 정책과 벡터 후처리 정책을 한곳에 모아둔다.
인덱싱 시점과 온라인 검색 시점이 같은 전처리 규칙, 같은 디바이스 선택,
같은 정규화 규칙을 공유하지 않으면 유사도 계산 의미가 어긋나기 때문이다.
"""

import os

import numpy as np

from src.config import EMBEDDING_MODELS, HF_CACHE_DIR, OPTIONAL_MODELS
from src.utils.device import resolve_torch_device
from src.utils.hf_cache import resolve_local_hf_snapshot

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

try:
    from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
except Exception:  # pragma: no cover
    hf_disable_progress_bars = None

transformers_logging.set_verbosity_error()
if hf_disable_progress_bars is not None:
    hf_disable_progress_bars()


MODEL_CATALOG = {**EMBEDDING_MODELS, **OPTIONAL_MODELS}


def _load_sentence_transformer(model_name: str, device: str) -> SentenceTransformer:
    """가능하면 로컬 스냅샷을 우선 사용해 반복 실행을 오프라인/재현 가능하게 유지한다.

    다만 최초 환경 구성에서는 캐시를 채워야 할 수 있으므로 원격 모델 ID로의
    fallback 경로도 남겨둔다.
    """

    local_snapshot = resolve_local_hf_snapshot(model_name)
    try:
        return SentenceTransformer(
            local_snapshot or model_name,
            cache_folder=str(HF_CACHE_DIR),
            device=device,
            local_files_only=True,
        )
    except Exception:
        return SentenceTransformer(
            model_name,
            cache_folder=str(HF_CACHE_DIR),
            device=device,
        )


def list_available_models(include_optional: bool = False) -> dict[str, str]:
    """호출부가 프로젝트 설정과 어긋나지 않도록 등록된 별칭만 노출한다."""

    if include_optional:
        return MODEL_CATALOG
    return EMBEDDING_MODELS


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화해 이후 내적이 코사인 유사도처럼 동작하게 만든다.

    뒤쪽 밀집 검색 로직은 단순 행렬 곱을 사용하므로, 문서 벡터와 질의 벡터가
    같은 정규화 규칙을 공유해야 의도한 유사도 해석이 가능하다.
    """

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


class EmbeddingModelWrapper:
    """모델별 인코딩 규칙을 숨기고 프로젝트용 안정 인터페이스만 제공한다.

    검색 코드가 내부 라이브러리 객체 종류나 특정 모델 계열의 접두어 규칙까지
    알 필요는 없다. 이 규칙을 여기로 모아야 인덱싱 시점과 질의 시점 인코딩이
    서로 대칭적으로 유지된다.
    """

    _MODEL_CACHE: dict[str, SentenceTransformer] = {}

    def __init__(self, model_alias: str):
        """설정된 모델 별칭을 실제 모델명으로 해석하고 기존 로딩 결과를 재사용한다."""

        if model_alias not in MODEL_CATALOG:
            raise ValueError(f"Unknown model alias: {model_alias}")
        self.model_alias = model_alias
        self.model_name = MODEL_CATALOG[model_alias]
        self.device = resolve_torch_device()
        if self.model_name not in self._MODEL_CACHE:
            self._MODEL_CACHE[self.model_name] = _load_sentence_transformer(self.model_name, self.device)
        self.model = self._MODEL_CACHE[self.model_name]

    def _prepare_texts(self, texts: list[str], is_query: bool) -> list[str]:
        """인코딩 전에 모델 계열별 접두어 규칙을 적용한다.

        E5 계열은 학습 시 query/passage 접두어를 전제로 하므로, 이 규칙을
        호출부가 아니라 래퍼가 소유해야 인덱스 생성과 검색이 항상 같은 규칙을 쓴다.
        """

        if "e5" in self.model_alias:
            prefix = "query: " if is_query else "passage: "
            return [prefix + text for text in texts]
        return texts

    def encode_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """문서 텍스트를 인덱스 생성에 맞는 배치 단위로 임베딩한다."""

        configured_batch = int(os.getenv("EMBED_BATCH_SIZE", str(batch_size)).strip() or batch_size)
        prepared = self._prepare_texts(texts, is_query=False)
        embeddings = self.model.encode(
            prepared,
            batch_size=max(1, configured_batch),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return normalize_embeddings(embeddings.astype(np.float32))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """질의 전용 전처리 규칙을 안전하게 분리하기 위해 질의 인코딩 경로를 따로 둔다."""

        prepared = self._prepare_texts(texts, is_query=True)
        embeddings = self.model.encode(
            prepared,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return normalize_embeddings(embeddings.astype(np.float32))
