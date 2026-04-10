from __future__ import annotations

import os

import numpy as np

from src.config import EMBEDDING_MODELS, HF_CACHE_DIR, OPTIONAL_MODELS

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer


MODEL_CATALOG = {**EMBEDDING_MODELS, **OPTIONAL_MODELS}


def list_available_models(include_optional: bool = False) -> dict[str, str]:
    if include_optional:
        return MODEL_CATALOG
    return EMBEDDING_MODELS


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


class EmbeddingModelWrapper:
    _MODEL_CACHE: dict[str, SentenceTransformer] = {}

    def __init__(self, model_alias: str):
        if model_alias not in MODEL_CATALOG:
            raise ValueError(f"Unknown model alias: {model_alias}")
        self.model_alias = model_alias
        self.model_name = MODEL_CATALOG[model_alias]
        if self.model_name not in self._MODEL_CACHE:
            self._MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name, cache_folder=str(HF_CACHE_DIR))
        self.model = self._MODEL_CACHE[self.model_name]

    def _prepare_texts(self, texts: list[str], is_query: bool) -> list[str]:
        # E5 계열은 query/passsage 프롬프트 접두어가 중요하다.
        if "e5" in self.model_alias:
            prefix = "query: " if is_query else "passage: "
            return [prefix + text for text in texts]
        return texts

    def encode_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        prepared = self._prepare_texts(texts, is_query=False)
        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return normalize_embeddings(embeddings.astype(np.float32))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prepared = self._prepare_texts(texts, is_query=True)
        embeddings = self.model.encode(
            prepared,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return normalize_embeddings(embeddings.astype(np.float32))
