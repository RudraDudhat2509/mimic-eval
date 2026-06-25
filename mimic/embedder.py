from __future__ import annotations

import numpy as np

_MODEL_CACHE: dict = {}


class Embedder:
    """Warm, batched, version-pinned wrapper over sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    @property
    def version(self) -> str:
        return f"sentence-transformers/{self.model_name}"

    def _model(self):
        if self.model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer  # lazy: no torch at import
            _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
        return _MODEL_CACHE[self.model_name]

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        vecs = self._model().encode(texts, batch_size=64, normalize_embeddings=True)
        return np.round(np.asarray(vecs, dtype=float), 6)
