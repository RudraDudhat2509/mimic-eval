from __future__ import annotations

import hashlib

import numpy as np


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests. Same text -> same vector."""
    version = "fake-v1"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()      # 32 bytes
            v = np.frombuffer(digest, dtype=np.float32).astype(float)  # 8 floats
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            norm = float(np.linalg.norm(v)) or 1.0
            out.append(np.round(v / norm, 6))
        return np.array(out, dtype=float)
