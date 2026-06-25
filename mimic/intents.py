from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class IntentModel:
    names: list[str]
    centroids: np.ndarray

    @staticmethod
    def slug(name: str) -> str:
        return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")

    def assign(self, vector: np.ndarray) -> tuple[str, float]:
        sims = [_cosine(vector, c) for c in self.centroids]
        i = int(np.argmax(sims))
        return self.names[i], round(sims[i], 6)

    def save(self, path) -> None:
        data = {"names": list(self.names), "centroids": self.centroids.tolist()}
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "IntentModel":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(names=list(data["names"]),
                   centroids=np.array(data["centroids"], dtype=float))


def _best_k(X: np.ndarray, k_range: tuple[int, int], seed: int) -> int:
    lo, hi = k_range
    hi = min(hi, len(X) - 1)
    if hi <= lo:
        return max(2, min(lo, len(X) - 1))
    best_k, best_score = lo, -2.0
    for k in range(lo, hi + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def discover_intents(sentences: list[str], embedder, namer: Callable[[list[str]], str] | None = None,
                     k_range: tuple[int, int] = (3, 8), seed: int = 42) -> IntentModel:
    X = embedder.encode(sentences)
    k = _best_k(X, k_range, seed)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    centroids = np.round(km.cluster_centers_, 6)

    names = []
    for ci in range(k):
        members = [sentences[i] for i in range(len(sentences)) if km.labels_[i] == ci]
        names.append(namer(members) if namer and members else f"intent_{ci}")
    return IntentModel(names=names, centroids=centroids)
