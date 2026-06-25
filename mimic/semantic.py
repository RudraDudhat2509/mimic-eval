from __future__ import annotations

import numpy as np

from mimic.segment import split
from mimic.types import Feature


def _round(x: float) -> float:
    return round(float(x), 6)


def _max_mean_cross(va: np.ndarray, vb: np.ndarray) -> tuple[float, float]:
    if va.shape[0] == 0 or vb.shape[0] == 0:
        return 0.0, 0.0
    sims = va @ vb.T                       # vectors are normalized by the embedder
    return _round(float(sims.max())), _round(float(sims.mean()))


class SemanticExtractor:
    def __init__(self, embedder, intent_model, fields: list[str], intent_field: str):
        self.embedder = embedder
        self.intents = intent_model
        self.fields = fields
        self.intent_field = intent_field

    def feature_names(self) -> list[str]:
        names = []
        for i, a in enumerate(self.fields):
            for b in self.fields[i + 1:]:
                names += [f"xsim_max__{a}__{b}", f"xsim_mean__{a}__{b}"]
        names += [f"sim_intent__{self.intents.slug(n)}" for n in self.intents.names]
        return names

    def _embed_field(self, inputs, field) -> np.ndarray:
        sents = split(str(inputs.get(field, "")))
        return self.embedder.encode(sents) if sents else np.zeros((0, 0))

    def extract(self, inputs, only=None) -> list[Feature]:
        out: dict[str, float] = {}
        embedded = {f: self._embed_field(inputs, f) for f in self.fields}
        for i, a in enumerate(self.fields):
            for b in self.fields[i + 1:]:
                mx, mn = _max_mean_cross(embedded[a], embedded[b])
                out[f"xsim_max__{a}__{b}"] = mx
                out[f"xsim_mean__{a}__{b}"] = mn

        # intent-similarity: max similarity of any sentence in intent_field to each centroid
        vb = embedded.get(self.intent_field, np.zeros((0, 0)))
        for ci, name in enumerate(self.intents.names):
            key = f"sim_intent__{self.intents.slug(name)}"
            if vb.shape[0] == 0:
                out[key] = 0.0
            else:
                c = self.intents.centroids[ci]
                nc = float(np.linalg.norm(c)) or 1.0
                sims = (vb @ c) / nc
                out[key] = _round(float(np.max(sims)))

        wanted = only if only is not None else self.feature_names()
        return [Feature(n, out[n], "semantic", "expensive") for n in wanted if n in out]


class CombinedExtractor:
    def __init__(self, lexical, semantic):
        self.lexical = lexical
        self.semantic = semantic

    def feature_names(self) -> list[str]:
        return self.lexical.feature_names() + self.semantic.feature_names()

    def extract(self, inputs, only=None) -> list[Feature]:
        feats = self.lexical.extract(inputs) + self.semantic.extract(inputs)
        if only is not None:
            wanted = set(only)
            feats = [f for f in feats if f.name in wanted]
        return feats
