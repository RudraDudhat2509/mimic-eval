from __future__ import annotations

import numpy as np

from mimic.segment import split


def feature_matrix(examples, extractor):
    names = extractor.feature_names()
    rows = []
    for ex in examples:
        feats = {f.name: float(f.value) for f in extractor.extract(ex.inputs)}
        rows.append([feats[n] for n in names])
    X = np.array(rows, dtype=float) if rows else np.zeros((0, len(names)))
    y = [ex.verdict for ex in examples]
    return X, y, names


def embedding_matrix(examples, embedder, fields, round_to: int = 6) -> np.ndarray:
    sentences: list[str] = []
    spans: list[tuple[int, int]] = []
    for ex in examples:
        start = len(sentences)
        for fld in fields:
            sentences += split(str(ex.inputs.get(fld, "")))
        spans.append((start, len(sentences)))

    encoded = embedder.encode(sentences) if sentences else np.zeros((0, 0))
    dim = encoded.shape[1] if encoded.shape[0] > 0 else 0

    rows = []
    for (s, e) in spans:
        if e > s:
            rows.append(np.round(encoded[s:e].mean(axis=0), round_to))
        else:
            rows.append(np.zeros(dim))
    return np.array(rows, dtype=float) if rows else np.zeros((0, dim))
