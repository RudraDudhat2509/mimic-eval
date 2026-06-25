from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score, f1_score

from mimic.types import Example, Rule
from mimic.extractor import Extractor
from mimic.matrix import feature_matrix

_PLAIN = {
    "word_count": "the answer length",
    "negation_count": "how many negations the answer has",
    "has_uncertainty": "the answer admits uncertainty",
    "novel_word_ratio": "how many answer words are absent from the context",
    "entity_overlap_ratio": "the answer mentions the same things as the context",
}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class DistillationEngine:
    def __init__(self, extractor, threshold: float = 0.85):
        self.extractor = extractor
        self.threshold = threshold

    def _matrix(self, examples):
        X, y_raw, names = feature_matrix(examples, self.extractor)
        classes = sorted(set(y_raw), key=lambda v: str(v))
        code = {c: i for i, c in enumerate(classes)}
        y = np.array([code[v] for v in y_raw], dtype=int)
        return X, y, names, classes

    def fit(self, examples):
        X, y, names, classes = self._matrix(examples)
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3,
                                      class_weight="balanced", random_state=42)
        tree.fit(X, y)
        rules = self._paths_to_rules(tree, names, X, y, classes)
        kept = [r for r in rules if r.confidence_interval[0] >= self.threshold]

        y_pred = tree.predict(X)
        labels = list(range(len(classes)))
        f1s = f1_score(y, y_pred, labels=labels, average=None, zero_division=0)
        report = {
            "kappa": float(cohen_kappa_score(y, y_pred)),
            "per_class_f1": {str(classes[i]): float(f1s[i]) for i in labels},
            "coverage": sum(r.coverage for r in kept) / len(examples) if examples else 0.0,
            "features_used": sorted({r.feature for r in kept}),
        }
        return kept, report

    def _paths_to_rules(self, tree, names, X, y, classes):
        t = tree.tree_
        rules = []

        def recurse(node, conds):
            if t.children_left[node] == t.children_right[node]:
                code = int(t.value[node][0].argmax())
                verdict = classes[code]
                mask = np.ones(len(X), dtype=bool)
                for fi, op, thr in conds:
                    mask &= (X[:, fi] <= thr) if op == "<=" else (X[:, fi] > thr)
                covered = int(mask.sum())
                if covered == 0:
                    return
                correct = int((y[mask] == code).sum())
                lo, hi = wilson_interval(correct, covered)
                main = conds[-1][0] if conds else 0
                feat = names[main]
                cond_str = " and ".join(f"{names[fi]} {op} {thr:.3f}" for fi, op, thr in conds)
                rules.append(Rule(feature=feat, condition=cond_str or "always",
                                  plain_english=_PLAIN.get(feat, feat), verdict=verdict,
                                  confidence=correct / covered, confidence_interval=(lo, hi),
                                  coverage=covered))
                return
            fi, thr = t.feature[node], t.threshold[node]
            recurse(t.children_left[node], conds + [(fi, "<=", thr)])
            recurse(t.children_right[node], conds + [(fi, ">", thr)])

        recurse(0, [])
        return rules
