from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class _Coded:
    """Shared label encode/decode using sorted(str) class order."""

    def _encode(self, y) -> np.ndarray:
        self.classes_ = sorted(set(y), key=lambda v: str(v))
        self._code = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self._code[v] for v in y], dtype=int)

    def _decode(self, codes) -> list:
        return [self.classes_[int(c)] for c in codes]


class TreeModel(_Coded):
    def __init__(self, max_depth: int = 4, random_state: int = 42):
        self.clf = DecisionTreeClassifier(max_depth=max_depth,
                                          class_weight="balanced", random_state=random_state)

    def fit(self, X, y):
        self.clf.fit(X, self._encode(y))
        return self

    def predict(self, X) -> list:
        return self._decode(self.clf.predict(X))

    def attribution(self) -> str:
        return f"decision tree (depth {self.clf.get_depth()}) - see extracted rules"


class SparseLinearModel(_Coded):
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.clf = LogisticRegression(solver="liblinear",
                                      C=C, random_state=random_state)
        self.names = None

    def fit(self, X, y, names=None):
        self.names = names
        self.clf.fit(X, self._encode(y))
        return self

    def predict(self, X) -> list:
        return self._decode(self.clf.predict(X))

    def _rows(self):
        # coef_ is (1, nfeat) for binary, (nclass, nfeat) for multiclass OvR
        coefs = np.atleast_2d(self.clf.coef_)
        if coefs.shape[0] == 1:           # binary: row maps to the positive class
            return {self.classes_[1]: coefs[0]}
        return {self.classes_[i]: coefs[i] for i in range(coefs.shape[0])}

    def attribution(self, top: int = 3) -> dict:
        names = self.names or [f"f{i}" for i in range(np.atleast_2d(self.clf.coef_).shape[1])]
        out = {}
        for cls, w in self._rows().items():
            order = np.argsort(w)
            out[str(cls)] = {
                "drives_toward": [(names[i], round(float(w[i]), 4)) for i in order[::-1][:top]],
                "drives_away": [(names[i], round(float(w[i]), 4)) for i in order[:top]],
            }
        return out


class EmbeddingLinearModel(_Coded):
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.clf = LogisticRegression(max_iter=1000, C=C, random_state=random_state)

    def fit(self, X, y):
        self.clf.fit(X, self._encode(y))
        return self

    def predict(self, X) -> list:
        return self._decode(self.clf.predict(X))

    def attribution(self) -> str:
        return "embedding model: not feature-attributable (black box)"
