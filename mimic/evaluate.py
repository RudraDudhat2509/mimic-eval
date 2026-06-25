from __future__ import annotations

import re
from typing import Any, Callable

from sklearn.metrics import cohen_kappa_score, f1_score

from mimic.types import Artifact, Example


def load_judge(artifact: Artifact) -> Callable[..., tuple]:
    ns: dict = {}
    exec(artifact.content, ns)
    return ns["mimic_judge"]


def evaluate_artifact(artifact: Artifact, examples: list[Example]) -> dict[str, Any]:
    judge = load_judge(artifact)
    y_true: list[int] = []
    y_pred: list[int] = []
    for ex in examples:
        verdict, _conf, _feat = judge(**ex.inputs)
        if verdict is None:               # artifact abstained -> not covered
            continue
        y_true.append(1 if ex.verdict else 0)
        y_pred.append(1 if verdict else 0)

    n_total = len(examples)
    n_covered = len(y_pred)
    kappa = 0.0
    if n_covered and len(set(y_true)) > 1:
        kappa = float(cohen_kappa_score(y_true, y_pred))

    return {
        "n_total": n_total,
        "n_covered": n_covered,
        "coverage": n_covered / n_total if n_total else 0.0,
        "kappa": kappa,
        "per_class_f1": {
            "True": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) if y_pred else 0.0,
            "False": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)) if y_pred else 0.0,
        },
    }


def _pythonize(condition: str) -> str:
    """Turn feature names in condition into f[...] lookups.

    E.g., "word_count <= 5.000" -> "f['word_count'] <= 5.000"
    """
    def repl(m: re.Match) -> str:
        return f"f[{m.group(0)!r}]"
    return re.sub(r"[a-z_][a-z0-9_]*(?=\s*(<=|>=|==|!=|<|>))", repl, condition)


def _apply(rules, feats: dict):
    """Apply rules in order; first matching rule decides; else abstain (None)."""
    for r in rules:
        if r.condition == "always" or eval(_pythonize(r.condition), {"__builtins__": {}}, {"f": feats}):
            return r.verdict
    return None


def evaluate_rules(rules, extractor, examples) -> dict[str, Any]:
    """Evaluate rules against examples; measure coverage and category-aware agreement.

    Returns: n_total, n_covered, coverage, kappa, per_class_f1 (over covered subset).
    """
    y_true, y_pred = [], []
    for ex in examples:
        feats = {f.name: f.value for f in extractor.extract(ex.inputs)}
        pred = _apply(rules, feats)
        if pred is None:
            continue
        y_true.append(ex.verdict)
        y_pred.append(pred)

    classes = sorted(set(y_true) | set(y_pred), key=lambda v: str(v))
    code = {c: i for i, c in enumerate(classes)}
    yt = [code[v] for v in y_true]
    yp = [code[v] for v in y_pred]
    n_total, n_covered = len(examples), len(yt)
    kappa = 0.0
    if n_covered and len(set(yt)) > 1:
        kappa = float(cohen_kappa_score(yt, yp))
    labels = list(range(len(classes)))
    f1s = f1_score(yt, yp, labels=labels, average=None, zero_division=0) if yt else []
    return {
        "n_total": n_total,
        "n_covered": n_covered,
        "coverage": n_covered / n_total if n_total else 0.0,
        "kappa": kappa,
        "per_class_f1": {str(classes[i]): float(f1s[i]) for i in labels} if yt else {},
    }
