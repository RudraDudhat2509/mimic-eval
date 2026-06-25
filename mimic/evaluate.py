from __future__ import annotations

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
