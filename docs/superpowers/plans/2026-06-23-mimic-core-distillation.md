# Mimic — Core Distillation Implementation Plan (Slice 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core that turns an LLM judge plus a set of examples into a working, deterministic rules artifact you can call without the LLM.

**Architecture:** A `@mimic.judge` decorator registers a judge. A Collector generates diverse inputs via an injected LLM and collects the judge's verdicts. An Extractor computes cheap, dependency-free lexical features per input. A DistillationEngine fits a class-balanced decision tree, walks its root-to-leaf paths into human-readable Rules with Wilson confidence intervals, and reports agreement as Cohen's kappa. A Generator turns the kept rules into a pruned Python artifact that computes only the features it needs.

**Tech Stack:** Python 3.11+, pytest, scikit-learn, NumPy. The LLM client and judge function are injected (mocked in tests) — no network calls in this slice.

## Global Constraints

- Python 3.11+ (uses `tomllib`, `X | Y` type unions, `match`).
- No network calls in core modules — the LLM client is always injected.
- Heavy NLP deps (spaCy, sentence-transformers) are NOT used in this slice; features here are pure-Python lexical. Semantic/embedding features arrive in Slice 2.
- Agreement is reported as **Cohen's kappa + per-class F1**, never raw accuracy.
- A rule ships only if the **lower bound** of its confidence interval clears the judge's `threshold`.
- Generated artifacts compute **only** the features they use (`features_used`), never the full battery.
- Verdicts in this slice are boolean (`bool`). Score judges (int/float) are out of scope for Slice 1.
- Module package name is `mimic`; distribution name is `mimic-eval`.

---

### Task 1: Project scaffold + core types

**Files:**
- Create: `pyproject.toml`
- Create: `mimic/__init__.py`
- Create: `mimic/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: dataclasses `JudgeConfig`, `Example`, `Feature`, `Rule`, `Artifact` with the exact fields below; all importable from `mimic.types`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
from mimic.types import JudgeConfig, Example, Feature, Rule, Artifact


def test_example_holds_inputs_and_verdict():
    ex = Example(id="a1", inputs={"context": "c", "response": "r"},
                 verdict=True, source="generated")
    assert ex.inputs["response"] == "r"
    assert ex.verdict is True
    assert ex.source == "generated"


def test_rule_keeps_plain_english_and_interval():
    rule = Rule(
        feature="entity_overlap_ratio",
        condition="entity_overlap_ratio >= 0.8",
        plain_english="the answer mentions the same things as the context",
        verdict=True,
        confidence=0.91,
        confidence_interval=(0.84, 0.95),
        coverage=41,
    )
    assert rule.confidence_interval[0] == 0.84
    assert "same things" in rule.plain_english


def test_artifact_tracks_features_used_and_kappa():
    art = Artifact(type="code", content="def f(): ...",
                   features_used=["entity_overlap_ratio"],
                   coverage=0.85, kappa=0.79,
                   per_class_f1={"True": 0.92, "False": 0.81},
                   optimize="speed")
    assert art.features_used == ["entity_overlap_ratio"]
    assert art.kappa == 0.79
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mimic'`

- [ ] **Step 3: Write the scaffold and types**

```toml
# pyproject.toml
[project]
name = "mimic-eval"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["scikit-learn>=1.4", "numpy>=1.26"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["mimic*"]
```

```python
# mimic/__init__.py
from mimic.types import JudgeConfig, Example, Feature, Rule, Artifact

__all__ = ["JudgeConfig", "Example", "Feature", "Rule", "Artifact"]
```

```python
# mimic/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass
class JudgeConfig:
    name: str
    description: str
    fn: Callable[..., bool]
    optimize: Literal["speed", "accuracy", "interpretability"] = "speed"
    threshold: float = 0.85


@dataclass
class Example:
    id: str
    inputs: dict[str, Any]
    verdict: bool | None = None
    source: Literal["generated", "production"] = "generated"


@dataclass
class Feature:
    name: str
    value: float | bool | int
    category: Literal["linguistic", "semantic", "structural"]
    cost_tier: Literal["cheap", "expensive"] = "cheap"


@dataclass
class Rule:
    feature: str
    condition: str
    plain_english: str
    verdict: bool
    confidence: float
    confidence_interval: tuple[float, float]
    coverage: int


@dataclass
class Artifact:
    type: Literal["code", "model", "rules"]
    content: str | bytes
    features_used: list[str]
    coverage: float
    kappa: float
    per_class_f1: dict[str, float]
    optimize: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_types.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml mimic/__init__.py mimic/types.py tests/test_types.py
git commit -m "feat: project scaffold and core dataclasses"
```

---

### Task 2: Decorator + registry

**Files:**
- Create: `mimic/decorators.py`
- Modify: `mimic/__init__.py`
- Test: `tests/test_decorators.py`

**Interfaces:**
- Consumes: `JudgeConfig` from `mimic.types`.
- Produces: `judge(description, optimize="speed", threshold=0.85)` decorator; module-level `_registry: dict[str, JudgeConfig]`; `get_registry() -> dict[str, JudgeConfig]`; `clear_registry() -> None`. The decorated function keeps a `_mimic_config` attribute and is callable unchanged. A duplicate name raises `ValueError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decorators.py
import pytest
from mimic.decorators import judge, get_registry, clear_registry


def setup_function():
    clear_registry()


def test_decorator_registers_judge_and_stays_callable():
    @judge("checks grounding", optimize="speed", threshold=0.9)
    def grounding_judge(context: str, response: str) -> bool:
        return context in response

    assert grounding_judge("ab", "xaby") is True            # still callable
    cfg = get_registry()["grounding_judge"]
    assert cfg.description == "checks grounding"
    assert cfg.optimize == "speed"
    assert cfg.threshold == 0.9
    assert grounding_judge._mimic_config is cfg


def test_duplicate_name_raises():
    @judge("first")
    def dup() -> bool:
        return True

    with pytest.raises(ValueError, match="already registered"):
        @judge("second")
        def dup() -> bool:  # noqa: F811
            return False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_decorators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mimic.decorators'`

- [ ] **Step 3: Write the decorator**

```python
# mimic/decorators.py
from __future__ import annotations

import functools
from typing import Callable, Literal

from mimic.types import JudgeConfig

_registry: dict[str, JudgeConfig] = {}


def get_registry() -> dict[str, JudgeConfig]:
    return _registry


def clear_registry() -> None:
    _registry.clear()


def judge(
    description: str,
    optimize: Literal["speed", "accuracy", "interpretability"] = "speed",
    threshold: float = 0.85,
):
    def decorator(fn: Callable[..., bool]) -> Callable[..., bool]:
        name = fn.__name__
        if name in _registry:
            raise ValueError(f"judge {name!r} already registered")
        config = JudgeConfig(name, description, fn, optimize, threshold)
        _registry[name] = config

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper._mimic_config = config
        return wrapper

    return decorator
```

```python
# mimic/__init__.py
from mimic.types import JudgeConfig, Example, Feature, Rule, Artifact
from mimic.decorators import judge, get_registry, clear_registry

__all__ = [
    "JudgeConfig", "Example", "Feature", "Rule", "Artifact",
    "judge", "get_registry", "clear_registry",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_decorators.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add mimic/decorators.py mimic/__init__.py tests/test_decorators.py
git commit -m "feat: @mimic.judge decorator and registry"
```

---

### Task 3: Extractor — cheap lexical features

**Files:**
- Create: `mimic/extractor.py`
- Test: `tests/test_extractor.py`

**Interfaces:**
- Consumes: `Feature` from `mimic.types`.
- Produces: `Extractor` class with `extract(inputs: dict[str, Any], only: list[str] | None = None) -> list[Feature]` and `feature_names() -> list[str]`. Feature names produced: `word_count`, `negation_count`, `has_uncertainty`, `novel_word_ratio`, `entity_overlap_ratio`. When `only` is given, returns only those features (pruning). All features are `cost_tier="cheap"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extractor.py
from mimic.extractor import Extractor


def test_word_count_and_negation():
    ex = Extractor()
    feats = {f.name: f.value for f in ex.extract({"response": "this is not a test"})}
    assert feats["word_count"] == 5
    assert feats["negation_count"] == 1


def test_uncertainty_detected():
    ex = Extractor()
    feats = {f.name: f.value for f in ex.extract({"response": "I don't know honestly"})}
    assert feats["has_uncertainty"] is True


def test_overlap_and_novelty_use_context_and_response():
    ex = Extractor()
    inputs = {"context": "returns within 30 days", "response": "returns within 30 days always"}
    feats = {f.name: f.value for f in ex.extract(inputs)}
    # 4 of 5 response words appear in context -> novelty 1/5
    assert abs(feats["novel_word_ratio"] - 0.2) < 1e-9
    assert feats["entity_overlap_ratio"] == 1.0   # all context words present


def test_pruning_returns_only_requested():
    ex = Extractor()
    feats = ex.extract({"response": "hello world"}, only=["word_count"])
    assert [f.name for f in feats] == ["word_count"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mimic.extractor'`

- [ ] **Step 3: Write the extractor**

```python
# mimic/extractor.py
from __future__ import annotations

import re
from typing import Any

from mimic.types import Feature

_NEGATIONS = {"no", "not", "never", "none", "cannot", "n't"}
_UNCERTAIN = ["i don't know", "i do not know", "not sure", "unsure", "maybe", "i'm not sure"]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


class Extractor:
    _NAMES = ["word_count", "negation_count", "has_uncertainty",
              "novel_word_ratio", "entity_overlap_ratio"]

    def feature_names(self) -> list[str]:
        return list(self._NAMES)

    def extract(self, inputs: dict[str, Any], only: list[str] | None = None) -> list[Feature]:
        response = str(inputs.get("response", ""))
        context = str(inputs.get("context", ""))
        rwords = _words(response)
        cwords = set(_words(context))

        values: dict[str, float | bool | int] = {
            "word_count": len(rwords),
            "negation_count": sum(1 for w in rwords if w in _NEGATIONS),
            "has_uncertainty": any(p in response.lower() for p in _UNCERTAIN),
            "novel_word_ratio": (
                sum(1 for w in rwords if w not in cwords) / len(rwords) if rwords else 0.0
            ),
            "entity_overlap_ratio": (
                sum(1 for w in cwords if w in set(rwords)) / len(cwords) if cwords else 0.0
            ),
        }

        wanted = only if only is not None else self._NAMES
        cat = {"word_count": "structural", "negation_count": "linguistic",
               "has_uncertainty": "linguistic", "novel_word_ratio": "semantic",
               "entity_overlap_ratio": "semantic"}
        return [Feature(name, values[name], cat[name], "cheap")
                for name in wanted if name in values]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extractor.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add mimic/extractor.py tests/test_extractor.py
git commit -m "feat: cheap lexical feature extractor with pruning"
```

---

### Task 4: Distillation Engine — tree to rules with confidence intervals

**Files:**
- Create: `mimic/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `Example`, `Rule` from `mimic.types`; `Extractor` from `mimic.extractor`.
- Produces: `DistillationEngine(extractor, threshold)` with `fit(examples: list[Example]) -> tuple[list[Rule], dict]`. The dict reports `{"kappa": float, "per_class_f1": dict[str,float], "coverage": float, "features_used": list[str]}`. Only rules whose `confidence_interval[0] >= threshold` are returned. Helper `wilson_interval(successes: int, n: int) -> tuple[float, float]` is module-level and tested directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine.py
from mimic.types import Example
from mimic.extractor import Extractor
from mimic.engine import DistillationEngine, wilson_interval


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(9, 10)
    assert 0.0 <= lo <= 0.9 <= hi <= 1.0


def _separable_examples():
    # response fully inside context -> grounded True; lots of novel words -> False
    pos = [Example(id=f"p{i}", inputs={"context": "returns within thirty days policy",
                                       "response": "returns within thirty days"}, verdict=True)
           for i in range(15)]
    neg = [Example(id=f"n{i}", inputs={"context": "returns within thirty days policy",
                                       "response": "shipping costs nine dollars flat extra surcharge applies always"},
                   verdict=False) for i in range(15)]
    return pos + neg


def test_fit_learns_separable_rules():
    eng = DistillationEngine(Extractor(), threshold=0.7)
    rules, report = eng.fit(_separable_examples())
    assert len(rules) >= 1
    assert report["kappa"] > 0.8
    assert all(r.confidence_interval[0] >= 0.7 for r in rules)
    assert set(report["features_used"]).issubset(set(Extractor().feature_names()))


def test_fit_drops_low_confidence_rules():
    # random verdicts -> no rule should clear a high threshold
    exs = [Example(id=str(i), inputs={"context": "a b c", "response": "x y z"},
                   verdict=(i % 2 == 0)) for i in range(20)]
    eng = DistillationEngine(Extractor(), threshold=0.95)
    rules, _ = eng.fit(exs)
    assert rules == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mimic.engine'`

- [ ] **Step 3: Write the engine**

```python
# mimic/engine.py
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score, f1_score

from mimic.types import Example, Rule
from mimic.extractor import Extractor

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
    def __init__(self, extractor: Extractor, threshold: float = 0.85):
        self.extractor = extractor
        self.threshold = threshold

    def _matrix(self, examples: list[Example]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        names = self.extractor.feature_names()
        rows = []
        for ex in examples:
            feats = {f.name: float(f.value) for f in self.extractor.extract(ex.inputs)}
            rows.append([feats[n] for n in names])
        X = np.array(rows, dtype=float)
        y = np.array([1 if ex.verdict else 0 for ex in examples], dtype=int)
        return X, y, names

    def fit(self, examples: list[Example]) -> tuple[list[Rule], dict[str, Any]]:
        X, y, names = self._matrix(examples)
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3,
                                      class_weight="balanced", random_state=42)
        tree.fit(X, y)

        rules = self._paths_to_rules(tree, names, X, y)
        kept = [r for r in rules if r.confidence_interval[0] >= self.threshold]

        y_pred = tree.predict(X)
        report = {
            "kappa": float(cohen_kappa_score(y, y_pred)),
            "per_class_f1": {
                "True": float(f1_score(y, y_pred, pos_label=1, zero_division=0)),
                "False": float(f1_score(y, y_pred, pos_label=0, zero_division=0)),
            },
            "coverage": sum(r.coverage for r in kept) / len(examples) if examples else 0.0,
            "features_used": sorted({r.feature for r in kept}),
        }
        return kept, report

    def _paths_to_rules(self, tree, names, X, y) -> list[Rule]:
        t = tree.tree_
        rules: list[Rule] = []

        def recurse(node: int, conds: list[tuple[int, str, float]]):
            if t.children_left[node] == t.children_right[node]:   # leaf
                verdict = bool(int(t.value[node][0].argmax()))
                mask = np.ones(len(X), dtype=bool)
                for fi, op, thr in conds:
                    mask &= (X[:, fi] <= thr) if op == "<=" else (X[:, fi] > thr)
                covered = int(mask.sum())
                if covered == 0:
                    return
                correct = int(((y[mask] == 1) == verdict).sum())
                lo, hi = wilson_interval(correct, covered)
                main_fi = conds[-1][0] if conds else 0
                feat = names[main_fi]
                cond_str = " and ".join(f"{names[fi]} {op} {thr:.3f}" for fi, op, thr in conds)
                rules.append(Rule(
                    feature=feat, condition=cond_str or "always",
                    plain_english=_PLAIN.get(feat, feat),
                    verdict=verdict, confidence=correct / covered,
                    confidence_interval=(lo, hi), coverage=covered,
                ))
                return
            fi, thr = t.feature[node], t.threshold[node]
            recurse(t.children_left[node], conds + [(fi, "<=", thr)])
            recurse(t.children_right[node], conds + [(fi, ">", thr)])

        recurse(0, [])
        return rules
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add mimic/engine.py tests/test_engine.py
git commit -m "feat: distillation engine with Wilson CIs and kappa"
```

---

### Task 5: Artifact Generator — pruned Python code

**Files:**
- Create: `mimic/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: `Rule`, `Artifact` from `mimic.types`.
- Produces: `ArtifactGenerator` with `to_code(rules: list[Rule], report: dict, optimize: str) -> Artifact`. The artifact's `content` is Python source defining `def mimic_judge(**inputs):` returning `(verdict | None, confidence, feature)`. The code computes only the features the rules reference (pruned) by reusing `Extractor`. `Artifact.features_used` lists exactly those features.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator.py
from mimic.types import Rule
from mimic.generator import ArtifactGenerator


def _rule():
    return Rule(feature="entity_overlap_ratio",
                condition="entity_overlap_ratio > 0.800",
                plain_english="the answer mentions the same things as the context",
                verdict=True, confidence=0.94, confidence_interval=(0.88, 0.97),
                coverage=40)


def test_generated_code_runs_and_decides():
    art = ArtifactGenerator().to_code([_rule()],
            {"kappa": 0.8, "per_class_f1": {"True": 0.9, "False": 0.8}, "coverage": 0.85},
            optimize="speed")
    assert art.features_used == ["entity_overlap_ratio"]

    ns: dict = {}
    exec(art.content, ns)
    verdict, conf, feat = ns["mimic_judge"](context="returns within 30 days",
                                            response="returns within 30 days")
    assert verdict is True
    assert feat == "entity_overlap_ratio"


def test_no_match_returns_none():
    art = ArtifactGenerator().to_code([_rule()],
            {"kappa": 0.8, "per_class_f1": {}, "coverage": 0.85}, optimize="speed")
    ns: dict = {}
    exec(art.content, ns)
    verdict, conf, feat = ns["mimic_judge"](context="abc", response="totally different words here")
    assert verdict is None
    assert feat == "no_match"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mimic.generator'`

- [ ] **Step 3: Write the generator**

```python
# mimic/generator.py
from __future__ import annotations

import re

from mimic.types import Rule, Artifact

_HEADER = '''\
from mimic.extractor import Extractor

_EX = Extractor()


def mimic_judge(**inputs):
    """Generated by Mimic. Returns (verdict | None, confidence, feature)."""
    f = {feat.name: feat.value for feat in _EX.extract(inputs, only=%(used)r)}
'''


class ArtifactGenerator:
    def to_code(self, rules: list[Rule], report: dict, optimize: str) -> Artifact:
        used = sorted({r.feature for r in rules})
        lines = [_HEADER % {"used": used}]
        for r in rules:
            cond = self._pythonize(r.condition)
            lines.append(f"    # {r.plain_english}  (confidence {r.confidence:.2f})")
            lines.append(f"    if {cond}:")
            lines.append(f"        return {r.verdict}, {r.confidence:.2f}, {r.feature!r}")
        lines.append("    return None, 0.0, 'no_match'")
        content = "\n".join(lines) + "\n"
        return Artifact(type="code", content=content, features_used=used,
                        coverage=report.get("coverage", 0.0),
                        kappa=report.get("kappa", 0.0),
                        per_class_f1=report.get("per_class_f1", {}),
                        optimize=optimize)

    def _pythonize(self, condition: str) -> str:
        # turn "entity_overlap_ratio > 0.800 and word_count <= 5.000" into f[...] lookups
        def repl(m: re.Match) -> str:
            return f"f[{m.group(0)!r}]"
        return re.sub(r"[a-z_]+(?=\s*(<=|>))", repl, condition)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generator.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add mimic/generator.py tests/test_generator.py
git commit -m "feat: artifact generator emits pruned runnable code"
```

---

### Task 6: Collector — input generation + verdict collection

**Files:**
- Create: `mimic/collector.py`
- Test: `tests/test_collector.py`

**Interfaces:**
- Consumes: `Example`, `JudgeConfig` from `mimic.types`.
- Produces: `Collector(config, llm)` where `llm` is a callable `(prompt: str) -> str` returning a JSON array of input dicts. Methods: `generate_inputs(n: int) -> list[Example]` (parses the LLM's JSON into Examples with `source="generated"`, fresh ids) and `collect_verdicts(examples: list[Example]) -> list[Example]` (calls `config.fn(**ex.inputs)`, returns new Examples with verdict set). Both are synchronous in this slice; async batching is added in Slice 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collector.py
import json
from mimic.types import JudgeConfig, Example
from mimic.collector import Collector


def _fake_llm(prompt: str) -> str:
    return json.dumps([
        {"context": "returns within 30 days", "response": "returns within 30 days"},
        {"context": "free shipping over $50", "response": "shipping is $9.99"},
    ])


def test_generate_inputs_parses_json():
    cfg = JudgeConfig("j", "checks grounding", fn=lambda **k: True)
    examples = Collector(cfg, _fake_llm).generate_inputs(n=2)
    assert len(examples) == 2
    assert examples[0].inputs["response"] == "returns within 30 days"
    assert examples[0].source == "generated"
    assert examples[0].verdict is None
    assert examples[0].id != examples[1].id


def test_collect_verdicts_runs_judge():
    def judge_fn(context: str, response: str) -> bool:
        return response in context

    cfg = JudgeConfig("j", "checks grounding", fn=judge_fn)
    col = Collector(cfg, _fake_llm)
    examples = col.generate_inputs(n=2)
    scored = col.collect_verdicts(examples)
    assert scored[0].verdict is True       # response inside context
    assert scored[1].verdict is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mimic.collector'`

- [ ] **Step 3: Write the collector**

```python
# mimic/collector.py
from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import Callable

from mimic.types import Example, JudgeConfig

_PROMPT = """Generate {n} diverse test inputs for this judge.
Task: {description}

Split roughly equally across: clearly-positive, clearly-negative,
borderline, and edge cases (empty, very long, unusual format).
Return ONLY a JSON array of objects, each object an input dict."""


class Collector:
    def __init__(self, config: JudgeConfig, llm: Callable[[str], str]):
        self.config = config
        self.llm = llm

    def generate_inputs(self, n: int = 100) -> list[Example]:
        raw = self.llm(_PROMPT.format(n=n, description=self.config.description))
        items = json.loads(raw)
        return [Example(id=uuid.uuid4().hex[:8], inputs=item, source="generated")
                for item in items]

    def collect_verdicts(self, examples: list[Example]) -> list[Example]:
        return [replace(ex, verdict=bool(self.config.fn(**ex.inputs)))
                for ex in examples]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collector.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add mimic/collector.py tests/test_collector.py
git commit -m "feat: collector generates inputs and collects verdicts"
```

---

### Task 7: End-to-end distill pipeline

**Files:**
- Create: `mimic/pipeline.py`
- Modify: `mimic/__init__.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Collector`, `Extractor`, `DistillationEngine`, `ArtifactGenerator`, `JudgeConfig`.
- Produces: `distill(config: JudgeConfig, llm, n: int = 60) -> tuple[Artifact, list[Rule], dict]`, exported as `mimic.distill`. It generates inputs, collects verdicts, fits rules, and generates the artifact in `config.optimize` mode. Pure orchestration — no new logic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import json
from mimic.types import JudgeConfig
from mimic import distill


def _fake_llm(prompt: str) -> str:
    pos = [{"context": "returns within thirty days policy",
            "response": "returns within thirty days"} for _ in range(15)]
    neg = [{"context": "returns within thirty days policy",
            "response": "shipping costs nine dollars flat surcharge applies always extra"}
           for _ in range(15)]
    return json.dumps(pos + neg)


def test_distill_end_to_end_produces_runnable_artifact():
    def judge_fn(context: str, response: str) -> bool:
        return all(w in context for w in response.split())

    cfg = JudgeConfig("grounding_judge", "checks grounding", fn=judge_fn, threshold=0.7)
    artifact, rules, report = distill(cfg, _fake_llm, n=30)

    assert report["kappa"] > 0.8
    assert len(rules) >= 1

    ns: dict = {}
    exec(artifact.content, ns)
    verdict, _, _ = ns["mimic_judge"](context="returns within thirty days policy",
                                      response="returns within thirty days")
    assert verdict is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'distill'`

- [ ] **Step 3: Write the pipeline**

```python
# mimic/pipeline.py
from __future__ import annotations

from typing import Callable

from mimic.types import JudgeConfig, Artifact, Rule
from mimic.collector import Collector
from mimic.extractor import Extractor
from mimic.engine import DistillationEngine
from mimic.generator import ArtifactGenerator


def distill(config: JudgeConfig, llm: Callable[[str], str], n: int = 60
            ) -> tuple[Artifact, list[Rule], dict]:
    collector = Collector(config, llm)
    examples = collector.collect_verdicts(collector.generate_inputs(n))

    extractor = Extractor()
    rules, report = DistillationEngine(extractor, config.threshold).fit(examples)
    artifact = ArtifactGenerator().to_code(rules, report, config.optimize)
    return artifact, rules, report
```

```python
# mimic/__init__.py
from mimic.types import JudgeConfig, Example, Feature, Rule, Artifact
from mimic.decorators import judge, get_registry, clear_registry
from mimic.pipeline import distill

__all__ = [
    "JudgeConfig", "Example", "Feature", "Rule", "Artifact",
    "judge", "get_registry", "clear_registry", "distill",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ -v`
Expected: PASS (all tests green across the suite)

- [ ] **Step 5: Commit**

```bash
git add mimic/pipeline.py mimic/__init__.py tests/test_pipeline.py
git commit -m "feat: end-to-end distill pipeline"
```

---

## Validation Addendum — proving it on real data (Tasks 8–9)

Tasks 1–7 are validated on synthetic data. These two tasks validate Mimic against a **real, public, labeled dataset** (HaluEval QA) and report the honest **held-out** number — the evidence that the tool actually works, and the first thing the README should cite.

**Approach:** the *proxy test* — treat each dataset row's gold label as "the judge's verdict," distill rules on a train split, then measure agreement on a held-out test split. This needs no LLM and no network in the core modules (the download lives in a script under `eval/`). HaluEval QA is chosen because its schema is stable and simple: each record has `knowledge`, `question`, `right_answer`, `hallucinated_answer` — yielding one grounded example (`knowledge`, `right_answer` → True) and one hallucinated example (`knowledge`, `hallucinated_answer` → False). This maps directly onto Mimic's grounding features (entity overlap, novel-word ratio).

---

### Task 8: Holdout evaluation function

**Files:**
- Create: `mimic/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Artifact`, `Example` from `mimic.types`; the generated artifact's `mimic_judge(**inputs) -> (verdict | None, confidence, feature)`.
- Produces: `evaluate_artifact(artifact: Artifact, examples: list[Example]) -> dict` with keys `n_total`, `n_covered`, `coverage`, `kappa`, `per_class_f1`. It execs the artifact, runs every example, treats a `None` verdict as an abstention (not covered), and computes kappa + per-class F1 on the covered subset against each example's gold `verdict`. Also exposes `load_judge(artifact) -> Callable`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
from mimic.types import Rule, Example
from mimic.generator import ArtifactGenerator
from mimic.evaluate import evaluate_artifact


def _single_rule_artifact():
    rule = Rule(feature="entity_overlap_ratio",
                condition="entity_overlap_ratio > 0.800",
                plain_english="answer mentions the same things as the context",
                verdict=True, confidence=0.9, confidence_interval=(0.85, 0.95), coverage=10)
    return ArtifactGenerator().to_code(
        [rule], {"kappa": 0.9, "per_class_f1": {}, "coverage": 0.9}, optimize="speed")


def test_evaluate_reports_coverage_and_abstention():
    art = _single_rule_artifact()
    examples = [
        Example(id="1", inputs={"context": "returns within thirty days",
                                "response": "returns within thirty days"}, verdict=True),
        Example(id="2", inputs={"context": "returns within thirty days",
                                "response": "returns within thirty days"}, verdict=True),
        Example(id="3", inputs={"context": "free shipping over fifty",
                                "response": "totally unrelated novel words here friend"}, verdict=False),
    ]
    report = evaluate_artifact(art, examples)
    assert report["n_total"] == 3
    assert report["n_covered"] == 2          # ex3 has 0 overlap -> rule abstains
    assert report["coverage"] == 2 / 3
    assert report["per_class_f1"]["True"] == 1.0


def test_evaluate_kappa_with_both_classes():
    rules = [
        Rule("entity_overlap_ratio", "entity_overlap_ratio > 0.800", "overlap high",
             True, 0.9, (0.85, 0.95), 10),
        Rule("novel_word_ratio", "novel_word_ratio > 0.400", "many novel words",
             False, 0.9, (0.85, 0.95), 10),
    ]
    art = ArtifactGenerator().to_code(
        rules, {"kappa": 0.9, "per_class_f1": {}, "coverage": 0.9}, optimize="speed")
    examples = [
        Example(id="1", inputs={"context": "alpha beta gamma",
                                "response": "alpha beta gamma"}, verdict=True),
        Example(id="2", inputs={"context": "alpha beta gamma",
                                "response": "zzz qqq www novel unrelated words here now"}, verdict=False),
    ]
    report = evaluate_artifact(art, examples)
    assert report["n_covered"] == 2
    assert report["kappa"] == 1.0            # perfect agreement across both classes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mimic.evaluate'`

- [ ] **Step 3: Write the evaluator**

```python
# mimic/evaluate.py
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
    if n_covered and len(set(y_true)) > 1 and len(set(y_pred)) > 1:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add mimic/evaluate.py tests/test_evaluate.py
git commit -m "feat: holdout evaluation of artifact vs gold labels"
```

---

### Task 9: Real-dataset harness (HaluEval QA)

**Files:**
- Create: `mimic/datasets.py`
- Create: `eval/run_halueval.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `Example` from `mimic.types`; `Extractor`, `DistillationEngine`, `ArtifactGenerator`, `evaluate_artifact`.
- Produces: in `mimic/datasets.py` (pure, no network): `parse_jsonl(text: str) -> list[dict]` and `halueval_records_to_examples(records: list[dict]) -> list[Example]` (each record → two Examples, `source="production"`). In `eval/run_halueval.py`: a runnable script that downloads HaluEval QA, distills on a train split, and prints held-out kappa / F1 / coverage. The script is the ONLY place a network call is allowed (it is not imported by the `mimic` package).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_datasets.py
from mimic.datasets import parse_jsonl, halueval_records_to_examples


def test_parse_jsonl_skips_blank_lines():
    text = '{"a": 1}\n\n{"a": 2}\n'
    assert parse_jsonl(text) == [{"a": 1}, {"a": 2}]


def test_halueval_record_yields_two_labeled_examples():
    recs = [{"knowledge": "The sky is blue.", "question": "color?",
             "right_answer": "The sky is blue.", "hallucinated_answer": "The sky is green."}]
    exs = halueval_records_to_examples(recs)
    assert len(exs) == 2
    assert exs[0].verdict is True
    assert exs[0].inputs == {"context": "The sky is blue.", "response": "The sky is blue."}
    assert exs[0].source == "production"
    assert exs[1].verdict is False
    assert exs[1].inputs["response"] == "The sky is green."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mimic.datasets'`

- [ ] **Step 3: Write the dataset reshaper**

```python
# mimic/datasets.py
from __future__ import annotations

import json

from mimic.types import Example


def parse_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def halueval_records_to_examples(records: list[dict]) -> list[Example]:
    """Each HaluEval QA record yields two grounding examples:
    (knowledge, right_answer) -> grounded True; (knowledge, hallucinated_answer) -> False.
    """
    examples: list[Example] = []
    for i, rec in enumerate(records):
        ctx = rec["knowledge"]
        examples.append(Example(id=f"{i}-r",
                                inputs={"context": ctx, "response": rec["right_answer"]},
                                verdict=True, source="production"))
        examples.append(Example(id=f"{i}-h",
                                inputs={"context": ctx, "response": rec["hallucinated_answer"]},
                                verdict=False, source="production"))
    return examples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_datasets.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the harness script**

```python
# eval/run_halueval.py
"""Validate Mimic on real HaluEval QA data (proxy test: gold labels as the judge's verdicts).

Usage: python eval/run_halueval.py [--n 1000] [--threshold 0.6] [--seed 42]

Downloads HaluEval QA, builds (context, response) -> grounded examples, distills rules on
a train split, and reports kappa / F1 / coverage on a HELD-OUT test split (the honest number).
"""
from __future__ import annotations

import argparse
import random
import urllib.request

from mimic.datasets import parse_jsonl, halueval_records_to_examples
from mimic.extractor import Extractor
from mimic.engine import DistillationEngine
from mimic.generator import ArtifactGenerator
from mimic.evaluate import evaluate_artifact

URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="max records (each yields 2 examples)")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Downloading HaluEval QA from {URL} ...")
    text = urllib.request.urlopen(URL).read().decode("utf-8")
    records = parse_jsonl(text)[: args.n]
    examples = halueval_records_to_examples(records)

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = int(len(examples) * 0.8)
    train, test = examples[:split], examples[split:]
    print(f"{len(examples)} examples  |  train {len(train)}  test {len(test)}")

    rules, train_report = DistillationEngine(Extractor(), args.threshold).fit(train)
    artifact = ArtifactGenerator().to_code(rules, train_report, "speed")

    print(f"\nLearned {len(rules)} rules (train kappa {train_report['kappa']:.2f}):")
    for r in rules:
        verdict = "GROUNDED" if r.verdict else "NOT GROUNDED"
        print(f"  - {r.plain_english} -> {verdict}  ({r.confidence:.0%}, covers {r.coverage})")

    holdout = evaluate_artifact(artifact, test)
    print("\nHeld-out test results (the honest number):")
    print(f"  coverage: {holdout['coverage']:.0%}  ({holdout['n_covered']}/{holdout['n_total']})")
    print(f"  kappa:    {holdout['kappa']:.2f}")
    print(f"  F1:       grounded {holdout['per_class_f1']['True']:.2f} / "
          f"not-grounded {holdout['per_class_f1']['False']:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-run the harness (manual, needs network)**

Run: `python eval/run_halueval.py --n 500`
Expected: prints a learned rule list and a held-out block with a real `kappa` value (a positive number well above 0 indicates the lexical rules capture real grounding signal). This is a qualitative smoke check, not a unit test — if the download fails (offline), skip and note it.

- [ ] **Step 7: Commit**

```bash
git add mimic/datasets.py eval/run_halueval.py tests/test_datasets.py
git commit -m "feat: HaluEval validation harness with held-out reporting"
```

---

## Self-Review

**Spec coverage (Slice 1 scope):**
- Decorator + registry → Task 2 ✓
- Collector (input gen + verdicts) → Task 6 ✓
- Extractor with pruning + cost tiers → Task 3 ✓ (cheap tier; semantic/embedding deferred to Slice 2, per Global Constraints)
- Distillation Engine: class-balanced tree, CV-style confidence via Wilson intervals, kappa + per-class F1, threshold on lower bound → Task 4 ✓
- Artifact Generator: pruned runnable code, `features_used` → Task 5 ✓
- End-to-end orchestration → Task 7 ✓
- Holdout evaluation function (kappa/F1/coverage of artifact vs gold) → Task 8 ✓
- Real-dataset validation harness (HaluEval QA, held-out reporting) → Task 9 ✓
- Deferred to later slices (out of this plan, by design): middleware, intent, shadow, cascade runtime, logger, CLI, Output & Voice, score judges, embeddings. Tracked for Slice 2 and Slice 3.

**Placeholder scan:** No TBD/TODO; every code step shows full content; tests contain real assertions.

**Type consistency:** `Example(id, inputs, verdict, source)`, `Rule(...confidence_interval...)`, `Artifact(...features_used, kappa, per_class_f1...)` used identically across Tasks 1–7. `Extractor.extract(inputs, only=...)` signature matches its use in engine, generator, and the generated artifact. `distill(config, llm, n)` return triple matches its test. `JudgeConfig.fn` called as `fn(**inputs)` consistently.

---

## Execution Handoff

This is Slice 1 of 3. Tasks 1–7 are built, reviewed, and shipped. Tasks 8–9 (the validation addendum) are the next to execute — they prove the tool on real HaluEval QA data and produce the held-out kappa number for the README. Slices 2 (runtime: middleware, cascade, shadow, logger) and 3 (CLI + Output & Voice) get their own plans afterward.
