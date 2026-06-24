from __future__ import annotations

from dataclasses import dataclass
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
