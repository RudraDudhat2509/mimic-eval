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
) -> Callable[[Callable[..., bool]], Callable[..., bool]]:
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
