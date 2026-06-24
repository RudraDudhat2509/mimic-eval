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
