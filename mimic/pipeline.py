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
