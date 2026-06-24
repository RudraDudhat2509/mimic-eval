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
