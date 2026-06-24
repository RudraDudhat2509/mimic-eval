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
