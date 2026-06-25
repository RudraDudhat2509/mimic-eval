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
