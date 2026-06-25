from mimic.types import Rule, Example
from mimic.extractor import Extractor
from mimic.evaluate import evaluate_rules


def test_evaluate_rules_categorical_and_idempotent():
    rules = [
        Rule("word_count", "word_count <= 5.000", "short", "billing", 0.9, (0.85, 0.95), 10),
        Rule("word_count", "word_count > 20.000", "long", "sales", 0.9, (0.85, 0.95), 10),
    ]
    ex = Extractor()
    examples = [
        Example(id="1", inputs={"response": "two words"}, verdict="billing"),
        Example(id="2", inputs={"response": " ".join(["w"] * 30)}, verdict="sales"),
        Example(id="3", inputs={"response": " ".join(["w"] * 10)}, verdict="tech"),  # abstain
    ]
    r1 = evaluate_rules(rules, ex, examples)
    r2 = evaluate_rules(rules, ex, examples)
    assert r1 == r2                              # idempotent
    assert r1["n_total"] == 3
    assert r1["n_covered"] == 2                  # ex3 matches no rule
    assert r1["kappa"] == 1.0                    # both covered predicted correctly
