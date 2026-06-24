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
