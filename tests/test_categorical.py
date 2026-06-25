from mimic.types import Example
from mimic.extractor import Extractor
from mimic.engine import DistillationEngine
from mimic.generator import ArtifactGenerator


def _three_class_examples():
    # word_count drives 3 routes: short->"billing", medium->"tech", long->"sales"
    def make(nwords, label):
        return Example(id=f"{label}{nwords}",
                       inputs={"response": " ".join(["word"] * nwords)}, verdict=label)
    exs = []
    for _ in range(8):
        exs += [make(2, "billing"), make(10, "tech"), make(30, "sales")]
    return exs


def test_engine_learns_multiclass_rules():
    eng = DistillationEngine(Extractor(), threshold=0.6)
    rules, report = eng.fit(_three_class_examples())
    verdicts = {r.verdict for r in rules}
    assert verdicts.issubset({"billing", "tech", "sales"})
    assert len(verdicts) >= 2
    assert report["kappa"] > 0.8
    assert set(report["per_class_f1"]) <= {"billing", "tech", "sales"}


def test_generator_emits_string_category_safely():
    from mimic.types import Rule
    rule = Rule(feature="word_count", condition="word_count <= 5.000",
                plain_english="short answer", verdict="billing",
                confidence=0.9, confidence_interval=(0.85, 0.95), coverage=10)
    art = ArtifactGenerator().to_code([rule],
            {"kappa": 0.9, "per_class_f1": {}, "coverage": 0.9}, optimize="speed")
    ns: dict = {}
    exec(art.content, ns)
    verdict, _conf, _feat = ns["mimic_judge"](response="hi there")
    assert verdict == "billing"
