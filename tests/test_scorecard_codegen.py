from mimic.types import Example
from mimic.extractor import Extractor
from mimic.matrix import feature_matrix
from mimic.models import SparseLinearModel


def _separable():
    exs = []
    for _ in range(12):
        exs.append(Example(id="s", inputs={"response": "one two"}, verdict="short"))
        exs.append(Example(id="l", inputs={"response": " ".join(["w"] * 25)}, verdict="long"))
    return exs


def test_scorecard_code_runs_and_matches_model():
    exs = _separable()
    X, y, names = feature_matrix(exs, Extractor())
    m = SparseLinearModel().fit(X, y, names=names)

    code = m.to_code()
    ns: dict = {}
    exec(code, ns)

    feats = {f.name: f.value for f in Extractor().extract({"response": "one two"})}
    category, score = ns["mimic_judge"](feats)
    assert category == m.predict(X[:1])[0]      # codegen agrees with the model
    assert isinstance(score, float)
