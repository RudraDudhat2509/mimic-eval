from mimic.types import Example
from mimic.extractor import Extractor
from mimic.matrix import feature_matrix, embedding_matrix
from mimic.models import TreeModel, SparseLinearModel, EmbeddingLinearModel
from tests.fakes import FakeEmbedder


def _separable():
    exs = []
    for _ in range(12):
        exs.append(Example(id="s", inputs={"response": "one two"}, verdict="short"))
        exs.append(Example(id="l", inputs={"response": " ".join(["w"] * 25)}, verdict="long"))
    return exs


def test_sparse_linear_fits_predicts_and_attributes():
    exs = _separable()
    X, y, names = feature_matrix(exs, Extractor())
    m = SparseLinearModel().fit(X, y, names=names)
    preds = m.predict(X)
    acc = sum(p == t for p, t in zip(preds, y)) / len(y)
    assert acc > 0.9
    attr = m.attribution()
    assert isinstance(attr, dict) and attr           # non-empty per-class attribution


def test_tree_and_embedding_models_predict():
    exs = _separable()
    X, y, names = feature_matrix(exs, Extractor())
    tpreds = TreeModel().fit(X, y).predict(X)
    assert set(tpreds) <= {"short", "long"}

    Xe = embedding_matrix(exs, FakeEmbedder(), fields=["response"])
    em = EmbeddingLinearModel().fit(Xe, y)
    assert set(em.predict(Xe)) <= {"short", "long"}
    assert "black box" in em.attribution().lower()


def test_models_are_deterministic():
    exs = _separable()
    X, y, names = feature_matrix(exs, Extractor())
    p1 = SparseLinearModel().fit(X, y, names=names).predict(X)
    p2 = SparseLinearModel().fit(X, y, names=names).predict(X)
    assert p1 == p2
