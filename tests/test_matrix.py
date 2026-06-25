import numpy as np
from mimic.types import Example
from mimic.extractor import Extractor
from mimic.matrix import feature_matrix, embedding_matrix
from tests.fakes import FakeEmbedder


def test_feature_matrix_shape_names_and_raw_labels():
    exs = [Example(id="1", inputs={"response": "hello world"}, verdict="a"),
           Example(id="2", inputs={"response": "x"}, verdict="b")]
    X, y, names = feature_matrix(exs, Extractor())
    assert names == Extractor().feature_names()
    assert X.shape == (2, len(names))
    assert y == ["a", "b"]


def test_embedding_matrix_deterministic_pooled_and_empty_safe():
    exs = [Example(id="1", inputs={"s": "hello world. bye now."}, verdict="a"),
           Example(id="2", inputs={"s": ""}, verdict="b")]
    X1 = embedding_matrix(exs, FakeEmbedder(), fields=["s"])
    X2 = embedding_matrix(exs, FakeEmbedder(), fields=["s"])
    assert np.array_equal(X1, X2)           # idempotent
    assert X1.shape[0] == 2
    assert np.allclose(X1[1], 0.0)          # empty field -> zero vector
    assert np.array_equal(X1, np.round(X1, 6))
