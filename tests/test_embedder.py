import numpy as np
from .fakes import FakeEmbedder


def test_fake_embedder_is_deterministic_and_rounded():
    emb = FakeEmbedder()
    a = emb.encode(["hello", "world"])
    b = emb.encode(["hello", "world"])
    assert np.array_equal(a, b)                 # idempotent
    assert a.shape[0] == 2
    assert np.array_equal(a, np.round(a, 6))    # already rounded


def test_fake_embedder_distinct_texts_differ():
    emb = FakeEmbedder()
    v = emb.encode(["alpha", "beta"])
    assert not np.array_equal(v[0], v[1])


def test_real_embedder_module_imports_without_torch():
    import mimic.embedder  # must not import sentence_transformers at module load
    assert hasattr(mimic.embedder, "Embedder")


def test_fake_embedder_empty_returns_empty_2d():
    out = FakeEmbedder().encode([])
    assert out.shape == (0, 0)
