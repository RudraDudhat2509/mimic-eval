import numpy as np
from tests.fakes import FakeEmbedder
from mimic.intents import discover_intents, IntentModel


def _sentences():
    # three lexical families; FakeEmbedder maps identical strings to identical vectors
    base = (["refund please"] * 6) + (["shipping status"] * 6) + (["password reset"] * 6)
    return base


def test_discovery_is_deterministic_and_assigns_nearest():
    emb = FakeEmbedder()
    m1 = discover_intents(_sentences(), emb, k_range=(3, 3), seed=42)
    m2 = discover_intents(_sentences(), emb, k_range=(3, 3), seed=42)
    assert np.array_equal(m1.centroids, m2.centroids)       # idempotent setup
    name, sim = m1.assign(emb.encode(["refund please"])[0])
    assert name in m1.names
    assert sim > 0.99                                        # exact match is closest


def test_default_names_and_custom_namer():
    emb = FakeEmbedder()
    m = discover_intents(_sentences(), emb, k_range=(3, 3),
                         namer=lambda sents: "named_" + sents[0].split()[0])
    assert all(n.startswith("named_") for n in m.names)


def test_save_load_roundtrip(tmp_path):
    emb = FakeEmbedder()
    m = discover_intents(_sentences(), emb, k_range=(3, 3))
    p = tmp_path / "intents.yaml"
    m.save(p)
    loaded = IntentModel.load(p)
    assert loaded.names == m.names
    assert np.allclose(loaded.centroids, m.centroids)
    assert IntentModel.slug("Refund Policy!") == "refund_policy"
