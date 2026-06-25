from tests.fakes import FakeEmbedder
from mimic.intents import discover_intents
from mimic.semantic import SemanticExtractor, CombinedExtractor
from mimic.extractor import Extractor


def _model(emb):
    sents = (["refund please"] * 6) + (["shipping status"] * 6) + (["password reset"] * 6)
    return discover_intents(sents, emb, k_range=(3, 3))


def test_cross_coupling_identical_fields_max_similarity_one():
    emb = FakeEmbedder()
    sx = SemanticExtractor(emb, _model(emb), fields=["a", "b"], intent_field="b")
    feats = {f.name: f.value for f in sx.extract({"a": "refund please", "b": "refund please"})}
    assert "xsim_max__a__b" in feats
    assert feats["xsim_max__a__b"] >= 0.99            # identical sentences
    assert all(f.cost_tier == "expensive" for f in sx.extract({"a": "x", "b": "y"}))


def test_intent_features_present_and_named():
    emb = FakeEmbedder()
    sx = SemanticExtractor(emb, _model(emb), fields=["a", "b"], intent_field="b")
    names = sx.feature_names()
    assert any(n.startswith("sim_intent__") for n in names)
    feats = {f.name: f.value for f in sx.extract({"a": "hi", "b": "refund please"})}
    assert any(k.startswith("sim_intent__") for k in feats)


def test_degenerate_inputs_default_zero_no_crash():
    emb = FakeEmbedder()
    sx = SemanticExtractor(emb, _model(emb), fields=["a", "b"], intent_field="b")
    feats = {f.name: f.value for f in sx.extract({"a": "", "b": ""})}
    assert feats["xsim_max__a__b"] == 0.0


def test_combined_extractor_concatenates_names():
    emb = FakeEmbedder()
    sx = SemanticExtractor(emb, _model(emb), fields=["a", "b"], intent_field="b")
    comb = CombinedExtractor(Extractor(), sx)
    names = comb.feature_names()
    assert "word_count" in names and any(n.startswith("xsim_") for n in names)
