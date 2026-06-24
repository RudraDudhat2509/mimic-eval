import pytest

from mimic.extractor import Extractor


def test_word_count_and_negation():
    ex = Extractor()
    feats = {f.name: f.value for f in ex.extract({"response": "this is not a test"})}
    assert feats["word_count"] == 5
    assert feats["negation_count"] == 1


def test_uncertainty_detected():
    ex = Extractor()
    feats = {f.name: f.value for f in ex.extract({"response": "I don't know honestly"})}
    assert feats["has_uncertainty"] is True


def test_overlap_and_novelty_use_context_and_response():
    ex = Extractor()
    inputs = {"context": "returns within 30 days", "response": "returns within 30 days always"}
    feats = {f.name: f.value for f in ex.extract(inputs)}
    # 4 of 5 response words appear in context -> novelty 1/5
    assert abs(feats["novel_word_ratio"] - 0.2) < 1e-9
    assert feats["entity_overlap_ratio"] == 1.0   # all context words present


def test_pruning_returns_only_requested():
    ex = Extractor()
    feats = ex.extract({"response": "hello world"}, only=["word_count"])
    assert [f.name for f in feats] == ["word_count"]


def test_unknown_only_name_raises():
    ex = Extractor()
    with pytest.raises(ValueError, match="unknown feature"):
        ex.extract({"response": "hello"}, only=["word_count", "not_a_feature"])

