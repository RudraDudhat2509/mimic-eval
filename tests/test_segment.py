from mimic.segment import split


def test_splits_on_sentence_punctuation():
    assert split("Hello world. How are you? I am fine!") == \
        ["Hello world.", "How are you?", "I am fine!"]


def test_no_boundary_returns_whole_as_one():
    assert split("just a fragment with no end") == ["just a fragment with no end"]


def test_empty_and_whitespace():
    assert split("") == []
    assert split("   \n  ") == []


def test_deterministic():
    t = "First. Second. Third."
    assert split(t) == split(t)
