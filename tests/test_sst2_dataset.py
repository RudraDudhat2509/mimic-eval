from mimic.datasets import sst2_rows_to_examples


def test_sst2_rows_map_to_labeled_examples():
    rows = [{"sentence": "a delightful film", "label": 1},
            {"sentence": "a dull slog", "label": 0}]
    exs = sst2_rows_to_examples(rows)
    assert len(exs) == 2
    assert exs[0].inputs == {"sentence": "a delightful film"}
    assert exs[0].verdict == "positive"
    assert exs[1].verdict == "negative"
    assert exs[0].source == "production"
