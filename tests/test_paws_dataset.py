from mimic.datasets import paws_rows_to_examples


def test_paws_row_maps_to_two_field_example():
    rows = [{"sentence1": "A cat sat.", "sentence2": "A feline rested.", "label": 1},
            {"sentence1": "He likes tea.", "sentence2": "He hates tea.", "label": 0}]
    exs = paws_rows_to_examples(rows)
    assert len(exs) == 2
    assert exs[0].inputs == {"sentence1": "A cat sat.", "sentence2": "A feline rested."}
    assert exs[0].verdict == "paraphrase"
    assert exs[1].verdict == "not_paraphrase"
    assert exs[0].source == "production"
