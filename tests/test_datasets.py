from mimic.datasets import parse_jsonl, halueval_records_to_examples


def test_parse_jsonl_skips_blank_lines():
    text = '{"a": 1}\n\n{"a": 2}\n'
    assert parse_jsonl(text) == [{"a": 1}, {"a": 2}]


def test_halueval_record_yields_two_labeled_examples():
    recs = [{"knowledge": "The sky is blue.", "question": "color?",
             "right_answer": "The sky is blue.", "hallucinated_answer": "The sky is green."}]
    exs = halueval_records_to_examples(recs)
    assert len(exs) == 2
    assert exs[0].verdict is True
    assert exs[0].inputs == {"context": "The sky is blue.", "response": "The sky is blue."}
    assert exs[0].source == "production"
    assert exs[1].verdict is False
    assert exs[1].inputs["response"] == "The sky is green."
