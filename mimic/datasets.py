from __future__ import annotations

import json

from mimic.types import Example


def parse_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def halueval_records_to_examples(records: list[dict]) -> list[Example]:
    """Each HaluEval QA record yields two grounding examples:
    (knowledge, right_answer) -> grounded True; (knowledge, hallucinated_answer) -> False.
    """
    examples: list[Example] = []
    for i, rec in enumerate(records):
        ctx = rec["knowledge"]
        examples.append(Example(id=f"{i}-r",
                                inputs={"context": ctx, "response": rec["right_answer"]},
                                verdict=True, source="production"))
        examples.append(Example(id=f"{i}-h",
                                inputs={"context": ctx, "response": rec["hallucinated_answer"]},
                                verdict=False, source="production"))
    return examples
