import json
from mimic.types import JudgeConfig, Example
from mimic.collector import Collector


def _fake_llm(prompt: str) -> str:
    return json.dumps([
        {"context": "returns within 30 days", "response": "returns within 30 days"},
        {"context": "free shipping over $50", "response": "shipping is $9.99"},
    ])


def test_generate_inputs_parses_json():
    cfg = JudgeConfig("j", "checks grounding", fn=lambda **k: True)
    examples = Collector(cfg, _fake_llm).generate_inputs(n=2)
    assert len(examples) == 2
    assert examples[0].inputs["response"] == "returns within 30 days"
    assert examples[0].source == "generated"
    assert examples[0].verdict is None
    assert examples[0].id != examples[1].id


def test_collect_verdicts_runs_judge():
    def judge_fn(context: str, response: str) -> bool:
        return response in context

    cfg = JudgeConfig("j", "checks grounding", fn=judge_fn)
    col = Collector(cfg, _fake_llm)
    examples = col.generate_inputs(n=2)
    scored = col.collect_verdicts(examples)
    assert scored[0].verdict is True       # response inside context
    assert scored[1].verdict is False
