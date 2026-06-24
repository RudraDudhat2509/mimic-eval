import json
from mimic.types import JudgeConfig
from mimic import distill


def _fake_llm(prompt: str) -> str:
    pos = [{"context": "returns within thirty days policy",
            "response": "returns within thirty days"} for _ in range(15)]
    neg = [{"context": "returns within thirty days policy",
            "response": "shipping costs nine dollars flat surcharge applies always extra"}
           for _ in range(15)]
    return json.dumps(pos + neg)


def test_distill_end_to_end_produces_runnable_artifact():
    def judge_fn(context: str, response: str) -> bool:
        return all(w in context for w in response.split())

    cfg = JudgeConfig("grounding_judge", "checks grounding", fn=judge_fn, threshold=0.7)
    artifact, rules, report = distill(cfg, _fake_llm, n=30)

    assert report["kappa"] > 0.8
    assert len(rules) >= 1

    ns: dict = {}
    exec(artifact.content, ns)
    verdict, _, _ = ns["mimic_judge"](context="returns within thirty days policy",
                                      response="returns within thirty days")
    assert verdict is True
