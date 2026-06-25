# Mimic

**Stop paying an LLM to judge the same things over and over.**

If you evaluate your AI's output with an LLM judge (Claude, GPT, etc.), you pay for every single verdict — forever — and you can never see *why* a verdict was given. Mimic watches your judge, learns the patterns that actually drive its decisions, and replaces it with cheap, deterministic code that you can read, edit, and run for free.

It never decides blind: Mimic always shadows the real judge first, proving it agrees on your own traffic before it ever takes over.

> **Status:** early. This repo is **Slice 1 of 3** — the core distillation engine (turn a judge + examples into a runnable rules artifact). The always-on middleware, shadow runtime, and CLI land in Slices 2 and 3. See [the design spec](docs/superpowers/specs/2026-06-23-mimic-design.md).

## The idea in one example

You have a judge that checks if a support answer is grounded in the docs:

```python
import mimic

@mimic.judge("checks if an answer is grounded in the retrieved context")
def grounding_judge(context: str, response: str) -> bool:
    return claude.complete(...) == "yes"
```

Mimic generates diverse test cases, runs them through your judge once, and learns rules like:

```
When the answer mentions the same things as the context (8 of 10 key terms)
  → GROUNDED   (94% confident, covers 41 cases)
```

Then it hands you a free, deterministic function that reproduces the judge — and tells you, honestly, how often it agrees and where it's weak.

## What's here now (Slice 1)

```python
from mimic import distill, JudgeConfig

artifact, rules, report = distill(config, llm)   # llm injected
print(report["kappa"])          # agreement with the judge, chance-corrected
exec(artifact.content, ns)      # ns["mimic_judge"](**inputs) -> (verdict, confidence, feature)
```

- `@mimic.judge` decorator + registry
- LLM-driven input generation + verdict collection
- Dependency-free lexical feature extraction (pruned per judge)
- Decision-tree distillation with **Wilson confidence intervals** and **Cohen's kappa** (never raw accuracy)
- Pruned, runnable Python artifact generation
- Holdout evaluation + a validation harness against real data

## Does it actually work? (validation on real data)

Tested against **HaluEval QA**, a public hallucination dataset. Each record becomes two grounding examples (a grounded answer and a hallucinated one); Mimic distills rules on a train split and is scored on a **record-level held-out split** — both halves of any record stay on the same side, so no context bleeds across the boundary.

On `--n 500`, seed 42:

| Metric | Held-out |
|---|---|
| Cohen's kappa | **0.96** |
| F1 (grounded / not) | 0.98 / 0.98 |
| Coverage | 92% |

Reproduce it:

```bash
python eval/run_halueval.py --n 500
```

**Read this before quoting the number.** HaluEval is lexically easy by construction — hallucinated answers inject novel tokens while grounded answers reuse the source passage, so Mimic's lexical features (`novel_word_ratio`, `entity_overlap_ratio`) separate the classes almost perfectly. The 0.96 reflects how tractable *this* dataset is — exactly the kind of judge where cheap rules suffice — not a general hallucination-detection claim. Harder, more semantic judges will score lower, which is precisely why the full design falls back to the real LLM on the cases rules can't handle (coming in Slice 2).

## Design principles

- **Honest metrics** — agreement is reported as kappa + per-class F1, not raw accuracy.
- **A rule ships only if its confidence interval's lower bound clears your threshold.**
- **Plain English first** — the tool reads like a person explaining, not a debugger. Raw numbers live behind a `--technical` flag.
- **You can edit everything** — Mimic proposes, you dispose.

## Develop

```bash
pip install -e ".[dev]"
pytest -q
```

Requires Python 3.11+.

## License

MIT
