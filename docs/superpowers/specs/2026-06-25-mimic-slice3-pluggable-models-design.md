# Mimic — Slice 3 Design Spec: Pluggable Models + Agreement Benchmark
**Date:** 2026-06-25
**Status:** Approved for implementation planning
**Builds on:** Slice 1 (core distillation) + Slice 2 (semantic + categorical), both shipped

---

## Why this slice exists

Mimic's promise: replace a low-entropy LLM *decision* with deterministic code that **behaves almost the same as the LLM** — target ≤5–10% drop. The LLM is the ceiling; the metric that matters is **agreement with the labels it produces**, not accuracy-vs-gold.

The blocker is the current model. Slice 1/2 distill with a single shallow **decision tree** over a handful of aggregate features — the lossiest point on the accuracy↔interpretability frontier (this is why PAWS landed at kappa 0.24). The research is clear (SetFit; ACL-2024 LLM-label distillation; EBM/InterpretML): a cheap **linear model over good features** routinely lands within a few points of fine-tuned transformers, and the bigger lever is the *features*, not the model family.

This slice makes the decision-maker **pluggable** and proves, with a side-by-side benchmark, which model best matches the labels on a real low-entropy task.

---

## The metric

**Agreement with the label source** = held-out **Cohen's kappa + per-class F1** between a model's predictions and the labels it was distilled from. The label source is pluggable:
- **v1: the dataset's gold labels** — free, no API, isolates model choice cleanly.
- **later: real LLM verdicts via OpenRouter** — the true "agreement with the LLM" number.

The benchmark compares **full-coverage predictions** across models (no abstention) so the model families are compared apples-to-apples. (Abstention/coverage is a separate product concern handled by the future cascade slice, not this comparison.)

---

## Scope

**In:**
- A pluggable model abstraction: **tree** (existing), **sparse linear** (L1 logistic regression over the named lexical+semantic features), and **embedding-linear** (logistic regression over the raw sentence embedding — the SetFit-style accuracy ceiling / black box).
- Codegen + feature attribution for the **sparse-linear** model (a deterministic weighted-scorecard artifact).
- A benchmark harness on **SST-2** (binary sentiment) reporting held-out kappa / per-class F1 across all three models, alongside a published-SOTA reference line.

**Out (deferred):**
- **EBM** (Explainable Boosting Machine) — strongest interpretable option, but adds the `interpret` dependency + binned-shape codegen. Fast-follow, designed to slot into the same abstraction.
- **Embedding-linear codegen** — its standalone artifact needs the runtime embedder; belongs with the runtime slice. Benchmark-only here.
- Cascade / shadow / middleware / CLI.

---

## Components

### `mimic/models.py` — the pluggable decision-makers

A small protocol so the engine and benchmark treat models uniformly. Each model fits on a feature (or embedding) matrix and predicts categories.

```
Protocol Distiller:
    fit(X: ndarray, y: list[category]) -> None
    predict(X: ndarray) -> list[category]
    attribution() -> human-readable summary
```

Concrete models:

- **`TreeModel`** — wraps the existing decision-tree path. `attribution()` = the plain-English rules already produced by the engine. (Refactor: the engine's tree-fitting is exposed so both the rule path and this benchmark use one code path — no duplication.)
- **`SparseLinearModel`** — scikit-learn `LogisticRegression(penalty="l1", solver="liblinear", multi_class="ovr")`. Multi-class native, fast, deterministic. `attribution()` = the **weighted scorecard**: each class's top + / − features by coefficient ("toxic is driven by: profanity_sim (+1.2), politeness_sim (−0.8)"). Codegen: a deterministic `mimic_judge` computing `score_c = Σ wᵢ·fᵢ + bᵢ` per class and returning `argmax`. Coefficients **rounded** for cross-machine stability.
- **`EmbeddingLinearModel`** — `LogisticRegression` over the raw sentence embedding of each example (built by mean-pooling the embedder's sentence vectors for the example's text). Black box (no named attribution — `attribution()` returns "embedding model: not feature-attributable"). The accuracy-ceiling reference for the benchmark. No standalone codegen this slice.

### Feature/embedding matrices (`mimic/matrix.py` or folded into the benchmark)

Two deterministic builders, reusing existing extractors:
- `feature_matrix(examples, extractor) -> (X, classes)` — the named lexical+semantic features (what tree + sparse-linear consume). This already exists inside the engine; extract it into a shared helper so models and benchmark share one builder.
- `embedding_matrix(examples, embedder, fields) -> X` — per-example embedding (mean of the example's sentence vectors across its text fields), rounded.

### Benchmark harness (`eval/run_sst2.py` + `mimic/datasets.py: sst2_rows_to_examples`)

- `sst2_rows_to_examples(rows)` (pure): SST-2 row (`sentence`, `label`) → `Example` with verdict `"positive"`/`"negative"`, `inputs={"sentence": ...}`, `source="production"`.
- `eval/run_sst2.py` (network only here): downloads SST-2, record-level train/test split, builds the feature matrix + embedding matrix, fits **tree / sparse-linear / embedding-linear**, and prints each model's **held-out kappa + per-class F1**, plus a reference line for published fine-tuned SOTA (~0.93–0.96 accuracy) so the gap is visible. Label source defaults to gold; a `--from-llm` flag is reserved for the future OpenRouter path (documented, not implemented this slice).

---

## Interpretability per model (what "feature attribution" means for each)

| Model | Attribution | Editable? | Codegen this slice |
|---|---|---|---|
| Tree | plain-English if/then rules | thresholds, by hand | yes (exists) |
| Sparse-linear | weighted scorecard (top ± features per class) | drop/zero a feature, nudge a weight | yes (new) |
| Embedding-linear | none (black box) | no | no (deferred) |

The sparse-linear scorecard is the "behaves like the LLM *and* tells you why" sweet spot the project is aiming for.

---

## Determinism

Fixed `random_state` on every model; L1 solver is deterministic; coefficients and embedding-matrix values **rounded to 6 decimals**; class ordering via `sorted(..., key=str)`. Same inputs → identical predictions and identical generated code.

---

## Testing

- **SparseLinearModel:** fits a separable 2-class synthetic set; `predict` recovers it; `attribution()` surfaces the driving feature; generated scorecard code `exec`s and returns the right category; coefficients rounded.
- **EmbeddingLinearModel:** fits on a synthetic embedding matrix (FakeEmbedder); `predict` works; `attribution()` reports black-box.
- **Matrix builders:** deterministic; feature matrix matches `extractor.feature_names()` ordering; embedding matrix shape + rounding.
- **`sst2_rows_to_examples`:** label 1 → "positive", 0 → "negative", single-field inputs, `source="production"`.
- **Determinism gate:** fit twice, identical coefficients + predictions.
- **Benchmark smoke-run:** real SST-2, prints three held-out kappas; sparse-linear and/or embedding-linear expected ≥ tree.

---

## Non-Goals (this slice)

- EBM (fast-follow, same abstraction).
- Embedding-linear standalone codegen (needs runtime embedder).
- Real LLM-label distillation (flag reserved; gold labels this slice).
- Cascade / shadow / middleware / CLI / many-class tasks (Banking77/CLINC150 need a deeper model).

---

## Success criteria

- The engine fits **tree, sparse-linear, and embedding-linear** through one uniform interface.
- The SST-2 benchmark prints held-out **kappa + per-class F1** for all three next to a published-SOTA reference, so the gap to the ceiling is a number, not a claim.
- On SST-2, **sparse-linear and/or embedding-linear beat the tree** and land within a clearly-stated margin of fine-tuned SOTA — the evidence for "a cheap model behaves almost like the big one."
- The sparse-linear model ships a **deterministic, editable weighted-scorecard artifact** with readable feature attribution.
- Everything deterministic and reproducible; unit tests use the fake embedder (no model download).
