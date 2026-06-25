# Mimic — Slice 2 Design Spec: Generic Semantic Features + Categorical Labels
**Date:** 2026-06-25
**Status:** Approved for implementation planning
**Builds on:** Slice 1 (core distillation, shipped on `main`)

---

## The USP this slice unlocks

Mimic's real purpose is **not** limited to eval judges:

> Mimic replaces any LLM call that is secretly a *decision* — with deterministic code that is nearly as good, far cheaper, and instant — wherever the task is deterministic enough to allow it.

Evals were the beachhead. Slice 1 proved the loop on boolean grounding judges using word-level (lexical) features. This slice generalizes Mimic toward its actual scope by adding two things:

1. **Meaning, not just words** — sentence-level semantic features so Mimic can replicate decisions that depend on *meaning*, not surface vocabulary.
2. **Categories, not just true/false** — so Mimic can replicate routing, classification, and guardrail decisions, not only binary judges.

### The criterion: distill decisions, not creations

- **Decisions** (distillable): low-entropy output — yes/no, pick-one category, a score, a small fixed structure — where the answer is a function of features in the input. Classifiers and gates wearing a generation costume.
- **Creations** (keep the LLM): high-entropy output — prose, summaries, code, conversation — where phrasing and novelty are the value.

One line: **low output entropy + answer driven by input features = distillable.**

### Use cases this generalization serves

Routing & triage ("which agent/tool handles this?", "in scope?", "escalate to human?"); guardrails & safety ("jailbreak?", "policy violation?", PII/content moderation); classification & tagging (sentiment, toxicity, topic, ticket category, lead qualification); RAG plumbing ("is this chunk relevant?", "needs human review?"); lightweight fixed-schema extraction; matching & dedup. All share one shape: an expensive, slow, non-repeatable model producing a tiny, repeatable answer.

**Where Mimic honestly does not apply:** open-ended generation, summarization, code, chat, anything reasoning-heavy or creative. Mimic should *say so* rather than pretend.

---

## Failure modes this slice fixes

Slice 1 judges by comparing words. Five ways that breaks, each fixed by sentence-level meaning:

1. **Same meaning, different words** — "money back within a month" vs. source "refunds processed in 30 days": correct, but near-zero word overlap, so lexical rules wrongly flag it. Meaning-vectors place them together.
2. **Different meaning, same words** (the dangerous one) — "returns are **not** covered within 30 days" reuses the source's vocabulary but says the opposite; lexical rules wrongly pass it. Meaning-vectors (and negation signal) catch the divergence.
3. **No sense of topic** — an answer about shipping to a refund question; words may overlap, but it answered the wrong thing. Named intents give the system a notion of *what the text is about*.
4. **One bad sentence diluted in a good answer** — bag-of-words averages over the whole response, hiding a single fabricated sentence. Per-sentence cross-coupling makes the bad one stand out.
5. **Brittleness to rewording** — lexical rules pinned to surface words rot when docs or phrasing change. Meaning is more stable than wording.

---

## Scope

**In:**
- Deterministic sentence segmentation
- Shared, warm, version-pinned embedder (batched encode)
- Setup-time intent discovery (cluster → freeze centroids → name once via LLM, editable)
- Runtime generic semantic features (cross-coupling + named intent-similarity), augmenting lexical features
- Categorical labels end to end (`bool | int | str`)
- Idempotency/determinism guarantees + tests

**Out (deferred to later slices):** trained small-model (`optimize="accuracy"`), cascade runtime, shadow/observe-serve, middleware interceptor, CLI, Output & Voice.

---

## Architecture

```
Bootstrap examples (inputs + categorical verdicts)
        │
        ▼
  Segmenter  ──► sentences per text field (deterministic, with fallback)
        │
        ▼
  Embedder   ──► batched sentence vectors (warm model, pinned, rounded)
        │
        ├─ setup once ─► Intent discovery: cluster → freeze centroids → LLM names (editable)
        │                       └─► .mimic/<name>/intents.yaml
        ▼
  SemanticExtractor (runtime) ─► cross-coupling features + named intent-similarity features
        │                         (augment Slice 1 lexical features; rounded for determinism)
        ▼
  DistillationEngine (existing, now multi-class) ─► rules with categorical verdicts
        ▼
  ArtifactGenerator (existing) ─► pruned code returning (category | None, confidence, feature)
```

Cheap lexical rules still evaluate first and short-circuit before any embedding is computed (the cost tier from the master design finally earns its keep).

---

## Components

### `mimic/segment.py` — deterministic sentence splitter
- Pure-Python, regex-based; no heavy dependency.
- Splits on sentence-final punctuation with common-abbreviation guards.
- **Fallback:** if a chunk yields no boundaries, return the whole chunk as a single sentence.
- Same input → same sentence list, always. Unit-tested for determinism and messy input (code blocks, lists, no punctuation, empty).

Interface: `split(text: str) -> list[str]`.

### `mimic/embedder.py` — shared meaning-model
- Wraps `sentence-transformers` `all-MiniLM-L6-v2`.
- **Loaded once and kept warm** (module-level singleton / lazy cache); never reloaded per call.
- `encode(texts: list[str]) -> np.ndarray` — **batched** in a single call.
- Output vectors **rounded to fixed precision (6 decimals)** so different machines cannot disagree on a borderline rule.
- Model name + version recorded so artifacts are tied to a known embedder.

Interface: `Embedder(model_name=...).encode(texts) -> np.ndarray`; `Embedder.version -> str`.

### `mimic/intents.py` — intent discovery (setup-time only)
- Input: all sentences across bootstrap examples.
- Embed (batched) → cluster with **KMeans, fixed `random_state`** → choose the number of intents **deterministically** (silhouette score over a fixed candidate range, e.g. 3–12; pick the best, deterministic tie-break).
- **Freeze centroids** to `.mimic/<name>/intents.yaml`.
- Ask an LLM to **name** each cluster **once** (offline, from representative sentences). Names are written to the same file and are **editable**. The LLM only names; it never runs at prediction time.
- Idempotent given fixed sentences + seed.

Interface: `discover_intents(sentences, embedder, llm=None, k_range=(3,12)) -> IntentModel`; `IntentModel.save(path)`, `IntentModel.load(path)`, `IntentModel.assign(vector) -> (intent_name, similarity)`.

### `mimic/semantic.py` — runtime semantic features
For each input, batch-embed its sentences and compute (all values **rounded**):
- **Cross-coupling features** — for each ordered pair of text fields present, `max` and `mean` cosine similarity between their sentence sets. Generic "how semantically related is field A to field B" (covers grounding, relevance, dedup).
- **Named intent-similarity features** — for each frozen centroid, the maximum similarity any sentence in a chosen field reaches → feature `similarity_to_intent("<name>")`. Dynamic (discovered), named (readable), idempotent (frozen geometry).
- **Empty/degenerate inputs** have defined behavior: no sentences → similarity features default to `0.0`; never NaN, never divide-by-zero.

These are returned as `Feature` objects with `category="semantic"`, `cost_tier="expensive"`, and joined with Slice 1's lexical features for the engine. Pruning still applies — the generated artifact computes only the semantic features its rules reference, and only after cheap rules fail to resolve.

Interface: `SemanticExtractor(embedder, intent_model).extract(inputs, only=None) -> list[Feature]`.

### Categorical labels (engine, types, generator, metrics)
- `Example.verdict`, `Rule.verdict`, and the artifact's return generalize from `bool` to **`bool | int | str`** (a category).
- `DistillationEngine`: the decision tree is already multi-class; rule verdicts carry the category; metrics remain **Cohen's kappa + per-class F1**, now multi-class (per-class F1 keyed by category label).
- `ArtifactGenerator`: generated `mimic_judge(**inputs)` returns `(category | None, confidence, feature)`; the `verdict` literal in each branch is the category value (rendered with `repr`).
- Confidence intervals (Wilson) and the lower-bound threshold gate are unchanged — computed per rule against its own category.

---

## Idempotency — the production spine

Non-negotiable; an eval/replacement tool that wobbles between runs is worthless.

- Embedder model **version-pinned**; recorded in config and alongside the intent model.
- Intent centroids **frozen to disk**; runtime assignment is **nearest-centroid by cosine** — pure geometry.
- **No LLM call ever on the runtime/feature path** (LLM only names intents once, offline).
- All semantic feature values **rounded to fixed precision** (cross-machine stability).
- Segmentation deterministic with fallback.
- **Determinism test:** extract features twice on the same inputs and assert byte-identical output. This is a build gate.

---

## Disk layout additions

```
.mimic/<name>/
├── intents.yaml      # frozen centroids + editable intent names + embedder version
└── config.yaml       # records embedder model + version (added field)
```

`intents.yaml` is human-readable and editable: a user can rename an intent, and the change takes effect with no recompute (names are labels over fixed centroids).

---

## Cost & honesty

- Setup embeds every sentence of every bootstrap example — **batched into one pass**; estimated cost/time shown before running.
- Embedding time reported separately from rule-evaluation time (the master design's honest-latency rule).
- Cheap lexical rules short-circuit before any embedding is computed at runtime.

---

## Testing

- **Segmentation:** determinism; messy inputs (code, lists, no punctuation, empty) handled without crash.
- **Embedder:** batched encode shape; rounding applied; warm singleton reused (no reload).
- **Intents:** deterministic clustering given fixed seed + sentences; `assign` returns nearest centroid; save/load round-trips; editing a name in the file changes the feature label without changing geometry.
- **Semantic features:** cross-coupling and intent-similarity values correct on known small inputs; degenerate inputs default cleanly; pruning honored.
- **Categorical engine/artifact:** a 3-class synthetic task distills into multi-class rules; generated artifact returns the right category; kappa + per-class F1 reported per category.
- **Determinism gate:** double-extraction byte-identical.
- **Real-data smoke:** extend the harness to a categorical task (or a paraphrase-heavy grounding set) to show semantic features hold up where lexical features collapse.

---

## Non-Goals (this slice)

- No trained small-model / `optimize="accuracy"` artifact.
- No cascade, shadow, middleware, or runtime serving.
- No CLI or Output & Voice.
- No multi-modal inputs; no generation tasks.

---

## Success criteria

- Mimic distills a **non-grounding, multi-category** decision (e.g., a small routing or toxicity task) into plain-English, categorical rules.
- On a **paraphrase- or contradiction-heavy** set where lexical features collapse, semantic features keep held-out kappa materially higher than Slice 1 lexical-only.
- Feature extraction is **provably idempotent** (double-extraction test passes).
- No LLM call occurs on the runtime/feature path.
- Rules stay plain-English and editable; intent names are human-editable with no recompute.
- Everything runs locally and free after the one-time setup embed.
