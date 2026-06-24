# Mimic — Design Spec
**Date:** 2026-06-23 (rev 2)
**Status:** Approved for implementation planning
**Package name:** `mimic-eval`

---

## Problem

LLM judges are the standard way to evaluate AI output quality. They are:
- **Expensive** — every eval call costs money, every run, forever
- **Non-deterministic** — the same input gives different scores across runs
- **Black boxes** — no way to audit why a specific verdict was given
- **Drift-prone** — the underlying model updates, scores shift silently

Teams either pay forever or manually build deterministic heuristics per judge — which is slow, inconsistent, and never reused.

## Solution

Mimic wraps your LLM judge in an **always-on middleware** that intercepts every eval call. The middleware captures the input's intent and features with lightweight local NLP, runs a three-layer cascade (rules → small model → LLM fallback), and captures the output. It **always shadows** — recording what each layer *would have* decided versus what the LLM actually decides — so the rules are continuously validated against real traffic, never trusted blind.

**Core innovation:** existing tools replace expensive LLM judges with cheaper LLM judges. Mimic replaces them with deterministic code or tiny models — the output format chosen dynamically based on what the use case needs — and proves the replacement is safe on the user's own live data before it ever decides a verdict.

**Market gap:** the pattern is practiced manually by advanced teams and exists in research papers, but no open-source tool automates the full pipeline. Snorkel AI is the closest conceptual predecessor but requires manual labeling function design and is enterprise/paid.

---

## Scope (v1)

**In:** LLM evaluation judges only — any function that takes text inputs and returns a verdict (bool, int, float).

**Out:** general LLM call replacement, model training platforms, cloud deployment, multi-modal inputs.

---

## UX Philosophy

> Mimic proposes. User disposes. Every step can be inspected, every output can be edited, nothing is permanent until confirmed.
>
> Mimic never decides blind. It always shadows the real judge before and while it serves.

Concretely:
- Every intermediate state (inputs, rules, artifact) is stored as human-readable YAML or Python on disk
- Every stage shows a preview and asks before proceeding
- Every output can be opened in an editor and modified (editor auto-detected — `$EDITOR` → VS Code → notepad on Windows)
- Warnings over errors — bad inputs get flagged, not crashed on
- `remimic` merges new findings with existing rules, never replaces blindly
- `mimic status` always shows current state of every judge
- **Shadow is mandatory and continuous** — you cannot skip it

---

## Output & Voice

The best tools read like a person explaining, not a debugger dumping. Every line Mimic prints to a human follows five rules:

1. **Plain sentence first, number as evidence.** The number lives in parentheses as proof, never as the subject. Not `entity_overlap_ratio = 0.81 (≥0.80)` but "the answer mentioned the same things as the doc (8 of 10 key terms)."
2. **No internal vocabulary by default.** `kappa`, `entity_overlap_ratio`, `confidence interval`, "shadow verdict," "deciding layer," "intent" — none of these surface in default output.
3. **Translate every metric into meaning.** `confidence 0.88` → "fairly sure." `kappa 0.82` → "agrees with the AI about 8 times in 10." `coverage 70%` → "handles 7 of every 10 checks on its own."
4. **Every output ends with what to do** — as a pasteable command.
5. **Disagreement is a story, not a symbol.** Never a bare `✗`. Explain what went wrong in one or two sentences.

**Two registers.** Default output is plain English for any reader. Every command also accepts `--technical` (alias `-v`), which exposes the raw metrics — `kappa`, feature names, thresholds, confidence intervals, agreement booleans — for power users and anyone auditing the repo. Plain by default, real numbers one flag away.

### Reference: `mimic why` (default vs `--technical`)

Default:
```
$ mimic why c4f1a9

This was a troubleshooting question.

  Mimic said:    GROUNDED  (answer matches the help doc)
  The real AI:   NOT GROUNDED
  → They disagreed.

Why Mimic said grounded:
  The answer mentioned the same things as the help doc
  (8 of the 10 key terms). That usually means it's grounded.

Why it got fooled this time:
  The answer used the right words but described the wrong steps.
  This shortcut only checks the words — not the meaning. That's
  its blind spot.

How sure was Mimic?  Fairly sure (88%). But this is exactly the
kind of case where word-matching slips up.

What you can do:
  Keep these on the real AI         →  mimic guard troubleshooting
  Teach Mimic from cases like this  →  mimic remimic --from-logs
```

Technical:
```
$ mimic why c4f1a9 --technical
  intent: troubleshooting
  entity_overlap_ratio = 0.81  (threshold ≥ 0.80)  → Rule 1 → GROUNDED
  deciding layer: rule   confidence: 0.88   CI [0.81–0.93]
  shadow LLM verdict: NOT GROUNDED   agreement: false
  kappa (7d, this intent): 0.71
```

### Reference: `status` and `explain` (default voice)

```
$ mimic status

grounding_judge  —  live and saving you money

  Handling on its own:  7 of 10 checks   (the rest go to the real AI)
  Matches the real AI:  about 8 times in 10
  Saving:               ~$74/month

  ⚠ Weak spot: troubleshooting questions. Those still go to the AI.
```

```
$ mimic explain

This week, Mimic handled 7 of every 10 grounding checks by itself
and matched the real AI about 8 times in 10. That saved you $14.80
after costs — it paid for itself on day 6.

Strong on policy and chit-chat questions; still leans on the real
AI for troubleshooting (on purpose — it's weak there).

  Want the raw numbers?  mimic explain --technical
```

All technical metrics defined elsewhere in this spec (kappa, per-class F1, confidence intervals, feature names) are **internal and `--technical`-only**. They never appear in default output.

---

## Architecture

### The Middleware (the backbone)

Every eval call — in training, shadow, or live serving — flows through one interceptor. This is the heart of the system.

```
        Eval call: judge(**inputs)
                  │
                  ▼
        ┌───────────────────────────────────────┐
        │           MIMIC MIDDLEWARE            │
        │  (always-on interceptor at API layer) │
        │                                       │
        │  1. Capture input                     │
        │  2. Lightweight NLP (one embedding):  │
        │       • intent classification         │
        │       • feature extraction (pruned)   │
        │  3. Cascade for a verdict             │
        │  4. ALWAYS shadow → record what each   │
        │     layer would say vs what LLM says  │
        │  5. Capture output (verdict)          │
        │  6. Log everything                    │
        └───────────────────────────────────────┘
                  │
                  ▼
             returns verdict
```

### The Cascade (inside the middleware)

```
Rule Layer      — pruned features + if/else, cheapest path
  ↓ (low confidence)
Model Layer     — tiny classifier, still local/free
  ↓ (low confidence)
LLM Fallback    — original judge
```

### Shadow — always on, two phases

Shadow is never optional. It runs in every phase; only *who decides* changes.

| Phase | Who decides the verdict | What shadow records | Risk |
|---|---|---|---|
| **observe** (mandatory start) | LLM always | what rules/model *would* have said on real traffic | zero — LLM still decides |
| **serve** (after user opts in) | rules/model when confident, else LLM | a sampled % of calls *also* hit the LLM to keep comparing | bounded by sample rate |

The user only flips `observe → serve` after seeing real agreement numbers (kappa, per-class F1) on their own traffic. Even in `serve`, a configurable shadow sample (default 10%) keeps validating, so drift is caught continuously — not just when someone runs `remimic`.

---

## Module Structure

```
mimic/
├── __init__.py        # exports @mimic.judge
├── decorators.py      # decorator + judge registry
├── middleware.py      # the interceptor — capture, route, shadow, log
├── intent.py          # lightweight intent classification (reuses embedding)
├── collector.py       # input generation (LLM) + production-log streaming
├── extractor.py       # feature extraction, tiered cheap→expensive, pruned per judge
├── engine.py          # distillation — decision tree → rules, CV + kappa
├── generator.py       # artifact producer (code / model / rule table)
├── runtime.py         # cascade routing logic
├── shadow.py          # shadow recording + agreement stats
├── logger.py          # SQLite write + read
├── explainer.py       # log aggregation, kappa/F1, intent segmentation, savings
└── cli.py             # Click CLI — run, remimic, logs, explain, edit, status, why
```

---

## Core Data Structures

```python
@dataclass
class Example:
    id: str
    inputs: dict[str, Any]
    verdict: bool | int | None
    intent: str | None
    source: Literal["generated", "production"]

@dataclass
class Feature:
    name: str
    value: float | bool | int
    category: Literal["linguistic", "semantic", "structural"]
    cost_tier: Literal["cheap", "expensive"]   # drives compute order at runtime

@dataclass
class Rule:
    feature: str
    condition: str               # human-readable: "entity_overlap_ratio >= 0.8"
    plain_english: str           # "When the answer mentions things from the source"
    verdict: bool | int
    confidence: float            # cross-validated mean
    confidence_interval: tuple[float, float]   # honest range, not a point
    coverage: int

@dataclass
class Artifact:
    type: Literal["code", "model", "rules"]
    content: str | bytes
    features_used: list[str]     # only these are computed at runtime
    coverage: float
    kappa: float                 # agreement with judge, chance-corrected
    per_class_f1: dict[str, float]
    optimize: str

@dataclass
class CallLog:
    id: str
    timestamp: datetime
    judge_name: str
    inputs: dict
    intent: str | None
    verdict: bool | int          # the verdict actually returned
    deciding_layer: Literal["rule", "model", "llm"]
    confidence: float
    feature_triggered: str | None
    # shadow fields — what each layer WOULD have said
    shadow_rule_verdict: bool | int | None
    shadow_model_verdict: bool | int | None
    shadow_llm_verdict: bool | int | None
    agreement: bool | None       # did the deciding layer match the LLM? (when both ran)
    feature_ms: int              # honest: time spent in NLP feature extraction
    logic_ms: int                # time spent in tree/model logic
    cost_usd: float
```

---

## Component Specs

### Decorator + Registry (`decorators.py`)

```python
@mimic.judge(
    "checks if a RAG response is grounded in retrieved context",
    optimize="speed"
)
def grounding_judge(context: str, response: str) -> bool:
    return claude.complete(...)
```

The decorator registers the judge and installs the **middleware** in place of the raw function. Judges are namespaced by module path; a true name collision errors loudly instead of silently overwriting.

---

### Middleware (`middleware.py`) — the interceptor

Wraps every judge call. One embedding is computed per input and **reused** for both intent and semantic features (the key efficiency win — never embed the same text twice).

```python
async def __call__(self, **inputs) -> bool | int:
    t0 = time.monotonic()

    # one embedding, reused downstream
    embedding = self.embedder.encode(self._primary_text(inputs))

    intent   = self.intent.classify(embedding)            # cheap, reuses embedding
    features = self.extractor.extract(inputs, embedding,  # only features the artifact needs
                                      only=self.artifact.features_used)
    feature_ms = elapsed(t0)

    verdict, layer, conf, feat = await self.cascade.evaluate(features, inputs)

    # ALWAYS shadow
    shadow = await self.shadow.maybe_compare(inputs, verdict, layer)

    self.logger.write(inputs, intent, verdict, layer, conf, feat,
                      shadow, feature_ms, logic_ms=elapsed(t0) - feature_ms)
    return verdict
```

---

### Intent (`intent.py`) — lightweight

- Reuses the embedding already computed by the middleware (no extra model call)
- v1: cluster bootstrap inputs in embedding space, auto-name clusters via one LLM call at setup, then classify live inputs by nearest cluster
- Cost at runtime: a vector distance comparison — sub-millisecond
- Value: lets `explain` segment accuracy by intent ("rules are 95% on factual lookups, 61% on multi-step reasoning"), and detects when production intent mix drifts from the bootstrap mix

---

### Extractor (`extractor.py`) — tiered + pruned

| Feature | Tier | Needs model? |
|---|---|---|
| `word_count`, `negation_count`, `has_uncertainty`, keyword hits | **cheap** | no (sub-ms) |
| `past_tense_count`, `entity_count` | cheap-ish | spaCy |
| `entity_overlap_ratio`, `semantic_similarity`, `novel_word_ratio` | **expensive** | embedding (reused) |

Two rules at runtime:
1. **Prune** — compute only the features the distilled artifact actually uses (`features_used`), never the full battery.
2. **Cheap-first** — evaluate rules resting on cheap features before computing any expensive feature; if a cheap rule resolves the call, skip the embedding entirely.

---

### Distillation Engine (`engine.py`) — honest metrics

```python
def fit(self, matrix: FeatureMatrix) -> list[Rule]:
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3,
                                  class_weight="balanced")   # handle imbalance
    # k-fold CV for confidence + interval, NOT a single tiny holdout
    rules = self._tree_to_rules_with_cv(tree, matrix, folds=5)
    return [r for r in rules if r.confidence_interval[0] >= self.threshold]
```

- **Class-weighted** so a lazy "always yes" rule can't ride the majority class
- **k-fold cross-validation** for confidence — each rule gets a mean and an interval; a rule only ships if the *lower* bound clears the threshold
- Reports **Cohen's kappa** and **per-class F1** as the agreement metric, never raw accuracy
- **Score judges (int/float):** regression tree + tolerance band — "within ±1 of the judge" counts as agreement; or bucketize to low/med/high when the user prefers

---

### Artifact Generator (`generator.py`)

| `optimize` | Output | Honest latency |
|---|---|---|
| `speed` | Python if/else over **pruned** features | cheap-feature path: <2ms; embedding path: +10–40ms |
| `accuracy` | Serialized LogisticRegression | +10ms |
| `interpretability` | Markdown rule table | for audit |

Generated code is commented with each rule's `plain_english` so it reads without context.

---

### Shadow (`shadow.py`)

```python
async def maybe_compare(self, inputs, verdict, layer) -> ShadowRecord | None:
    if self.phase == "observe":
        llm_verdict = await self.llm_judge(**inputs)   # LLM decides; we record rules
        return ShadowRecord(llm=llm_verdict, agreement=(verdict == llm_verdict))
    if self.phase == "serve" and random() < self.sample_rate:   # default 0.10
        llm_verdict = await self.llm_judge(**inputs)
        return ShadowRecord(llm=llm_verdict, agreement=(verdict == llm_verdict))
    return None
```

Continuous agreement stats feed `explain` and trigger a drift warning when agreement on the live sample drops below a configurable floor.

---

### Logger (`logger.py`)

SQLite — zero dependency, local. Schema stores the deciding verdict, all shadow verdicts, intent, agreement, and the split feature/logic timing.

```sql
CREATE TABLE calls (
    id, timestamp, judge_name, inputs_json,
    intent, verdict, deciding_layer, confidence, feature,
    shadow_rule_verdict, shadow_model_verdict, shadow_llm_verdict, agreement,
    feature_ms, logic_ms, cost_usd
);
CREATE INDEX idx_judge_layer ON calls(judge_name, deciding_layer);
CREATE INDEX idx_intent      ON calls(judge_name, intent);
CREATE INDEX idx_timestamp   ON calls(timestamp);
```

For high-throughput pipelines, writes are batched on a background queue so logging never blocks the eval path.

---

### Disk Layout (`.mimic/`)

```
.mimic/
├── mimic.toml                  # project defaults (optimize, threshold, shadow rate)
└── grounding_judge/
    ├── inputs.yaml             # editable — generated examples
    ├── rules.yaml              # editable — plain-English comment above each rule
    ├── artifact.py             # editable — generated code (pruned features)
    ├── config.yaml             # editable — phase, optimize, threshold
    ├── history/                # last N artifact versions for rollback
    └── calls.db                # SQLite log
```

---

## CLI

All command output follows the **Output & Voice** rules above — plain English by default, raw metrics behind `--technical`. The example blocks below are shown in the `--technical` register for precision; the default register reads like the `status`/`explain`/`why` examples in that section.

### `mimic run`
Discovers judges, generates inputs, collects verdicts, distills rules (CV + kappa), generates artifact. Interactive by default; `--yes` for non-interactive CI. New judges **start in `observe` phase** — shadow only, LLM decides.

### `mimic serve <judge>`
Flips a judge from `observe` to `serve` after the user reviews agreement numbers. Refuses to flip if kappa on real traffic is below the floor, with an override flag.

### `mimic remimic`
Re-samples from the LLM (active-learning: prioritizes near-threshold cases), shows per-rule diff, merges on confirmation. `--from-logs` pulls real production examples.

### `mimic why <call_id>`
Per-call debugging: shows intent, feature values, the exact rule/tree path, the deciding layer, and (if shadowed) what the LLM said.

```
$ mimic why abc123
grounding_judge · intent: factual-lookup
  entity_overlap_ratio = 0.83  (≥ 0.80) → GROUNDED
  deciding layer: rule   confidence: 0.91 [0.84–0.95]
  shadow LLM verdict: GROUNDED ✓ agreement
```

### `mimic explain`
Coverage, kappa + per-class F1, **intent-segmented accuracy**, honest latency split, and **net** savings (gross minus bootstrap + remimic cost) with a break-even line.

```
$ mimic explain
grounding_judge — last 7 days · phase: serve
  Agreement (kappa): 0.81   F1: grounded 0.93 / not-grounded 0.74
  Rule layer:   612 calls (61%)  feature 31ms + logic 2ms
  Model layer:  235 calls (24%)
  LLM (fallback+shadow): 153 calls (15%)  $0.46/day
  By intent:  factual-lookup 95% · summary 88% · multi-step reasoning 61% ⚠
  Net saved this month: $74  (after $15 maintenance) · broke even day 4
```

### `mimic status`
Per-judge phase, coverage, kappa, last sample, and a worth-it verdict for low-volume judges.

```
grounding_judge    serve    kappa 0.81  coverage 88%  net +$74/mo
helpfulness_judge  observe  kappa 0.55  ⚠ not safe to serve yet
tiny_judge         observe  30 evals/day → won't break even, keep on LLM
```

### `mimic edit <judge>` / `logs`
Edit any artifact in an auto-detected editor. `logs` filters per-call records including shadow disagreements.

---

## Technology Stack

| Component | Library | Why |
|---|---|---|
| Embedding (shared) | sentence-transformers `all-MiniLM-L6-v2` | one model serves intent + semantic features |
| Intent | embedding clustering + nearest-centroid | no extra model, sub-ms at runtime |
| NLP features | spaCy | local, cheap tier |
| Rule learning | scikit-learn `DecisionTreeClassifier` (class-weighted) | serializable to if/else |
| Score judges | scikit-learn regression tree + tolerance band | handles 0–10 judges |
| Small model | scikit-learn `LogisticRegression` | fast, no GPU |
| Storage | SQLite (`sqlite3` stdlib) | zero dependency |
| CLI | Click | standard |
| Async | `asyncio` | parallel verdict collection + non-blocking shadow |
| Config | TOML (`tomllib` stdlib) | human-readable |

**Model warmth:** embedder and spaCy load once into a warm process (library use) or a `mimic serve` daemon — never per-call cold loads.

---

## Non-Goals (v1)

- Not a new LLM judge framework (deepeval, RAGAS already exist)
- Not a model training platform
- Not cloud-dependent — everything runs locally after bootstrap
- No multi-modal inputs
- No GUI / web dashboard

---

## Success Criteria

- `pip install mimic-eval` + `@mimic.judge` + `mimic run` = a judge in **observe** within 10 minutes
- Shadow proves agreement on real traffic before any rule decides a verdict
- Reported latency is honest (feature ms and logic ms shown separately)
- Agreement reported as **kappa + per-class F1**, never raw accuracy
- `mimic status` tells low-volume users when Mimic is *not* worth it
- Every intermediate file editable by hand without breaking the tool
- Continuous shadow + `remimic` catch judge drift automatically
- Zero required cloud dependencies after bootstrap
