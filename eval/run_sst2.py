# eval/run_sst2.py
"""Benchmark: how close does a cheap model get to the labels on SST-2 sentiment?
Fits tree + sparse-linear (named features) and embedding-linear (raw embedding),
prints held-out agreement (kappa / macro-F1 / accuracy) next to fine-tuned SOTA.

Usage: python eval/run_sst2.py [--n 3000] [--seed 42]
Label source: gold (a --from-llm OpenRouter path is reserved, not implemented this slice).
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request

from sklearn.metrics import cohen_kappa_score, f1_score

from mimic.datasets import sst2_rows_to_examples
from mimic.extractor import Extractor
from mimic.embedder import Embedder
from mimic.segment import split
from mimic.intents import discover_intents
from mimic.semantic import SemanticExtractor, CombinedExtractor
from mimic.matrix import feature_matrix, embedding_matrix
from mimic.models import TreeModel, SparseLinearModel, EmbeddingLinearModel

DATASET = "stanfordnlp/sst2"


def _load_rows(n: int) -> list[dict]:
    rows, offset = [], 0
    while len(rows) < n:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
               f"&config=default&split=train&offset={offset}&length=100")
        data = json.loads(urllib.request.urlopen(url).read().decode("utf-8"))
        batch = data.get("rows", [])
        if not batch:
            break
        for item in batch:
            r = item["row"]
            rows.append({"sentence": r["sentence"], "label": r["label"]})
            if len(rows) >= n:
                break
        offset += 100
    return rows


def _agreement(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred), key=str)
    code = {c: i for i, c in enumerate(classes)}
    yt = [code[v] for v in y_true]
    yp = [code[v] for v in y_pred]
    kappa = cohen_kappa_score(yt, yp) if len(set(yt)) > 1 else 0.0
    macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    acc = sum(a == b for a, b in zip(yt, yp)) / len(yt) if yt else 0.0
    return kappa, macro_f1, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Downloading SST-2 ({DATASET}, first {args.n} rows) ...")
    rows = _load_rows(args.n)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rsplit = int(len(rows) * 0.8)
    train = sst2_rows_to_examples(rows[:rsplit])
    test = sst2_rows_to_examples(rows[rsplit:])
    y_test = [ex.verdict for ex in test]
    print(f"{len(rows)} sentences  |  train {len(train)}  test {len(test)}  (gold labels)")

    # named-feature matrices (lexical + semantic), shared by tree and sparse-linear
    embedder = Embedder()
    sents = []
    for ex in train:
        sents += split(ex.inputs["sentence"])
    intents = discover_intents(sents, embedder, k_range=(3, 8), seed=args.seed)
    sem = SemanticExtractor(embedder, intents, fields=["sentence"], intent_field="sentence")
    comb = CombinedExtractor(Extractor(), sem)
    # NOTE: SST-2 inputs use the "sentence" field, but Extractor's lexical features read
    # "response"/"context", so the 5 lexical features are constant-zero here — only the
    # semantic intent features carry signal for the named-feature models.
    Xtr, ytr, names = feature_matrix(train, comb)
    Xte, _, _ = feature_matrix(test, comb)

    # raw-embedding matrices for the accuracy-ceiling model
    Etr = embedding_matrix(train, embedder, fields=["sentence"])
    Ete = embedding_matrix(test, embedder, fields=["sentence"])

    results = {
        "tree (named feats)": TreeModel().fit(Xtr, ytr).predict(Xte),
        "sparse-linear (named)": SparseLinearModel().fit(Xtr, ytr, names=names).predict(Xte),
        "embedding-linear (raw)": EmbeddingLinearModel().fit(Etr, ytr).predict(Ete),
    }

    print("\nHeld-out agreement with gold labels:")
    print(f"  {'model':<24} {'kappa':>7} {'macroF1':>8} {'acc':>7}")
    for name, preds in results.items():
        k, f1, acc = _agreement(y_test, preds)
        print(f"  {name:<24} {k:>7.2f} {f1:>8.2f} {acc:>7.2f}")
    print("\n  reference: fine-tuned BERT/RoBERTa SOTA ~0.93-0.96 accuracy")
    print("  (gap = cost of going free + instant; sparse-linear is the readable sweet spot)")


if __name__ == "__main__":
    main()
