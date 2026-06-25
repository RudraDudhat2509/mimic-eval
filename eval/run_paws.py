"""Adversarial validation: does the kappa survive PAWS, where paraphrases use DIFFERENT
words and non-paraphrases share MANY words? Compares lexical-only vs lexical+semantic.

Usage: python eval/run_paws.py [--n 2000] [--threshold 0.6] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request

from mimic.datasets import paws_rows_to_examples
from mimic.extractor import Extractor
from mimic.embedder import Embedder
from mimic.segment import split
from mimic.intents import discover_intents
from mimic.semantic import SemanticExtractor, CombinedExtractor
from mimic.engine import DistillationEngine
from mimic.evaluate import evaluate_rules

# PAWS-Wiki labeled (final) via HuggingFace Datasets Server API (paginated JSON, no auth needed).
# Schema: id (int32), sentence1 (string), sentence2 (string), label (ClassLabel: 0=not, 1=para)
_HF_ROWS_URL = ("https://datasets-server.huggingface.co/rows"
                "?dataset=google-research-datasets%2Fpaws"
                "&config=labeled_final&split=train"
                "&offset={offset}&limit={limit}")
_PAGE = 100   # max allowed per HF datasets-server request


def _load_rows(n: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < n:
        limit = min(_PAGE, n - len(rows))
        url = _HF_ROWS_URL.format(offset=offset, limit=limit)
        resp = urllib.request.urlopen(url)
        payload = json.loads(resp.read().decode("utf-8"))
        batch = payload.get("rows", [])
        if not batch:
            break
        for entry in batch:
            r = entry["row"]
            rows.append({"sentence1": r["sentence1"], "sentence2": r["sentence2"],
                         "label": str(r["label"])})
        offset += len(batch)
        if len(batch) < limit:
            break
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Downloading PAWS-Wiki ... (first {args.n} rows)")
    rows = _load_rows(args.n)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rsplit = int(len(rows) * 0.8)
    train = paws_rows_to_examples(rows[:rsplit])
    test = paws_rows_to_examples(rows[rsplit:])
    print(f"{len(rows)} pairs  |  train {len(train)}  test {len(test)}  (record-level split)")

    # Lexical-only
    lex = Extractor()
    lex_rules, lex_rep = DistillationEngine(lex, args.threshold).fit(train)
    lex_hold = evaluate_rules(lex_rules, lex, test)

    # Lexical + semantic
    embedder = Embedder()
    sentences = []
    for ex in train:
        for fld in ("sentence1", "sentence2"):
            sentences += split(ex.inputs[fld])
    intents = discover_intents(sentences, embedder, k_range=(3, 8), seed=args.seed)
    sem = SemanticExtractor(embedder, intents, fields=["sentence1", "sentence2"],
                            intent_field="sentence2")
    comb = CombinedExtractor(lex, sem)
    comb_rules, comb_rep = DistillationEngine(comb, args.threshold).fit(train)
    comb_hold = evaluate_rules(comb_rules, comb, test)

    print("\nHeld-out kappa on PAWS (higher = rules reproduce the label better):")
    print(f"  lexical only      : kappa {lex_hold['kappa']:.2f}  coverage {lex_hold['coverage']:.0%}")
    print(f"  lexical + semantic: kappa {comb_hold['kappa']:.2f}  coverage {comb_hold['coverage']:.0%}")
    print("\nIf semantic >> lexical, the meaning features are doing real work on adversarial data.")


if __name__ == "__main__":
    main()
