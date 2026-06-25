"""Validate Mimic on real HaluEval QA data (proxy test: gold labels as the judge's verdicts).

Usage: python eval/run_halueval.py [--n 1000] [--threshold 0.6] [--seed 42]

Downloads HaluEval QA, builds (context, response) -> grounded examples, distills rules on
a train split, and reports kappa / F1 / coverage on a HELD-OUT test split (the honest number).
"""
from __future__ import annotations

import argparse
import random
import urllib.request

from mimic.datasets import parse_jsonl, halueval_records_to_examples
from mimic.extractor import Extractor
from mimic.engine import DistillationEngine
from mimic.generator import ArtifactGenerator
from mimic.evaluate import evaluate_artifact

URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="max records (each yields 2 examples)")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Downloading HaluEval QA from {URL} ...")
    text = urllib.request.urlopen(URL).read().decode("utf-8")
    records = parse_jsonl(text)[: args.n]

    rng = random.Random(args.seed)
    rng.shuffle(records)
    rsplit = int(len(records) * 0.8)
    train = halueval_records_to_examples(records[:rsplit])
    test = halueval_records_to_examples(records[rsplit:])
    print(f"{len(records)} records -> {len(train) + len(test)} examples  |  "
          f"train {len(train)}  test {len(test)}  (record-level split, no context bleed)")

    rules, train_report = DistillationEngine(Extractor(), args.threshold).fit(train)
    artifact = ArtifactGenerator().to_code(rules, train_report, "speed")

    print(f"\nLearned {len(rules)} rules (train kappa {train_report['kappa']:.2f}):")
    for r in rules:
        verdict = "GROUNDED" if r.verdict else "NOT GROUNDED"
        print(f"  - {r.plain_english} -> {verdict}  ({r.confidence:.0%}, covers {r.coverage})")

    holdout = evaluate_artifact(artifact, test)
    print("\nHeld-out test results (the honest number):")
    print(f"  coverage: {holdout['coverage']:.0%}  ({holdout['n_covered']}/{holdout['n_total']})")
    print(f"  kappa:    {holdout['kappa']:.2f}")
    print(f"  F1:       grounded {holdout['per_class_f1']['True']:.2f} / "
          f"not-grounded {holdout['per_class_f1']['False']:.2f}")


if __name__ == "__main__":
    main()
