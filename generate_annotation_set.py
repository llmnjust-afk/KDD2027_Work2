#!/usr/bin/env python3
"""Generate a 100-trace annotation set for inter-rater reliability (Cohen's kappa).

Samples failed traces from ALL experiments (main + UCI + final), extracts full
context, and outputs annotation forms for 2 annotators + a guide.

After annotation, run with --kappa to compute inter-rater agreement.

Usage:
    python generate_annotation_set.py --n 100 --output annotation_set
    python generate_annotation_set.py --kappa --output annotation_set
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def collect_all_failures(base_dir: str = ".") -> list:
    """Collect all failed runs from all experiment directories."""
    all_failures = []
    dirs = ["results_full", "results_uci", "results_final"]
    for d in dirs:
        path = os.path.join(base_dir, d)
        if not os.path.exists(path):
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith("_detail.json"):
                continue
            with open(os.path.join(path, fname)) as f:
                data = json.load(f)
            model = fname.replace("_detail.json", "")
            for run in data.get("runs", []):
                if not run.get("correct"):
                    all_failures.append({
                        "source_dir": d, "model": model, **run,
                    })
    return all_failures


def sample_failures(failures: list, n: int = 100, seed: int = 42) -> list:
    """Stratified sample of n failures across models."""
    rng = random.Random(seed)
    by_model = defaultdict(list)
    for f in failures:
        by_model[f.get("model", "?")].append(f)

    sampled = []
    total = len(failures)
    for model, runs in by_model.items():
        n_model = max(1, round(n * len(runs) / total))
        n_model = min(n_model, len(runs))
        sampled.extend(rng.sample(runs, n_model))

    if len(sampled) < n:
        remaining = [f for f in failures if f not in sampled]
        sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))
    elif len(sampled) > n:
        sampled = rng.sample(sampled, n)
    return sampled


def build_annotation_form(failures: list, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    annotations = []
    for i, f in enumerate(failures):
        annotations.append({
            "id": f"trace_{i:03d}",
            "task_id": f.get("task_id", ""),
            "model": f.get("model", ""),
            "source": f.get("source_dir", ""),
            "domain": f.get("domain", ""),
            "final_answer": f.get("final_answer", ""),
            "system_stage": f.get("stage", ""),
            "system_category": f.get("category", ""),
            "system_is_silent": f.get("is_silent", False),
            "tokens": f.get("tokens", 0),
            "annotator_1": {"stage": "", "category": "", "is_silent": "", "notes": ""},
            "annotator_2": {"stage": "", "category": "", "is_silent": "", "notes": ""},
        })

    with open(os.path.join(output_dir, "annotation_form.json"), "w") as fn:
        json.dump(annotations, fn, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "annotation_form.csv"), "w", newline="") as fn:
        w = csv.writer(fn)
        w.writerow([
            "id", "task_id", "model", "domain", "final_answer",
            "system_stage", "system_category", "system_silent",
            "annotator1_stage", "annotator1_category", "annotator1_silent", "annotator1_notes",
            "annotator2_stage", "annotator2_category", "annotator2_silent", "annotator2_notes",
        ])
        for i, f in enumerate(failures):
            w.writerow([
                f"trace_{i:03d}", f.get("task_id", ""), f.get("model", ""),
                f.get("domain", ""), str(f.get("final_answer", ""))[:100],
                f.get("stage", ""), f.get("category", ""), f.get("is_silent", ""),
                "", "", "", "", "", "", "", "",
            ])

    guide = """# Failure Annotation Guide

## Task
For each failed trace, independently assign:
1. **Stage**: planning | tool_use | execution | interpretation
2. **Category**: specific failure type (see taxonomy)
3. **is_silent**: True (code ran, answer wrong) | False (code crashed)

## Taxonomy
- planning: wrong_operation, wrong_decomposition, temporal_leakage, data_leakage
- tool_use: wrong_tool, wrong_params, over_privileged
- execution (loud): runtime_error, type_error, key_error, security_block
- interpretation (silent): wrong_aggregation, wrong_index, misread_output, hallucinated_answer

## Rules
- Annotate INDEPENDENTLY
- final_answer=None usually means execution (loud)
- final_answer=value but wrong usually means interpretation (silent)
- System's classification is for reference only; do not be biased by it
"""
    with open(os.path.join(output_dir, "ANNOTATION_GUIDE.md"), "w") as fn:
        fn.write(guide)

    print(f"Annotation forms written to {output_dir}/")
    print(f"  - annotation_form.json ({len(annotations)} traces)")
    print(f"  - annotation_form.csv (spreadsheet format)")
    print(f"  - ANNOTATION_GUIDE.md")


def compute_kappa(annotations: list) -> dict:
    stages_1 = [a["annotator_1"]["stage"] for a in annotations if a["annotator_1"].get("stage")]
    stages_2 = [a["annotator_2"]["stage"] for a in annotations if a["annotator_2"].get("stage")]
    if len(stages_1) != len(stages_2) or not stages_1:
        return {"error": "annotations incomplete", "n_filled": len(stages_1)}
    n = len(stages_1)
    categories = sorted(set(stages_1 + stages_2))
    agree = sum(1 for a, b in zip(stages_1, stages_2) if a == b)
    po = agree / n
    c1, c2 = Counter(stages_1), Counter(stages_2)
    pe = sum((c1[c] / n) * (c2[c] / n) for c in categories)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {
        "n_annotated": n,
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
        "cohens_kappa": round(kappa, 4),
        "interpretation": ("almost perfect" if kappa > 0.81 else "substantial" if kappa > 0.61
                           else "moderate" if kappa > 0.41 else "fair" if kappa > 0.21 else "poor"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--output", default="annotation_set")
    ap.add_argument("--kappa", action="store_true")
    args = ap.parse_args()

    if args.kappa:
        form_path = os.path.join(args.output, "annotation_form.json")
        if not os.path.exists(form_path):
            print(f"ERROR: {form_path} not found.")
            return 1
        with open(form_path) as f:
            annotations = json.load(f)
        result = compute_kappa(annotations)
        print(json.dumps(result, indent=2))
        return 0

    base_dir = os.path.dirname(os.path.abspath(__file__))
    failures = collect_all_failures(base_dir)
    print(f"Collected {len(failures)} failed traces from all experiments")
    if not failures:
        print("No failures found.")
        return 1
    sampled = sample_failures(failures, args.n)
    print(f"Sampled {len(sampled)} traces (stratified by model)")
    print(f"Model dist: {dict(Counter(f.get('model') for f in sampled))}")
    print(f"Stage dist: {dict(Counter(f.get('stage') for f in sampled))}")
    build_annotation_form(sampled, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
