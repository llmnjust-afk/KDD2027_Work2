"""Failure-trace sampling and annotation guide generator.

KDD requires that a failure taxonomy be validated by inter-rater reliability
(Cohen's kappa >= 0.7). This module:
  1. Samples N failed traces from experiment results
  2. Extracts the trace context (question, code, output, answer, classification)
  3. Outputs a structured annotation form (JSON + CSV) for two human annotators
  4. Provides a kappa computation function to run after annotation

Usage:
    python generate_annotation_set.py --input results_full --n 100 --output annotation_set
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def sample_failed_traces(results_dir: str, n: int = 100, seed: int = 42) -> list:
    """Sample N failed traces from experiment detail files."""
    all_runs = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_detail.json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        for run in data.get("runs", []):
            if not run.get("correct", True):  # only failures
                all_runs.append(run)

    rng = random.Random(seed)
    if len(all_runs) <= n:
        sampled = all_runs
    else:
        sampled = rng.sample(all_runs, n)

    print(f"Sampled {len(sampled)} failed traces out of {len(all_runs)} total failures")
    return sampled


def build_annotation_form(runs: list, output_dir: str) -> None:
    """Create annotation forms for two annotators."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON form (structured)
    annotations = []
    for i, run in enumerate(runs):
        annotations.append({
            "id": f"trace_{i:03d}",
            "task_id": run.get("task_id", ""),
            "model": run.get("model", ""),
            "variant": run.get("variant", ""),
            "rep": run.get("rep", 0),
            "final_answer": run.get("final_answer", ""),
            "system_classification": {
                "stage": run.get("stage", ""),
                "category": run.get("category", ""),
                "is_silent": run.get("is_silent", False),
            },
            # annotator fills these:
            "annotator_1": {"stage": "", "category": "", "is_silent": "", "notes": ""},
            "annotator_2": {"stage": "", "category": "", "is_silent": "", "notes": ""},
        })

    with open(os.path.join(output_dir, "annotation_form.json"), "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # CSV form (for easy spreadsheet annotation)
    with open(os.path.join(output_dir, "annotation_form.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "task_id", "model", "final_answer",
            "system_stage", "system_category", "system_silent",
            "annotator1_stage", "annotator1_category", "annotator1_silent", "annotator1_notes",
            "annotator2_stage", "annotator2_category", "annotator2_silent", "annotator2_notes",
        ])
        for i, run in enumerate(runs):
            w.writerow([
                f"trace_{i:03d}",
                run.get("task_id", ""),
                run.get("model", ""),
                run.get("final_answer", "")[:100],
                run.get("stage", ""),
                run.get("category", ""),
                run.get("is_silent", ""),
                "", "", "", "",  # annotator 1
                "", "", "", "",  # annotator 2
            ])

    # Annotation guide
    guide = """# Failure Annotation Guide

## Task
For each failed trace, independently assign:
1. **Stage**: Where did the failure originate?
   - `planning` -- wrong approach/operation chosen
   - `tool_use` -- wrong tool selected or parameterized
   - `execution` -- code raised an exception (loud failure)
   - `interpretation` -- code ran but answer was wrong (silent failure)
2. **Category**: Specific failure type (see taxonomy)
3. **is_silent**: Did the code execute without errors but produce a wrong answer?
   - `True` = code ran, answer wrong (silent)
   - `False` = code crashed or answer not produced (loud)

## 4-Stage Taxonomy

### Planning
- wrong_operation: used mean/median instead of sum/max
- wrong_decomposition: broke the problem into wrong sub-steps
- temporal_leakage: used future/test data in analysis
- data_leakage: used target variable as a feature

### Tool Use
- wrong_tool: trained ML model where simple pandas suffices
- wrong_params: correct tool but wrong parameters
- over_privileged: used higher-privilege tool than needed

### Execution
- runtime_error: general runtime exception
- type_error: type mismatch (e.g., summing strings)
- key_error: wrong column/index name
- security_block: sandbox blocked dangerous code

### Interpretation (Silent)
- wrong_aggregation: used count instead of sum, or similar
- wrong_index: off-by-one or wrong index mapping
- misread_output: answer not extracted correctly from output
- hallucinated_answer: answer not supported by the code output

## Rules
- Annotate independently (do not discuss with the other annotator)
- If unsure, mark your best guess and add notes
- Each trace gets exactly ONE stage + ONE category
- The system's classification is shown for reference but should NOT influence your judgment
"""
    with open(os.path.join(output_dir, "ANNOTATION_GUIDE.md"), "w") as f:
        f.write(guide)

    print(f"Annotation forms written to {output_dir}/")
    print(f"  - annotation_form.json ({len(annotations)} traces)")
    print(f"  - annotation_form.csv (spreadsheet format)")
    print(f"  - ANNOTATION_GUIDE.md (instructions)")


def compute_kappa(annotations: list) -> dict:
    """Compute Cohen's kappa between two annotators.

    Call this after both annotators have filled in their judgments.
    """
    from collections import Counter

    stages_1 = [a["annotator_1"]["stage"] for a in annotations if a["annotator_1"]["stage"]]
    stages_2 = [a["annotator_2"]["stage"] for a in annotations if a["annotator_2"]["stage"]]

    if len(stages_1) != len(stages_2) or not stages_1:
        return {"error": "annotations incomplete", "n": len(stages_1)}

    n = len(stages_1)
    categories = sorted(set(stages_1 + stages_2))

    # observed agreement
    agree = sum(1 for a, b in zip(stages_1, stages_2) if a == b)
    po = agree / n

    # expected agreement (chance)
    c1 = Counter(stages_1)
    c2 = Counter(stages_2)
    pe = sum((c1[c] / n) * (c2[c] / n) for c in categories)

    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    return {
        "n": n,
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
        "cohens_kappa": round(kappa, 4),
        "interpretation": (
            "almost perfect" if kappa > 0.81 else
            "substantial" if kappa > 0.61 else
            "moderate" if kappa > 0.41 else
            "fair" if kappa > 0.21 else
            "poor"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results_full", help="experiment results dir")
    ap.add_argument("--n", type=int, default=100, help="number of traces to sample")
    ap.add_argument("--output", default="annotation_set", help="output dir")
    ap.add_argument("--kappa", action="store_true", help="compute kappa from filled forms")
    args = ap.parse_args()

    if args.kappa:
        with open(os.path.join(args.output, "annotation_form.json")) as f:
            annotations = json.load(f)
        result = compute_kappa(annotations)
        print(json.dumps(result, indent=2))
        return

    runs = sample_failed_traces(args.input, args.n)
    build_annotation_form(runs, args.output)


if __name__ == "__main__":
    main()
