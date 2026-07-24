#!/usr/bin/env python3
"""Validate three completed blind forms and prepare agreement/adjudication files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


STAGES = [
    "analytical_plan",
    "code_generation",
    "runtime",
    "output_mismatch",
    "answer_error",
    "unclassifiable",
]


def content_key(row):
    public = {k: v for k, v in row.items() if k not in ("trace_id", "annotation")}
    blob = json.dumps(public, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def normalized_stage(annotation):
    if annotation.get("unclassifiable"):
        return "unclassifiable"
    return annotation.get("failure_stage", "")


def fleiss_kappa(rows, categories):
    if not rows:
        return float("nan")
    n_raters = len(rows[0])
    counts = []
    for ratings in rows:
        counter = Counter(ratings)
        counts.append([counter.get(category, 0) for category in categories])
    p_items = [
        (sum(value * value for value in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in counts
    ]
    p_bar = sum(p_items) / len(p_items)
    p_categories = [
        sum(row[index] for row in counts) / (len(counts) * n_raters)
        for index in range(len(categories))
    ]
    p_expected = sum(value * value for value in p_categories)
    if math.isclose(p_expected, 1.0):
        return 1.0 if math.isclose(p_bar, 1.0) else float("nan")
    return (p_bar - p_expected) / (1 - p_expected)


def cohen_kappa(left, right):
    categories = sorted(set(left) | set(right), key=str)
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(left_counts[c] * right_counts[c] for c in categories) / (n * n)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else float("nan")
    return (observed - expected) / (1 - expected)


def bootstrap_ci(rows, categories, seed=2027, iterations=5000):
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = fleiss_kappa(sample, categories)
        if not math.isnan(value):
            values.append(value)
    values.sort()
    if not values:
        return [None, None]
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


def qc_report(rows):
    formal_by_content = {
        content_key(row): row for row in rows if row["trace_id"].startswith("new_")
    }
    details = []
    for qc in (row for row in rows if row["trace_id"].startswith("qc_")):
        source = formal_by_content.get(content_key(qc))
        qa = qc["annotation"]
        sa = source["annotation"] if source else {}
        details.append(
            {
                "qc_trace_id": qc["trace_id"],
                "source_trace_id": source.get("trace_id") if source else None,
                "stage_match": normalized_stage(qa) == normalized_stage(sa),
                "silent_match": qa.get("is_silent") == sa.get("is_silent"),
                "origin_match": qa.get("originating_step") == sa.get("originating_step"),
            }
        )
    return {
        "matched": sum(item["source_trace_id"] is not None for item in details),
        "stage_matches": sum(item["stage_match"] for item in details),
        "silent_matches": sum(item["silent_match"] for item in details),
        "origin_matches": sum(item["origin_match"] for item in details),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--output", default="annotation_analysis_103")
    args = parser.parse_args()

    paths = {"a": Path(args.a), "b": Path(args.b), "c": Path(args.c)}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    qc = {name: qc_report(rows) for name, rows in data.items()}
    formal = {
        name: {row["trace_id"]: row for row in rows if row["trace_id"].startswith("new_")}
        for name, rows in data.items()
    }
    ids = sorted(set.intersection(*(set(values) for values in formal.values())))
    if len(ids) != 103:
        raise RuntimeError(f"Expected 103 shared formal traces; found {len(ids)}")

    raw = []
    adjudication = []
    provisional = []
    stage_rows = []
    silent_rows = []
    origin_rows = []
    all_classifiable_stage_rows = []

    for trace_id in ids:
        base = formal["a"][trace_id]
        annotations = {name: formal[name][trace_id]["annotation"] for name in "abc"}
        stages = [normalized_stage(annotations[name]) for name in "abc"]
        silents = [annotations[name].get("is_silent") for name in "abc"]
        origins = [annotations[name].get("originating_step") for name in "abc"]
        stage_rows.append(stages)
        silent_rows.append(silents)
        if all(isinstance(value, int) for value in origins):
            origin_rows.append(origins)
        if "unclassifiable" not in stages:
            all_classifiable_stage_rows.append(stages)

        stage_counts = Counter(stages)
        majority_stage, majority_count = stage_counts.most_common(1)[0]
        silent_counts = Counter(silents)
        majority_silent, silent_count = silent_counts.most_common(1)[0]
        origin_counts = Counter(origins)
        majority_origin, origin_count = origin_counts.most_common(1)[0]
        needs_adjudication = (
            majority_count < 3
            or majority_stage == "unclassifiable"
            or silent_count < 3
            or origin_count < 3
        )

        item = {
            "trace_id": trace_id,
            "question": base["question"],
            "ground_truth_answer": base["ground_truth_answer"],
            "ground_truth_path": base["ground_truth_path"],
            "agent_final_answer": base["agent_final_answer"],
            "steps": base["steps"],
            "annotations": annotations,
            "stage_votes": dict(stage_counts),
            "silent_votes": {str(k).lower(): v for k, v in silent_counts.items()},
            "origin_votes": {str(k): v for k, v in origin_counts.items()},
            "needs_adjudication": needs_adjudication,
        }
        raw.append(item)
        if needs_adjudication:
            adjudication.append(
                {
                    **item,
                    "adjudication": {
                        "gold_stage": "",
                        "gold_silent": "",
                        "gold_originating_step": "",
                        "adjudication_reason": "",
                    },
                }
            )
        provisional.append(
            {
                "trace_id": trace_id,
                "provisional_stage": majority_stage if majority_count >= 2 and majority_stage != "unclassifiable" else None,
                "provisional_silent": majority_silent if silent_count >= 2 else None,
                "provisional_originating_step": majority_origin if origin_count >= 2 and isinstance(majority_origin, int) else None,
                "stage_vote_count": majority_count,
                "needs_adjudication": needs_adjudication,
            }
        )

    pairwise = {}
    for left, right in (("a", "b"), ("a", "c"), ("b", "c")):
        pairwise[f"{left}_{right}"] = {
            "stage_kappa_including_unclassifiable": cohen_kappa(
                [normalized_stage(formal[left][i]["annotation"]) for i in ids],
                [normalized_stage(formal[right][i]["annotation"]) for i in ids],
            ),
            "silent_kappa": cohen_kappa(
                [formal[left][i]["annotation"].get("is_silent") for i in ids],
                [formal[right][i]["annotation"].get("is_silent") for i in ids],
            ),
        }

    stage_categories = STAGES
    silent_categories = [False, True]
    report = {
        "n_formal_traces": len(ids),
        "qc": qc,
        "stage_distribution": {
            name: dict(Counter(normalized_stage(formal[name][i]["annotation"]) for i in ids))
            for name in "abc"
        },
        "unclassifiable_counts": {
            name: sum(normalized_stage(formal[name][i]["annotation"]) == "unclassifiable" for i in ids)
            for name in "abc"
        },
        "three_way_full_stage_agreement": sum(len(set(row)) == 1 for row in stage_rows) / len(stage_rows),
        "three_way_full_silent_agreement": sum(len(set(row)) == 1 for row in silent_rows) / len(silent_rows),
        "three_way_full_origin_agreement_on_complete_rows": (
            sum(len(set(row)) == 1 for row in origin_rows) / len(origin_rows) if origin_rows else None
        ),
        "fleiss_stage_including_unclassifiable": fleiss_kappa(stage_rows, stage_categories),
        "fleiss_stage_including_unclassifiable_95ci": bootstrap_ci(stage_rows, stage_categories),
        "n_all_classifiable": len(all_classifiable_stage_rows),
        "fleiss_stage_all_classifiable_subset": fleiss_kappa(
            all_classifiable_stage_rows, STAGES[:-1]
        ),
        "fleiss_silent": fleiss_kappa(silent_rows, silent_categories),
        "fleiss_silent_95ci": bootstrap_ci(silent_rows, silent_categories),
        "pairwise_cohen": pairwise,
        "adjudication_queue_size": len(adjudication),
        "three_way_stage_disagreements": sum(len(set(row)) == 3 for row in stage_rows),
        "two_one_stage_splits": sum(len(set(row)) == 2 for row in stage_rows),
        "unanimous_stage": sum(len(set(row)) == 1 for row in stage_rows),
    }

    (out / "agreement_report_103.json").write_text(json.dumps(report, indent=2))
    (out / "raw_annotations_103.json").write_text(json.dumps(raw, indent=2))
    (out / "adjudication_queue_103.json").write_text(json.dumps(adjudication, indent=2))
    (out / "provisional_majority_gold_103.json").write_text(json.dumps(provisional, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
