#!/usr/bin/env python3
"""Evaluate the live rule classifier on the final 103/150 human gold sets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from agentfail.agent.react_agent import AgentTrace, TraceStep
from agentfail.agent.sandbox import ExecutionResult
from agentfail.benchmark.tasks import Task
from agentfail.diagnosis.classifier import FailureClassifier


CLASSES = ["analytical_plan", "code_generation", "runtime", "output_mismatch", "answer_error"]


def reconstruct(entry, new_schema):
    steps = []
    for raw in entry.get("steps", []):
        action_type = raw.get("action_type", "code")
        execution = None
        if action_type == "code":
            success = raw.get("success")
            execution = ExecutionResult(
                success=bool(success),
                stdout=raw.get("stdout") or "",
                error_type=raw.get("error_type"),
                error_message=raw.get("error_message"),
                answer=raw.get("answer"),
            )
        steps.append(
            TraceStep(
                step_index=int(raw.get("step", len(steps))),
                thought=raw.get("thought", ""),
                action_type=action_type,
                code=raw.get("code", ""),
                execution=execution,
            )
        )
    trace_id = entry["trace_id"] if new_schema else entry["id"]
    final_answer = entry.get("agent_final_answer") if new_schema else entry.get("final_answer")
    trace = AgentTrace(task_id=trace_id, model="", steps=steps, final_answer=final_answer)
    task = Task(
        task_id=trace_id,
        question=entry.get("question", ""),
        domain="unknown",
        answer=entry.get("ground_truth_answer") if new_schema else entry.get("gt_answer"),
        gt_path=entry.get("ground_truth_path", []) if new_schema else entry.get("gt_path", []),
        traps=[],
        data_generator=lambda _: None,
        # Every selected item is a human-confirmed failed trace. Avoid changing
        # the evaluation cohort through representation-specific answer checks.
        answer_checker=lambda _pred, _gold: False,
    )
    return trace_id, trace, task, final_answer


def score(rows):
    total = len(rows)
    report = {
        "n": total,
        "accuracy": sum(row["pred_stage"] == row["gold_stage"] for row in rows) / total,
        "silent_accuracy": sum(row["pred_silent"] == row["gold_silent"] for row in rows) / total,
        "origin_exact": sum(row["pred_origin"] == row["gold_origin"] for row in rows) / total,
        "origin_within_one": sum(abs(row["pred_origin"] - row["gold_origin"]) <= 1 for row in rows) / total,
        "gold_distribution": dict(Counter(row["gold_stage"] for row in rows)),
        "pred_distribution": dict(Counter(row["pred_stage"] for row in rows)),
        "per_class": {},
    }
    f1s = []
    observed_f1s = []
    for label in CLASSES:
        tp = sum(row["gold_stage"] == label and row["pred_stage"] == label for row in rows)
        fp = sum(row["gold_stage"] != label and row["pred_stage"] == label for row in rows)
        fn = sum(row["gold_stage"] == label and row["pred_stage"] != label for row in rows)
        support = sum(row["gold_stage"] == label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        if support:
            observed_f1s.append(f1)
        report["per_class"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    report["macro_f1_5class"] = sum(f1s) / len(f1s)
    report["macro_f1_observed"] = sum(observed_f1s) / len(observed_f1s)
    rng = random.Random(2027)
    bootstrap = []
    for _ in range(5000):
        sample = [rows[rng.randrange(total)] for _ in range(total)]
        bootstrap.append(
            sum(row["pred_stage"] == row["gold_stage"] for row in sample) / total
        )
    bootstrap.sort()
    report["accuracy_95ci"] = [
        bootstrap[int(0.025 * (len(bootstrap) - 1))],
        bootstrap[int(0.975 * (len(bootstrap) - 1))],
    ]
    report["mismatches"] = [row for row in rows if row["pred_stage"] != row["gold_stage"]]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold103", default="final_gold_150/adjudicated_gold_103.json")
    parser.add_argument("--gold150", default="final_gold_150/adjudicated_gold_150.json")
    parser.add_argument("--raw103", default="annotation_analysis_103/raw_annotations_103.json")
    parser.add_argument("--old-traces", default="annotation_set_v3/annotation_form.json")
    parser.add_argument("--output", default="classifier_eval_final150")
    args = parser.parse_args()

    gold103 = {item["trace_id"]: item for item in json.loads(Path(args.gold103).read_text())}
    gold150 = {item["trace_id"]: item for item in json.loads(Path(args.gold150).read_text())}
    raw103 = {item["trace_id"]: item for item in json.loads(Path(args.raw103).read_text())}
    old = {item["id"]: item for item in json.loads(Path(args.old_traces).read_text())}
    classifier = FailureClassifier()
    predictions = []

    for trace_id, gold in gold150.items():
        is_new = trace_id.startswith("new_")
        entry = raw103[trace_id] if is_new else old[trace_id]
        _, trace, task, final_answer = reconstruct(entry, is_new)
        result = classifier.classify(trace, task, final_answer)
        predictions.append(
            {
                "trace_id": trace_id,
                "cohort": "new103" if is_new else "old47",
                "gold_stage": gold["gold_stage"],
                "pred_stage": result.classification.stage.value,
                "gold_silent": bool(gold["gold_silent"]),
                "pred_silent": bool(result.classification.is_silent),
                "gold_origin": int(gold.get("gold_originating_step", 0)),
                "pred_origin": int(result.classification.step_index),
                "pred_category": result.classification.category.value,
                "pred_evidence": result.classification.evidence,
            }
        )

    reports = {
        "new103": score([row for row in predictions if row["cohort"] == "new103"]),
        "combined150": score(predictions),
        "code_generation_12": score([row for row in predictions if row["gold_stage"] == "code_generation"]),
        "observable_output_mismatch_39": score(
            [row for row in predictions if row["gold_stage"] == "output_mismatch" and not row["gold_silent"]]
        ),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2))
    (out / "metrics.json").write_text(json.dumps(reports, indent=2))
    compact = {
        name: {key: value for key, value in report.items() if key != "mismatches"}
        for name, report in reports.items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
