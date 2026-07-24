#!/usr/bin/env python3
"""Run leakage-free deterministic baselines on the final 150 human gold."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


CLASSES = ["code_generation", "runtime", "output_mismatch", "answer_error"]


def metrics(gold, predictions):
    result = {
        "n": len(gold),
        "accuracy": sum(a == b for a, b in zip(gold, predictions)) / len(gold),
        "gold_distribution": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(predictions)),
        "per_class": {},
    }
    f1s = []
    recalls = []
    for label in CLASSES:
        tp = sum(g == label and p == label for g, p in zip(gold, predictions))
        fp = sum(g != label and p == label for g, p in zip(gold, predictions))
        fn = sum(g == label and p != label for g, p in zip(gold, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        recalls.append(recall)
        result["per_class"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(g == label for g in gold),
        }
    result["macro_f1"] = sum(f1s) / len(f1s)
    result["balanced_accuracy"] = sum(recalls) / len(recalls)
    return result


def actual_exception(entry):
    for step in entry.get("steps", []):
        if step.get("action_type") != "code":
            continue
        if step.get("error_type") or step.get("error_message"):
            return True
        if step.get("success") is False and (
            "traceback" in str(step.get("stdout", "")).lower()
            or "error" in str(step.get("stdout", "")).lower()
        ):
            return True
    return False


def main():
    gold = json.load(open("final_gold_150/adjudicated_gold_150.json"))
    old = {
        item["id"]: item
        for item in json.load(open("annotation_set_v3/annotation_form.json"))
    }
    new = {
        item["trace_id"]: item
        for item in json.load(open("annotation_analysis_103/raw_annotations_103.json"))
    }
    rule_predictions = {
        item["trace_id"]: item
        for item in json.load(open("classifier_eval_final150/predictions.json"))
    }

    rows = []
    for item in gold:
        trace_id = item["trace_id"]
        is_new = trace_id.startswith("new_")
        trace = new[trace_id] if is_new else old[trace_id]
        steps = trace["steps"]
        final_answer = trace.get("agent_final_answer") if is_new else trace.get("final_answer")
        rows.append(
            {
                "trace_id": trace_id,
                "gold_stage": item["gold_stage"],
                "steps": steps,
                "final_answer": final_answer,
                "code_length": sum(len(step.get("code") or "") for step in steps),
                "actual_exception": actual_exception(trace),
            }
        )

    gold_stage = [row["gold_stage"] for row in rows]
    predictions = {
        "majority_fixed_output_mismatch": ["output_mismatch"] * len(rows),
        "code_length_200": [
            "runtime" if row["code_length"] < 200 else "output_mismatch" for row in rows
        ],
        "answer_presence": [
            "output_mismatch" if row["final_answer"] and str(row["final_answer"]).strip() else "runtime"
            for row in rows
        ],
        "observable_exception": [
            "runtime" if row["actual_exception"] else "output_mismatch" for row in rows
        ],
        "rule_classifier_oracle_assisted": [
            rule_predictions[row["trace_id"]]["pred_stage"] for row in rows
        ],
    }
    report = {name: metrics(gold_stage, pred) for name, pred in predictions.items()}

    # Step-localization baselines are valid only for the new 103, which have
    # adjudicated origin labels.
    gold103 = json.load(open("final_gold_150/adjudicated_gold_103.json"))
    raw103 = new
    rng = random.Random(2027)
    last, random_pred, origins = [], [], []
    for item in gold103:
        trace = raw103[item["trace_id"]]
        indices = [step["step"] for step in trace["steps"] if step.get("action_type") == "code"]
        if not indices:
            indices = [step["step"] for step in trace["steps"]]
        last.append(max(indices))
        random_pred.append(rng.choice(indices))
        origins.append(item["gold_originating_step"])
    report["step_localization_new103"] = {
        "n": len(origins),
        "last_step_accuracy": sum(a == b for a, b in zip(origins, last)) / len(origins),
        "last_step_mae": sum(abs(a - b) for a, b in zip(origins, last)) / len(origins),
        "random_step_accuracy_seed2027": sum(a == b for a, b in zip(origins, random_pred)) / len(origins),
        "random_step_mae_seed2027": sum(abs(a - b) for a, b in zip(origins, random_pred)) / len(origins),
    }

    out = Path("baseline_rerun_final150")
    out.mkdir(exist_ok=True)
    (out / "metrics_all_baselines.json").write_text(json.dumps(report, indent=2))
    for name, pred in predictions.items():
        payload = [
            {"trace_id": row["trace_id"], "prediction": prediction}
            for row, prediction in zip(rows, pred)
        ]
        (out / f"predictions_{name}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
