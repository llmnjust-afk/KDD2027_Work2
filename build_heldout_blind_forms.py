#!/usr/bin/env python3
"""Lock a classifier-blind held-out failure sample and create annotation forms."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = "agentfail-heldout-2027"


def rank(run):
    key = f"{SEED}|{run['model']}|{run['task_id']}"
    return hashlib.sha256(key.encode()).hexdigest()


def load_runs(path):
    runs = []
    for file in sorted(Path(path).glob("*.json")):
        runs.extend(json.loads(file.read_text()).get("runs", []))
    unique = {}
    for run in runs:
        key = (run.get("model"), run.get("task_id"))
        prior = unique.get(key)
        if prior is not None and prior != run:
            raise RuntimeError(f"conflicting duplicate run: {key}")
        unique[key] = run
    return list(unique.values())


def select(candidates, target, max_per_model):
    candidates = sorted(candidates, key=rank)
    selected = []
    selected_keys = set()
    model_counts = Counter()

    # First guarantee every available family-model cell one representative.
    cells = defaultdict(list)
    for run in candidates:
        cells[(run["family"], run["model"])].append(run)
    for key in sorted(cells):
        run = min(cells[key], key=rank)
        if model_counts[run["model"]] < max_per_model:
            selected.append(run)
            selected_keys.add((run["model"], run["task_id"]))
            model_counts[run["model"]] += 1

    # Fill by frozen hash order while respecting the preregistered model cap.
    for run in candidates:
        if len(selected) >= target:
            break
        key = (run["model"], run["task_id"])
        if key in selected_keys or model_counts[run["model"]] >= max_per_model:
            continue
        selected.append(run)
        selected_keys.add(key)
        model_counts[run["model"]] += 1
    return selected


def public_entry(trace_id, run):
    trace = run["trace"]
    steps = []
    for index, step in enumerate(trace.get("steps", [])):
        steps.append(
            {
                "step": index,
                "thought": step.get("thought", ""),
                "action_type": step.get("action_type", ""),
                "code": step.get("code", ""),
                "stdout": step.get("execution_stdout"),
                "success": step.get("execution_success"),
                "error_type": step.get("execution_error_type"),
                "error_message": step.get("execution_error_message"),
                "answer": step.get("execution_answer"),
            }
        )
    return {
        "trace_id": trace_id,
        "question": run["question"],
        "ground_truth_answer": run["gt_answer"],
        "ground_truth_path": run["gt_path"],
        "agent_final_answer": run["final_answer"],
        "steps": steps,
        "annotation": {
            "failure_stage": "",
            "is_silent": "",
            "originating_step": "",
            "confidence_1_to_5": "",
            "unclassifiable": False,
            "evidence": "",
        },
    }


GUIDE = """# AgentFail Independent Held-Out Annotation Guide

Annotate independently. Do not use an LLM and do not inspect classifier output,
model identity, other annotators' forms, or the private directory.

Choose the earliest stage supported by evidence:

1. `runtime`: an actual execution exception or traceback.
2. `analytical_plan`: before relevant code executes, Thought selects the wrong
   analysis target or operation.
3. `code_generation`: Thought is correct, but code uses a wrong column,
   operation, implementation, or omits the promised computation/output.
4. `output_mismatch`: the agent misreads, omits, or reports a result unsupported
   by execution output. Misinterpreting empty stdout from an unprinted expression
   belongs here.
5. `answer_error`: plan, code, output, and report are internally consistent but
   wrong relative to the ground truth.

Stage and observability are independent. Set `is_silent=false` when the trace
explicitly exposes failure (exception, abstention, or clear error message), even
if its stage is Output-Mismatch or Code-Generation. Fill all fields and cite
specific Thought/Code/stdout evidence.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="heldout_collection")
    parser.add_argument("--output", default="heldout_blind")
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--minimum", type=int, default=100)
    parser.add_argument("--max-per-model", type=int, default=45)
    parser.add_argument("--qc", type=int, default=5)
    args = parser.parse_args()

    runs = load_runs(args.collection)
    valid = [run for run in runs if run.get("status") == "OK" and run.get("trace")]
    failures = [run for run in valid if not run.get("correct")]
    selected = select(failures, args.target, args.max_per_model)
    if len(selected) < args.minimum:
        raise RuntimeError(
            f"Only {len(selected)} eligible held-out failures; need at least {args.minimum}. "
            "Collect an additional frozen repetition before sampling."
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    private = output / "private"
    private.mkdir(exist_ok=True)
    canonical = []
    mapping = []
    for index, run in enumerate(selected, start=1):
        trace_id = f"heldout_test_{index:03d}"
        canonical.append(public_entry(trace_id, run))
        mapping.append(
            {
                "trace_id": trace_id,
                "model": run["model"],
                "task_id": run["task_id"],
                "family": run["family"],
                "source_rank": rank(run),
            }
        )

    lock_payload = {
        "seed": SEED,
        "target": args.target,
        "selected": mapping,
    }
    lock_text = json.dumps(lock_payload, sort_keys=True, separators=(",", ":"))
    lock_sha = hashlib.sha256(lock_text.encode()).hexdigest()
    (private / "sample_lock.json").write_text(json.dumps(lock_payload, indent=2))
    (private / "sample_lock.sha256").write_text(f"{lock_sha}  sample_lock.json\n")
    (private / "canonical.json").write_text(json.dumps(canonical, indent=2))
    (output / "ANNOTATION_GUIDE.md").write_text(GUIDE)

    rng = random.Random(2027)
    qc_sources = rng.sample(range(len(canonical)), args.qc)
    qc_map = []
    for annotator_index, name in enumerate("abc", start=1):
        form = json.loads(json.dumps(canonical))
        for qc_index, source_index in enumerate(qc_sources, start=1):
            duplicate = json.loads(json.dumps(canonical[source_index]))
            duplicate["trace_id"] = f"heldout_qc_{annotator_index}_{qc_index:02d}"
            form.append(duplicate)
            qc_map.append(
                {
                    "annotator": name,
                    "qc_trace_id": duplicate["trace_id"],
                    "source_trace_id": canonical[source_index]["trace_id"],
                }
            )
        random.Random(2027 + annotator_index).shuffle(form)
        (output / f"annotator_{name}_form.json").write_text(json.dumps(form, indent=2))
    (private / "qc_mapping.json").write_text(json.dumps(qc_map, indent=2))

    summary = {
        "collected_runs": len(runs),
        "valid_runs": len(valid),
        "failed_runs": len(failures),
        "selected_failures": len(selected),
        "sample_lock_sha256": lock_sha,
        "models": dict(Counter(item["model"] for item in mapping)),
        "families": dict(Counter(item["family"] for item in mapping)),
    }
    (private / "sampling_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
