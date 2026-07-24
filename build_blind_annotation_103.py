#!/usr/bin/env python3
"""Build three blinded annotation packages for 103 newly sampled traces.

The script must run only after all model result files are complete. It samples
failed traces across model, suite, predicted stage, and trace length; removes
model/system-label information from annotator-facing files; randomizes order
independently; and inserts five blinded duplicate quality-control items.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from agentfail.benchmark.task_generators import build_full_benchmark
from agentfail.benchmark.uci_tasks import build_uci_benchmark


STAGES = {
    "analytical_plan",
    "code_generation",
    "runtime",
    "output_mismatch",
    "answer_error",
}


def load_tasks():
    tasks = list(build_full_benchmark().tasks)
    tasks.extend(build_uci_benchmark().tasks)
    return {task.task_id: task for task in tasks}


def load_runs(results_dir: Path):
    runs = []
    for path in sorted(results_dir.glob("*_detail.json")):
        payload = json.loads(path.read_text())
        runs.extend(payload.get("runs", []))
    return runs


def suite_of(task_id: str) -> str:
    return "uci" if task_id.startswith("uci_") else "synthetic"


def trace_length(run: dict) -> int:
    trace = run.get("trace") or {}
    return len(trace.get("steps", []))


def length_bucket(run: dict) -> str:
    n = trace_length(run)
    if n <= 1:
        return "one_step"
    if n <= 3:
        return "two_to_three"
    return "four_plus"


def existing_keys(annotation_path: Path):
    if not annotation_path.exists():
        return set()
    entries = json.loads(annotation_path.read_text())
    return {(entry.get("task_id"), entry.get("model")) for entry in entries}


def deduplicate_candidates(runs, excluded):
    groups = defaultdict(list)
    for run in runs:
        if run.get("correct") or run.get("status") == "ERR" or not run.get("trace"):
            continue
        stage = run.get("stage")
        if stage not in STAGES:
            continue
        key = (run.get("task_id"), run.get("model"))
        if key in excluded:
            continue
        groups[key].append(run)

    candidates = []
    for group in groups.values():
        group.sort(
            key=lambda run: (
                stage_priority(run.get("stage")),
                -trace_length(run),
                int(run.get("rep", 0)),
            )
        )
        candidates.append(group[0])
    return candidates


def stage_priority(stage: str) -> int:
    return {
        "code_generation": 0,
        "analytical_plan": 1,
        "answer_error": 2,
        "runtime": 3,
        "output_mismatch": 4,
    }.get(stage, 5)


def stratified_sample(candidates, target: int, seed: int):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for run in candidates:
        key = (
            run.get("model", "unknown"),
            suite_of(run.get("task_id", "")),
            run.get("stage", "unknown"),
            length_bucket(run),
        )
        strata[key].append(run)
    for values in strata.values():
        rng.shuffle(values)

    selected = []
    # Round-robin guarantees broad stratum coverage before proportional fill.
    active = sorted(strata)
    while active and len(selected) < target:
        next_active = []
        for key in active:
            if strata[key] and len(selected) < target:
                selected.append(strata[key].pop())
            if strata[key]:
                next_active.append(key)
        active = next_active

    if len(selected) < target:
        used = {(r.get("task_id"), r.get("model")) for r in selected}
        remaining = [
            run for run in candidates
            if (run.get("task_id"), run.get("model")) not in used
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: target - len(selected)])

    if len(selected) < target:
        raise RuntimeError(
            f"Only {len(selected)} unique failed model-task pairs available; need {target}."
        )
    return selected[:target]


def public_trace(entry_id: str, run: dict, task) -> dict:
    trace = run.get("trace") or {}
    steps = []
    for idx, step in enumerate(trace.get("steps", [])):
        steps.append(
            {
                "step": idx,
                "thought": step.get("thought", ""),
                "action_type": step.get("action_type", ""),
                "code": step.get("code", ""),
                "stdout": step.get("execution_stdout", ""),
                "success": step.get("execution_success"),
                "error_type": step.get("execution_error_type"),
                "error_message": step.get("execution_error_message"),
                "answer": step.get("execution_answer"),
            }
        )
    return {
        "trace_id": entry_id,
        "question": task.question,
        "ground_truth_answer": str(task.answer),
        "ground_truth_path": task.gt_path,
        "agent_final_answer": run.get("final_answer"),
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


def guide_text():
    return """# AgentFail Blind Annotation Guide

Annotate independently. Do not communicate trace-level decisions until all
forms are locked. Model identity, system predictions, and LLM pre-annotations
are intentionally hidden.

Use the first stage at which the failure is detectable:

1. `runtime`: code raises an exception or execution error. Silent = false.
2. `analytical_plan`: Thought states an incorrect approach before code.
3. `code_generation`: Thought is correct, but code implements a wrong operation.
4. `output_mismatch`: execution output supports one result, but the agent reports
   another result or hallucinates/misreads the output.
5. `answer_error`: code, output, and report are internally consistent, but the
   approach is wrong relative to the ground truth.

For every trace, fill all fields. Cite a Thought, Code, stdout, or error line in
`evidence`. Use `unclassifiable=true` only when the trace is insufficient.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_v2")
    parser.add_argument("--existing", default="annotation_form.json")
    parser.add_argument("--output", default="annotation_blind_103")
    parser.add_argument("--target", type=int, default=103)
    parser.add_argument("--duplicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    runs = load_runs(Path(args.results))
    candidates = deduplicate_candidates(runs, existing_keys(Path(args.existing)))
    selected = stratified_sample(candidates, args.target, args.seed)

    canonical = []
    private = []
    for index, run in enumerate(selected, start=1):
        trace_id = f"new_{index:03d}"
        task = tasks[run["task_id"]]
        canonical.append(public_trace(trace_id, run, task))
        private.append(
            {
                "trace_id": trace_id,
                "task_id": run.get("task_id"),
                "model": run.get("model"),
                "rep": run.get("rep"),
                "suite": suite_of(run.get("task_id", "")),
                "system_stage": run.get("stage"),
                "system_category": run.get("category"),
                "system_silent": run.get("is_silent"),
                "trace_length": trace_length(run),
            }
        )

    (out / "ANNOTATION_GUIDE.md").write_text(guide_text())
    private_dir = out / "private"
    private_dir.mkdir(exist_ok=True)
    (private_dir / "trace_mapping.json").write_text(json.dumps(private, indent=2))
    (private_dir / "canonical_unshuffled.json").write_text(json.dumps(canonical, indent=2))

    rng = random.Random(args.seed)
    duplicate_indices = rng.sample(range(len(canonical)), args.duplicates)
    qc_mapping = []
    for annotator_index, name in enumerate(("a", "b", "c"), start=1):
        form = json.loads(json.dumps(canonical))
        for duplicate_number, source_index in enumerate(duplicate_indices, start=1):
            duplicate = json.loads(json.dumps(canonical[source_index]))
            duplicate["trace_id"] = f"qc_{annotator_index}_{duplicate_number:02d}"
            form.append(duplicate)
            qc_mapping.append(
                {
                    "annotator": name,
                    "qc_trace_id": duplicate["trace_id"],
                    "source_trace_id": canonical[source_index]["trace_id"],
                }
            )
        random.Random(args.seed + annotator_index).shuffle(form)
        (out / f"annotator_{name}_form.json").write_text(json.dumps(form, indent=2))

    (private_dir / "qc_mapping.json").write_text(json.dumps(qc_mapping, indent=2))

    summary = {
        "unique_traces": len(canonical),
        "rows_per_annotator": len(canonical) + args.duplicates,
        "models": dict(Counter(item["model"] for item in private)),
        "suites": dict(Counter(item["suite"] for item in private)),
        "system_stages_hidden": dict(Counter(item["system_stage"] for item in private)),
        "length_buckets": dict(
            Counter(length_bucket(run) for run in selected)
        ),
    }
    (private_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
