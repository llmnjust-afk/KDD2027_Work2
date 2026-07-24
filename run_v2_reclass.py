#!/usr/bin/env python3
"""B-lite: Re-run synthetic + UCI tasks with full trace logging + v2 classifier.

Saves full traces (code, thought, execution output) and v2 five-stage
classifications. Populates CODE_GENERATION by comparing agent code to gt_path.

Usage:
  OPENAI_API_KEY=sk-... python3 run_v2_reclass.py --models gpt-4o-mini gpt-4o deepseek-chat --reps 3
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentfail.benchmark.task_generators import build_full_benchmark
from agentfail.benchmark.uci_tasks import build_uci_benchmark
from agentfail.benchmark.tasks import TaskSet
from agentfail.agent.react_agent import ReActAgent
from agentfail.agent.sandbox import CodeSandbox
from agentfail.diagnosis.classifier import FailureClassifier
from agentfail.diagnosis.propagation import PropagationAnalyzer
from agentfail.diagnosis.causality import CausalReplay
from agentfail.llm.openai_backend import OpenAIBackend

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")

MODELS = {
    "gpt-4o":      ("gpt-4o", 2.5, 10.0),
    "gpt-4o-mini": ("gpt-4o-mini", 0.15, 0.60),
    "deepseek-chat": ("deepseek-chat", 0.14, 0.28),
}

def load_tasks():
    synth = build_full_benchmark()
    try:
        uci = build_uci_benchmark()
        tasks = list(synth.tasks) + list(uci.tasks)
    except Exception:
        tasks = list(synth.tasks)
    return tasks

def atomic_save(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def run_one(model_key, task, rep, max_steps=6, timeout=90, api_retries=2):
    import tempfile
    api_name, pin, pout = MODELS[model_key]
    llm = OpenAIBackend(
        model=api_name,
        api_key=API_KEY,
        base_url=BASE_URL,
        price_in=pin,
        price_out=pout,
        timeout=timeout,
        max_retries=api_retries,
    )
    workdir = tempfile.mkdtemp(prefix=f"agentfail_{task.task_id}_")
    sandbox = CodeSandbox(workdir)
    task.prepare_data(workdir)
    agent = ReActAgent(llm=llm, sandbox=sandbox, max_steps=max_steps)
    trace = agent.run(task.task_id, task.question)
    clf = FailureClassifier()
    classified = clf.classify(trace, task, trace.final_answer)
    propagator = PropagationAnalyzer()
    propagation = propagator.analyze(classified)
    try:
        causal_replay = CausalReplay()
        attribution = causal_replay.attribute(classified, task)
        causal = attribution.repair_succeeded
    except Exception:
        causal = False
    return {
        "task_id": task.task_id,
        "rep": rep,
        "model": model_key,
        "correct": classified.task_correct,
        "stage": classified.classification.stage.value,
        "category": classified.classification.category.value,
        "is_silent": classified.is_silent_failure,
        "propagation_depth": propagation.propagation_depth,
        "causal_attributed": causal,
        "tokens": trace.total_tokens.total_tokens,
        "final_answer": str(trace.final_answer)[:200] if trace.final_answer else None,
        "evidence": classified.classification.evidence,
        "gt_path": task.gt_path,
        "trace": trace.to_dict(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-4o", "deepseek-chat"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--output", default="results_v2")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--api-retries", type=int, default=2)
    ap.add_argument("--run-retries", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--suite", default="all", choices=["all", "synthetic", "uci"])
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit("OPENAI_API_KEY is required")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Require 0 <= shard-index < shard-count")

    tasks = load_tasks()
    if args.suite == "synthetic":
        tasks = [t for t in tasks if not t.task_id.startswith(("uci_", "dsbench_"))]
    elif args.suite == "uci":
        tasks = [t for t in tasks if t.task_id.startswith("uci_")]
    if args.shard_count > 1:
        tasks = [
            task for index, task in enumerate(tasks)
            if index % args.shard_count == args.shard_index
        ]
    print(f"Loaded {len(tasks)} tasks, models={args.models}, reps={args.reps}")
    print(f"Total runs: {len(tasks) * len(args.models) * args.reps}")

    os.makedirs(args.output, exist_ok=True)
    all_runs = []
    total = len(tasks) * len(args.models) * args.reps
    done = 0
    t0 = time.time()

    for model_key in args.models:
        output_path = os.path.join(args.output, f"{model_key}_detail.json")
        model_runs = []
        if args.resume and os.path.exists(output_path):
            try:
                model_runs = json.load(open(output_path)).get("runs", [])
            except (OSError, json.JSONDecodeError):
                model_runs = []
        completed = {(r.get("task_id"), int(r.get("rep", 0))) for r in model_runs}
        all_runs.extend(model_runs)
        print(f"{model_key}: resuming with {len(completed)} completed runs", flush=True)
        for task in tasks:
            for rep in range(args.reps):
                done += 1
                if (task.task_id, rep) in completed:
                    continue
                r = None
                last_error = None
                for attempt in range(args.run_retries + 1):
                    try:
                        r = run_one(
                            model_key,
                            task,
                            rep,
                            args.max_steps,
                            args.timeout,
                            args.api_retries,
                        )
                        r["status"] = "OK"
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < args.run_retries:
                            time.sleep(min(2 ** attempt, 8))
                if r is None:
                    r = {
                        "task_id": task.task_id, "rep": rep, "model": model_key,
                        "correct": False, "stage": "runtime", "category": "runtime_exception",
                        "is_silent": False, "propagation_depth": 0, "causal_attributed": False,
                        "tokens": 0, "final_answer": None, "evidence": f"ERROR: {last_error}",
                        "gt_path": task.gt_path, "trace": None, "status": "ERR",
                    }
                model_runs.append(r)
                all_runs.append(r)
                atomic_save(output_path, {"runs": model_runs})
                print(
                    f"[{done}/{total}] {model_key} {task.task_id} r{rep}: "
                    f"status={r.get('status')} correct={r['correct']} stage={r['stage']} "
                    f"silent={r['is_silent']} tok={r['tokens']}",
                    flush=True,
                )

    # summary
    from collections import Counter
    stage_dist = Counter(r["stage"] for r in all_runs if not r["correct"])
    cat_dist = Counter(r["category"] for r in all_runs if not r["correct"])
    elapsed = time.time() - t0
    summary = {
        "total_runs": len(all_runs),
        "failed": sum(1 for r in all_runs if not r["correct"]),
        "correct": sum(1 for r in all_runs if r["correct"]),
        "stage_distribution": dict(stage_dist.most_common()),
        "category_distribution": dict(cat_dist.most_common()),
        "elapsed_s": round(elapsed, 1),
        "models": args.models,
        "reps": args.reps,
    }
    if len(args.models) > 1:
        atomic_save(os.path.join(args.output, "summary.json"), summary)
    print(f"\n=== DONE in {elapsed:.0f}s ===")
    print(f"Total: {len(all_runs)}, Failed: {summary['failed']}")
    print("Stage distribution:", dict(stage_dist.most_common()))
    print("Category distribution:", dict(cat_dist.most_common()))

if __name__ == "__main__":
    main()
