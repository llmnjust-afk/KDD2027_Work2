#!/usr/bin/env python3
"""Final experiment: baseline-only on 280 tasks (80 synthetic + 200 real DSBench).

This is the definitive experiment for the benchmark paper. Runs only baselines
(no verifier variants -- method contributions already shown not to work).
5 models x 280 tasks x 3 repeats = 4200 runs.

The 200 real DSBench tasks provide external validity; the 80 synthetic tasks
provide controlled failure-trap coverage. Together they address the
external-validity concern that reviewers raised.

Usage:
    python run_final_experiment.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentfail.benchmark.task_generators import build_full_benchmark
from agentfail.benchmark.dsbench_real_adapter import build_dsbench_real_subset
from agentfail.benchmark.tasks import TaskSet
from agentfail.agent.sandbox import CodeSandbox
from agentfail.agent.react_agent import ReActAgent, AgentTrace
from agentfail.diagnosis.classifier import FailureClassifier
from agentfail.diagnosis.propagation import PropagationAnalyzer, PropagationReport
from agentfail.diagnosis.causality import CausalReplay, CausalAttribution
from agentfail.llm.openai_backend import OpenAIBackend
from agentfail.metrics.failure_metrics import compute_failure_metrics
from agentfail.metrics.economics import compute_economics
from agentfail.metrics.aggregate import compute_aggregate
import tempfile


MODEL_CONFIGS = {
    "gpt-4o-mini": ("gpt-4o-mini", "gpt-4o-mini", 0.15, 0.60),
    "gpt-4o": ("gpt-4o", "gpt-4o", 2.5, 10.0),
    "deepseek-chat": ("deepseek-chat", "deepseek-v3", 0.14, 0.28),
    "deepseek-r1": ("deepseek-reasoner", "deepseek-r1", 0.55, 2.19),
    "qwen3-max": ("qwen3-max-2026-01-23", "qwen3-max-2026", 0.5, 1.5),
}


def make_backend(api_name, pin, pout):
    return OpenAIBackend(
        model=api_name,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
        price_in=pin, price_out=pout, temperature=0.0, max_tokens=2048,
    )


def run_one(model_key, task, max_steps):
    api_name, display, pin, pout = MODEL_CONFIGS[model_key]
    llm = make_backend(api_name, pin, pout)
    workdir = tempfile.mkdtemp(prefix=f"af_{task.task_id}_")
    sandbox = CodeSandbox(workdir)
    task.prepare_data(workdir)
    agent = ReActAgent(llm=llm, sandbox=sandbox, max_steps=max_steps)
    trace = agent.run(task.task_id, task.question)
    classified = FailureClassifier().classify(trace, task, trace.final_answer)
    propagation = PropagationAnalyzer().analyze(classified)
    causal = CausalReplay().attribute(classified, task)
    return trace, classified, propagation, causal, llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--models", nargs="*",
                    default=["gpt-4o-mini", "gpt-4o", "deepseek-chat",
                             "deepseek-r1", "qwen3-max"])
    ap.add_argument("--n-synthetic", type=int, default=16)
    ap.add_argument("--n-real", type=int, default=200)
    ap.add_argument("--output", default="results_final")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        return 1

    os.makedirs(args.output, exist_ok=True)

    # build task set: synthetic + real DSBench
    synth = build_full_benchmark(n_per_domain=args.n_synthetic)
    real = build_dsbench_real_subset(
        "/data/lab/DSBench", n_questions=args.n_real,
        preferred_task_ids=[
            "00000043", "00000033", "00000038", "00000010", "00000035",
            "00000030", "00000016", "00000005", "00000006", "00000007",
        ],
    )
    taskset = TaskSet(tasks=list(synth.tasks) + list(real.tasks))
    n_synth = len(synth)
    n_real = len(real)
    n_total = len(taskset)
    print(f"Final experiment: {n_synth} synthetic + {n_real} real DSBench = {n_total} tasks")
    print(f"Models: {args.models} | Repeats: {args.repeats} | Total runs: {n_total * args.repeats * len(args.models)}")

    per_model_runs = {}
    per_model_aggregate = {}

    for model_key in args.models:
        api_name, display, pin, pout = MODEL_CONFIGS[model_key]
        print(f"\n{'='*60}\nRunning: {model_key} ({display})\n{'='*60}")

        runs = []
        all_classified, all_propagations, all_traces = [], [], []
        all_correct, all_causal = [], []

        for task in taskset:
            for rep in range(args.repeats):
                t0 = time.time()
                try:
                    trace, classified, propagation, attribution, llm = run_one(
                        model_key, task, args.max_steps
                    )
                    status = "OK"
                except Exception as e:
                    print(f"  ERROR {task.task_id} r{rep}: {e}")
                    traceback.print_exc()
                    trace = AgentTrace(task_id=task.task_id, model=display)
                    classified = FailureClassifier().classify(trace, task, None)
                    propagation = PropagationReport(-1, -1, 0, False, False, [])
                    attribution = CausalAttribution(-1, "", False, 0.0, str(e))
                    status = "ERR"

                elapsed = time.time() - t0
                all_traces.append(trace)
                all_correct.append(classified.task_correct)
                all_classified.append(classified)
                all_propagations.append(propagation)
                all_causal.append(attribution.replay_succeeded)

                run_rec = {
                    "task_id": task.task_id, "rep": rep,
                    "model": display, "domain": task.domain,
                    "correct": classified.task_correct,
                    "stage": classified.classification.stage.value,
                    "category": classified.classification.category.value,
                    "is_silent": classified.is_silent_failure,
                    "propagation_depth": propagation.propagation_depth,
                    "recovered": propagation.was_recovered,
                    "causal_attributed": attribution.replay_succeeded,
                    "tokens": trace.total_tokens.total_tokens,
                    "elapsed_s": round(elapsed, 2),
                    "status": status,
                    "final_answer": str(trace.final_answer)[:200] if trace.final_answer else None,
                    "is_real": task.domain == "dsbench_real",
                }
                runs.append(run_rec)
                mark = "+" if classified.task_correct else ("~" if classified.is_silent_failure else "x")
                src = "R" if task.domain == "dsbench_real" else "S"
                print(f"  [{mark}|{src}] {task.task_id} r{rep} | stage={run_rec['stage']} "
                      f"silent={run_rec['is_silent']} tok={run_rec['tokens']} {elapsed:.1f}s")

        # per-run metrics
        per_run_metrics = []
        for rep in range(args.repeats):
            rep_runs = [r for r in runs if r["rep"] == rep]
            n = len(rep_runs)
            nc = sum(1 for r in rep_runs if r["correct"])
            ns = sum(1 for r in rep_runs if r["is_silent"])
            nf = n - nc
            tok = sum(r["tokens"] for r in rep_runs)
            per_run_metrics.append({
                "success_rate": nc / n if n else 0,
                "silent_failure_rate": ns / nf if nf else 0,
                "token_per_success": tok / nc if nc else float("inf"),
                "cost_per_success": (tok * pin / 1e6) / nc if nc else float("inf"),
            })

        agg = compute_aggregate(model_key, per_run_metrics, pin, pout)
        per_model_aggregate[model_key] = agg
        per_model_runs[model_key] = runs

        fm = compute_failure_metrics(all_classified, all_propagations, all_causal)
        em = compute_economics(all_traces, all_correct, pin, pout)

        with open(os.path.join(args.output, f"{model_key}_detail.json"), "w") as f:
            json.dump({
                "runs": runs, "failure_metrics": fm.as_dict(),
                "economics": em.as_dict(), "aggregate": agg.as_dict(),
            }, f, indent=2, ensure_ascii=False)

        print(f"\n  --- {model_key} summary ---")
        print(f"  success_rate: {agg.success_rate_mean:.4f} +/- {agg.success_rate_std:.4f}")
        print(f"  silent_fail:  {agg.sfr_mean:.4f}")
        print(f"  tok/success:  {agg.token_per_success_mean:.0f}")
        # per-domain breakdown
        real_runs = [r for r in runs if r["is_real"]]
        synth_runs = [r for r in runs if not r["is_real"]]
        sr_real = sum(1 for r in real_runs if r["correct"]) / len(real_runs) if real_runs else 0
        sr_synth = sum(1 for r in synth_runs if r["correct"]) / len(synth_runs) if synth_runs else 0
        print(f"  SR (real DSBench): {sr_real:.4f} | SR (synthetic): {sr_synth:.4f}")

    summary = {
        "experiment": "final_benchmark",
        "config": {
            "n_repeats": args.repeats, "max_steps": args.max_steps,
            "n_synthetic": n_synth, "n_real": n_real, "n_total": n_total,
            "models": args.models,
        },
        "aggregate": {m: a.as_dict() for m, a in per_model_aggregate.items()},
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("FINAL EXPERIMENT COMPLETE")
    print("=" * 60)
    for m, a in per_model_aggregate.items():
        print(f"  {m}: SR={a.success_rate_mean:.4f} SFR={a.sfr_mean:.4f} "
              f"tok/s={a.token_per_success_mean:.0f}")

    if not args.no_push:
        print("\n=== Auto-pushing final results to GitHub ===")
        script = Path(__file__).parent / "scripts" / "auto_push_results.sh"
        try:
            subprocess.run(
                ["bash", str(script), args.output,
                 f"Add final benchmark results: {n_total} tasks ({n_real} real DSBench + {n_synth} synthetic), {len(args.models)} models"],
                check=True,
            )
            print("GitHub push successful.")
        except subprocess.CalledProcessError as e:
            print(f"GitHub push failed (exit {e.returncode}), results saved locally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
