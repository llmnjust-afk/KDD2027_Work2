#!/usr/bin/env python3
"""Run real-model experiments and auto-push results to GitHub on completion.

Uses ChatAnywhere proxy (https://api.chatanywhere.tech/v1) as the OpenAI-
compatible endpoint. Three economical models:
  - gpt-4o-mini       (cheap closed)
  - deepseek-chat      (DeepSeek-V3, open)
  - qwen3-max-2026-01-23 (latest Qwen, 2026.01)

Each model runs the full benchmark with and without the failure-aware verifier,
so we get paired comparisons for the method contribution.

On completion, calls auto_push_results.sh to commit + push to GitHub.

Usage:
    python run_real_experiment.py --repeats 3
    python run_real_experiment.py --repeats 3 --models gpt-4o-mini
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

# ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from agentfail.benchmark.tasks import build_default_benchmark, TaskSet
from agentfail.benchmark.task_generators import build_full_benchmark
from agentfail.benchmark.dsbench_adapter import build_dsbench_mc_subset
from agentfail.agent.sandbox import CodeSandbox
from agentfail.agent.react_agent import ReActAgent, AgentTrace
from agentfail.diagnosis.classifier import FailureClassifier, ClassifiedTrace
from agentfail.diagnosis.propagation import PropagationAnalyzer, PropagationReport
from agentfail.diagnosis.causality import CausalReplay, CausalAttribution
from agentfail.llm.openai_backend import OpenAIBackend
from agentfail.method import FailureAwareVerifier
from agentfail.metrics.failure_metrics import compute_failure_metrics
from agentfail.metrics.economics import compute_economics
from agentfail.metrics.aggregate import compute_aggregate, paired_ttest, cohens_d
import tempfile


# Model configs: (api_name, display_name, price_in, price_out)
MODEL_CONFIGS = {
    "gpt-4o-mini": ("gpt-4o-mini", "gpt-4o-mini", 0.15, 0.60),
    "gpt-4o": ("gpt-4o", "gpt-4o", 2.5, 10.0),
    "deepseek-chat": ("deepseek-chat", "deepseek-v3", 0.14, 0.28),
    "deepseek-r1": ("deepseek-reasoner", "deepseek-r1", 0.55, 2.19),
    "qwen3-max": ("qwen3-max-2026-01-23", "qwen3-max-2026", 0.5, 1.5),
}


def make_backend(api_name: str, price_in: float, price_out: float) -> OpenAIBackend:
    return OpenAIBackend(
        model=api_name,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
        price_in=price_in,
        price_out=price_out,
        temperature=0.0,
        max_tokens=2048,
    )


def run_one(model_display: str, task, max_steps: int, use_verifier: bool):
    base_key = model_display.replace("+verifier", "")
    api_name, _, pin, pout = MODEL_CONFIGS[base_key]
    llm = make_backend(api_name, pin, pout)
    workdir = tempfile.mkdtemp(prefix=f"agentfail_{task.task_id}_")
    sandbox = CodeSandbox(workdir)
    task.prepare_data(workdir)

    verifier = FailureAwareVerifier(gt_path_keywords=task.gt_path) if use_verifier else None
    agent = ReActAgent(llm=llm, sandbox=sandbox, max_steps=max_steps, verifier=verifier)
    trace = agent.run(task.task_id, task.question)

    classifier = FailureClassifier()
    classified = classifier.classify(trace, task, trace.final_answer)
    propagator = PropagationAnalyzer()
    propagation = propagator.analyze(classified)
    causal = CausalReplay()
    attribution = causal.attribute(classified, task)

    return trace, classified, propagation, attribution, llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--models", nargs="*",
                    default=["gpt-4o-mini", "gpt-4o", "deepseek-chat",
                             "deepseek-r1", "qwen3-max"])
    ap.add_argument("--output", default="results_real")
    ap.add_argument("--n-per-domain", type=int, default=16,
                    help="tasks per domain (16 -> 80 total)")
    ap.add_argument("--dsbench", action="store_true", default=True,
                    help="include DSBench MC comparison subset")
    ap.add_argument("--no-dsbench", dest="dsbench", action="store_false")
    ap.add_argument("--no-push", action="store_true",
                    help="skip GitHub push on completion")
    args = ap.parse_args()

    # verify API key present
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Configure in ~/.bashrc.")
        return 1

    os.makedirs(args.output, exist_ok=True)
    # build full benchmark: 80 generated tasks + 15 DSBench MC
    taskset = build_full_benchmark(n_per_domain=args.n_per_domain)
    n_main = len(taskset)
    if args.dsbench:
        dsbench_tasks = build_dsbench_mc_subset(
            dsbench_data_path="/data/lab/DSBench/data_analysis/data.json",
            n_questions=15,
        )
        taskset = TaskSet(tasks=list(taskset.tasks) + list(dsbench_tasks.tasks))
    n_total = len(taskset)
    print(f"Benchmark: {n_main} generated + {n_total - n_main} DSBench MC = {n_total} tasks")
    all_variants = []
    for m in args.models:
        all_variants.append(m)
        all_variants.append(f"{m}+verifier")

    per_model_runs = {}
    per_model_aggregate = {}

    for variant in all_variants:
        base_model = variant.replace("+verifier", "")
        use_verifier = variant.endswith("+verifier")
        _, display, pin, pout = MODEL_CONFIGS[base_model]

        print(f"\n{'='*60}")
        print(f"Running: {variant} ({display})")
        print(f"{'='*60}")

        runs = []
        all_classified, all_propagations, all_traces = [], [], []
        all_correct, all_causal, invalid_tokens = [], [], []

        for task in taskset:
            for rep in range(args.repeats):
                t0 = time.time()
                try:
                    trace, classified, propagation, attribution, llm = run_one(
                        variant, task, args.max_steps, use_verifier
                    )
                    status = "OK"
                except Exception as e:
                    print(f"  ERROR on {task.task_id} rep {rep}: {e}")
                    traceback.print_exc()
                    # record as a failed run with empty trace
                    trace = AgentTrace(task_id=task.task_id, model=display)
                    classified = ClassifiedTrace(
                        trace=trace,
                        classification=FailureClassifier().classify(
                            trace, task, None
                        ).classification,
                        task_correct=False,
                    )
                    propagation = PropagationReport(-1, -1, 0, False, False, [])
                    attribution = CausalAttribution(-1, "", False, 0.0, str(e))
                    llm = make_backend(MODEL_CONFIGS[base_model][0], pin, pout)
                    status = "ERR"

                elapsed = time.time() - t0
                all_traces.append(trace)
                all_correct.append(classified.task_correct)
                all_classified.append(classified)
                all_propagations.append(propagation)
                all_causal.append(attribution.replay_succeeded)
                invalid_tokens.append(getattr(llm, "invalid_token_count", 0))

                run_rec = {
                    "task_id": task.task_id,
                    "rep": rep,
                    "model": display,
                    "variant": variant,
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
                    "final_answer": str(trace.final_answer)[:200],
                }
                runs.append(run_rec)
                mark = "+" if classified.task_correct else ("~" if classified.is_silent_failure else "x")
                print(f"  [{mark}] {task.task_id} r{rep} | stage={run_rec['stage']} "
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

        agg = compute_aggregate(variant, per_run_metrics, pin, pout)
        per_model_aggregate[variant] = agg
        per_model_runs[variant] = runs

        fm = compute_failure_metrics(all_classified, all_propagations, all_causal)
        em = compute_economics(all_traces, all_correct, pin, pout, invalid_tokens)

        with open(os.path.join(args.output, f"{variant}_detail.json"), "w") as f:
            json.dump({
                "runs": runs,
                "failure_metrics": fm.as_dict(),
                "economics": em.as_dict(),
                "aggregate": agg.as_dict(),
            }, f, indent=2, ensure_ascii=False)

        print(f"\n  --- {variant} summary ---")
        print(f"  success_rate: {agg.success_rate_mean:.4f} +/- {agg.success_rate_std:.4f}")
        print(f"  silent_fail:  {agg.sfr_mean:.4f}")
        print(f"  tok/success:  {agg.token_per_success_mean:.0f}")
        print(f"  $/success:    {agg.cost_per_success_mean:.4f}")

    # paired comparisons
    comparisons = {}
    for variant in all_variants:
        if variant.endswith("+verifier"):
            base = variant.replace("+verifier", "")
            if base in per_model_runs:
                base_series = _per_task_success(per_model_runs[base])
                ver_series = _per_task_success(per_model_runs[variant])
                t, p = paired_ttest(base_series, ver_series)
                d = cohens_d(base_series, ver_series)
                comparisons[f"{base}_vs_{variant}"] = {
                    "delta_success_rate": (sum(ver_series) - sum(base_series)) / len(base_series)
                        if base_series else 0,
                    "t_stat": round(t, 4),
                    "p_value": round(p, 6),
                    "cohens_d": round(d, 4),
                }

    summary = {
        "experiment": "real_models",
        "config": {
            "n_repeats": args.repeats, "max_steps": args.max_steps,
            "n_tasks": len(taskset), "models": args.models,
            "endpoint": os.environ.get("OPENAI_BASE_URL", ""),
        },
        "aggregate": {m: a.as_dict() for m, a in per_model_aggregate.items()},
        "comparisons": comparisons,
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    for m, a in per_model_aggregate.items():
        print(f"  {m}: SR={a.success_rate_mean:.4f} SFR={a.sfr_mean:.4f} "
              f"tok/s={a.token_per_success_mean:.0f}")
    if comparisons:
        print("\n  Paired (baseline vs verifier):")
        for k, c in comparisons.items():
            print(f"    {k}: delta={c['delta_success_rate']:+.4f} p={c['p_value']} d={c['cohens_d']}")

    # auto-push to GitHub
    if not args.no_push:
        print("\n=== Auto-pushing results to GitHub ===")
        script = Path(__file__).parent / "scripts" / "auto_push_results.sh"
        try:
            subprocess.run(
                ["bash", str(script), args.output,
                 f"Add real-model experiment results ({len(args.models)} models x {args.repeats} repeats)"],
                check=True,
            )
            print("GitHub push successful.")
        except subprocess.CalledProcessError as e:
            print(f"GitHub push failed (exit {e.returncode}), results saved locally.")
        except FileNotFoundError:
            print("auto_push_results.sh not found, skipping push.")
    else:
        print("--no-push specified, skipping GitHub push.")

    return 0


def _per_task_success(runs):
    by_task = {}
    for r in runs:
        by_task.setdefault(r["task_id"], []).append(1 if r["correct"] else 0)
    return [sum(v) / len(v) for v in by_task.values()]


if __name__ == "__main__":
    sys.exit(main())
