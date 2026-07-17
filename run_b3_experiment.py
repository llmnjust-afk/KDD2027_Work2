#!/usr/bin/env python3
"""B3 experiment: execution-consistency verifier.

Runs each (model, task) TWICE independently, compares answers. If they
disagree -> silent failure suspected -> run a third "reconciliation" attempt
that presents both answers and asks the agent to pick the correct one.

This replaces the fragile rule-based verifier with a self-consistency
mechanism that does NOT depend on ground-truth path keywords, eliminating
false positives.

Only runs B3 variants (gpt-4o-mini+consistency, gpt-4o+consistency, etc.)
so it can be run after the main experiment without re-running baselines.

Usage:
    python run_b3_experiment.py --repeats 3 --models gpt-4o-mini gpt-4o
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
from agentfail.benchmark.dsbench_adapter import build_dsbench_mc_subset
from agentfail.benchmark.tasks import TaskSet
from agentfail.agent.sandbox import CodeSandbox
from agentfail.agent.react_agent import ReActAgent, AgentTrace
from agentfail.diagnosis.classifier import FailureClassifier
from agentfail.diagnosis.propagation import PropagationAnalyzer
from agentfail.diagnosis.causality import CausalReplay
from agentfail.llm.openai_backend import OpenAIBackend
from agentfail.method.consistency_verifier import ExecutionConsistencyVerifier
from agentfail.metrics.failure_metrics import compute_failure_metrics
from agentfail.metrics.economics import compute_economics
from agentfail.metrics.aggregate import compute_aggregate, paired_ttest, cohens_d
import tempfile


MODEL_CONFIGS = {
    "gpt-4o-mini": ("gpt-4o-mini", "gpt-4o-mini", 0.15, 0.60),
    "gpt-4o": ("gpt-4o", "gpt-4o", 2.5, 10.0),
    "deepseek-chat": ("deepseek-chat", "deepseek-v3", 0.14, 0.28),
    "deepseek-r1": ("deepseek-reasoner", "deepseek-r1", 0.55, 2.19),
    "qwen3-max": ("qwen3-max-2026-01-23", "qwen3-max-2026", 0.5, 1.5),
}


def make_backend(api_name: str, pin: float, pout: float, temp: float = 0.0) -> OpenAIBackend:
    return OpenAIBackend(
        model=api_name,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
        price_in=pin, price_out=pout,
        temperature=temp, max_tokens=2048,
    )


def run_consistency_variant(model_key: str, task, max_steps: int):
    """Run B3: two independent runs + reconciliation if disagreement.

    Returns (final_trace, final_answer, total_tokens, consistency_info).
    """
    api_name, display, pin, pout = MODEL_CONFIGS[model_key]

    # Run 1 (primary) at temperature 0
    llm1 = make_backend(api_name, pin, pout, temp=0.0)
    sb1 = CodeSandbox(tempfile.mkdtemp())
    task.prepare_data(sb1.workdir)
    agent1 = ReActAgent(llm=llm1, sandbox=sb1, max_steps=max_steps)
    trace1 = agent1.run(task.task_id, task.question)

    # If primary produced no answer, it's a loud failure; no consistency check needed
    if trace1.final_answer is None:
        return trace1, None, trace1.total_tokens.total_tokens, {
            "primary_answer": None, "secondary_answer": None,
            "consistent": False, "reconciled": False,
        }

    # Run 2 (secondary) at temperature 0.3 to introduce variation
    llm2 = make_backend(api_name, pin, pout, temp=0.3)
    sb2 = CodeSandbox(tempfile.mkdtemp())
    task.prepare_data(sb2.workdir)
    agent2 = ReActAgent(llm=llm2, sandbox=sb2, max_steps=max_steps)
    trace2 = agent2.run(task.task_id, task.question)

    total_tok = trace1.total_tokens.total_tokens + trace2.total_tokens.total_tokens

    verifier = ExecutionConsistencyVerifier()
    verdict = verifier.check(trace1, trace2)

    if not verdict.needs_retry:
        # answers agree -> accept primary
        return trace1, trace1.final_answer, total_tok, {
            "primary_answer": verdict.primary_answer,
            "secondary_answer": verdict.secondary_answer,
            "consistent": True, "reconciled": False,
        }

    # disagreement -> reconciliation run: present both answers, ask agent to pick
    llm3 = make_backend(api_name, pin, pout, temp=0.0)
    sb3 = CodeSandbox(tempfile.mkdtemp())
    task.prepare_data(sb3.workdir)
    agent3 = ReActAgent(llm=llm3, sandbox=sb3, max_steps=max_steps)
    # augment the task description with the disagreement context
    aug_question = (
        f"{task.question}\n\n"
        f"NOTE: Two previous attempts gave different answers: "
        f"'{verdict.primary_answer}' and '{verdict.secondary_answer}'. "
        f"One may be wrong due to a silent error. Re-analyze the data carefully "
        f"and determine the correct answer."
    )
    trace3 = agent3.run(task.task_id, aug_question)
    total_tok += trace3.total_tokens.total_tokens

    # the reconciled trace is what we report
    # but we attach consistency metadata
    return trace3, trace3.final_answer, total_tok, {
        "primary_answer": verdict.primary_answer,
        "secondary_answer": verdict.secondary_answer,
        "consistent": False, "reconciled": True,
        "reconciled_answer": trace3.final_answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--models", nargs="*",
                    default=["gpt-4o-mini", "gpt-4o", "deepseek-chat",
                             "deepseek-r1", "qwen3-max"])
    ap.add_argument("--n-per-domain", type=int, default=16)
    ap.add_argument("--output", default="results_b3")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        return 1

    os.makedirs(args.output, exist_ok=True)
    taskset = build_full_benchmark(n_per_domain=args.n_per_domain)
    dsbench = build_dsbench_mc_subset(
        "/data/lab/DSBench/data_analysis/data.json", 15)
    taskset = TaskSet(tasks=list(taskset.tasks) + list(dsbench.tasks))
    n_total = len(taskset)
    print(f"B3 Experiment: {n_total} tasks, {len(args.models)} models, {args.repeats} repeats")

    # We need baseline results for paired comparison.
    # Load them from the main experiment if available.
    baseline_runs = {}
    baseline_dir = "results_full"
    if os.path.exists(baseline_dir):
        for model in args.models:
            path = os.path.join(baseline_dir, f"{model}_detail.json")
            if os.path.exists(path):
                with open(path) as f:
                    baseline_runs[model] = json.load(f).get("runs", [])
        print(f"Loaded baselines for: {list(baseline_runs.keys())}")

    per_model_runs = {}
    per_model_aggregate = {}

    for model_key in args.models:
        variant = f"{model_key}+consistency"
        _, display, pin, pout = MODEL_CONFIGS[model_key]

        print(f"\n{'='*60}")
        print(f"Running B3: {variant} ({display})")
        print(f"{'='*60}")

        runs = []
        all_classified, all_propagations, all_traces = [], [], []
        all_correct, all_causal = [], []

        for task in taskset:
            for rep in range(args.repeats):
                t0 = time.time()
                try:
                    trace, final_answer, total_tok, cons_info = run_consistency_variant(
                        model_key, task, args.max_steps
                    )
                    # build a minimal classified/propagation for metrics
                    clf = FailureClassifier()
                    classified = clf.classify(trace, task, final_answer)
                    prop = PropagationAnalyzer().analyze(classified)
                    causal = CausalReplay().attribute(classified, task)
                    status = "OK"
                except Exception as e:
                    print(f"  ERROR on {task.task_id} rep {rep}: {e}")
                    traceback.print_exc()
                    trace = AgentTrace(task_id=task.task_id, model=display)
                    classified = clf.classify(trace, task, None)
                    prop = PropagationReport(-1, -1, 0, False, False, [])
                    causal = CausalAttribution(-1, "", False, 0.0, str(e))
                    total_tok = 0
                    cons_info = {"consistent": False, "reconciled": False,
                                 "primary_answer": None, "secondary_answer": None}
                    final_answer = None
                    status = "ERR"

                elapsed = time.time() - t0
                all_traces.append(trace)
                all_correct.append(classified.task_correct)
                all_classified.append(classified)
                all_propagations.append(prop)
                all_causal.append(causal.replay_succeeded)

                # patch trace token count for economics (B3 uses 2-3 runs)
                trace.total_tokens = type(trace.total_tokens)(
                    prompt_tokens=total_tok // 2,
                    completion_tokens=total_tok // 2,
                )

                run_rec = {
                    "task_id": task.task_id, "rep": rep,
                    "model": display, "variant": variant,
                    "correct": classified.task_correct,
                    "stage": classified.classification.stage.value,
                    "category": classified.classification.category.value,
                    "is_silent": classified.is_silent_failure,
                    "propagation_depth": prop.propagation_depth,
                    "recovered": prop.was_recovered,
                    "causal_attributed": causal.replay_succeeded,
                    "tokens": total_tok,
                    "elapsed_s": round(elapsed, 2),
                    "status": status,
                    "final_answer": str(final_answer)[:200] if final_answer else None,
                    "consistent": cons_info.get("consistent", False),
                    "reconciled": cons_info.get("reconciled", False),
                    "primary_answer": str(cons_info.get("primary_answer", ""))[:100],
                    "secondary_answer": str(cons_info.get("secondary_answer", ""))[:100],
                }
                runs.append(run_rec)
                mark = "+" if classified.task_correct else ("~" if classified.is_silent_failure else "x")
                cons_mark = "=" if cons_info.get("consistent") else ("R" if cons_info.get("reconciled") else "?")
                print(f"  [{mark}|{cons_mark}] {task.task_id} r{rep} | tok={total_tok} {elapsed:.1f}s "
                      f"{'(reconciled)' if cons_info.get('reconciled') else ''}")

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
        em = compute_economics(all_traces, all_correct, pin, pout)

        with open(os.path.join(args.output, f"{variant}_detail.json"), "w") as f:
            json.dump({
                "runs": runs, "failure_metrics": fm.as_dict(),
                "economics": em.as_dict(), "aggregate": agg.as_dict(),
            }, f, indent=2, ensure_ascii=False)

        print(f"\n  --- {variant} summary ---")
        print(f"  success_rate: {agg.success_rate_mean:.4f} +/- {agg.success_rate_std:.4f}")
        print(f"  silent_fail:  {agg.sfr_mean:.4f}")
        print(f"  tok/success:  {agg.token_per_success_mean:.0f}")
        n_reconciled = sum(1 for r in runs if r.get("reconciled"))
        n_consistent = sum(1 for r in runs if r.get("consistent"))
        print(f"  consistent: {n_consistent}/{len(runs)} | reconciled: {n_reconciled}/{len(runs)}")

    # paired comparisons: baseline (from main experiment) vs B3
    comparisons = {}
    for model_key in args.models:
        variant = f"{model_key}+consistency"
        if model_key in baseline_runs and variant in per_model_runs:
            base_series = _per_task_success(baseline_runs[model_key])
            b3_series = _per_task_success(per_model_runs[variant])
            t, p = paired_ttest(base_series, b3_series)
            d = cohens_d(base_series, b3_series)
            comparisons[f"{model_key}_vs_{variant}"] = {
                "delta_success_rate": (sum(b3_series) - sum(base_series)) / len(base_series)
                    if base_series else 0,
                "t_stat": round(t, 4),
                "p_value": round(p, 6),
                "cohens_d": round(d, 4),
            }

    summary = {
        "experiment": "b3_consistency",
        "config": {
            "n_repeats": args.repeats, "max_steps": args.max_steps,
            "n_tasks": n_total, "models": args.models,
            "endpoint": os.environ.get("OPENAI_BASE_URL", ""),
        },
        "aggregate": {m: a.as_dict() for m, a in per_model_aggregate.items()},
        "comparisons": comparisons,
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("B3 EXPERIMENT COMPLETE")
    print("=" * 60)
    for m, a in per_model_aggregate.items():
        print(f"  {m}: SR={a.success_rate_mean:.4f} SFR={a.sfr_mean:.4f} "
              f"tok/s={a.token_per_success_mean:.0f}")
    if comparisons:
        print("\n  Paired (baseline vs B3 consistency):")
        for k, c in comparisons.items():
            sig = "***" if c["p_value"] < 0.01 else "**" if c["p_value"] < 0.05 else "*" if c["p_value"] < 0.1 else "ns"
            print(f"    {k}: delta={c['delta_success_rate']:+.4f} p={c['p_value']} {sig} d={c['cohens_d']}")

    if not args.no_push:
        print("\n=== Auto-pushing B3 results to GitHub ===")
        script = Path(__file__).parent / "scripts" / "auto_push_results.sh"
        try:
            subprocess.run(
                ["bash", str(script), args.output,
                 f"Add B3 consistency-verifier experiment results ({len(args.models)} models)"],
                check=True,
            )
            print("GitHub push successful.")
        except subprocess.CalledProcessError as e:
            print(f"GitHub push failed (exit {e.returncode}), results saved locally.")
    return 0


def _per_task_success(runs):
    by_task = {}
    for r in runs:
        by_task.setdefault(r["task_id"], []).append(1 if r["correct"] else 0)
    return [sum(v) / len(v) for v in by_task.values()]


# needed for the error fallback
from agentfail.diagnosis.propagation import PropagationReport
from agentfail.diagnosis.causality import CausalAttribution


if __name__ == "__main__":
    sys.exit(main())
