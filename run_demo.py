#!/usr/bin/env python3
"""End-to-end demo: run the full failure-diagnosis benchmark with zero API cost.

Runs the MockLLM grid (weak/medium/strong +/- verifier) over the default
benchmark, then prints the aggregate report, failure metrics, token economics,
and the paired comparison (baseline vs verifier). Produces results/summary.json.

Usage:
    python run_demo.py
    python run_demo.py --repeats 5 --models mock-weak mock-strong mock-strong+verifier
"""

from __future__ import annotations

import argparse
import json
import sys

from agentfail.eval.runner import run_experiment, RunConfig


def main():
    ap = argparse.ArgumentParser(description="AgentFail end-to-end demo")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--output", default="results")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    config = RunConfig(
        models=args.models or [
            "mock-weak", "mock-medium", "mock-strong",
            "mock-medium+verifier", "mock-strong+verifier",
        ],
        n_repeats=args.repeats,
        max_steps=args.max_steps,
        output_dir=args.output,
    )

    print("=" * 70)
    print("AgentFail: Failure-Diagnosis Benchmark for Data-Science Agents")
    print("=" * 70)
    print(f"Models: {config.models}")
    print(f"Repeats: {config.n_repeats} | Max steps: {config.max_steps}")
    print("-" * 70)

    summary = run_experiment(config)

    print("\n### AGGREGATE RESULTS ###")
    for model, agg in summary["aggregate"].items():
        print(f"\n[{model}]")
        print(f"  success_rate      : {agg['success_rate']}")
        print(f"  silent_failure    : {agg['silent_failure_rate']}")
        print(f"  token_per_success : {agg['token_per_success']}")
        print(f"  cost_per_success  : {agg['cost_per_success']}")
        print(f"  CI95 (success)    : {agg['ci95_success']}")

    print("\n### PAIRED COMPARISONS (baseline vs +verifier) ###")
    if not summary["comparisons"]:
        print("  (none)")
    for name, c in summary["comparisons"].items():
        print(f"\n  {name}")
        print(f"    delta_success_rate : {c['delta_success_rate']:+.4f}")
        print(f"    t_stat             : {c['t_stat']}")
        print(f"    p_value            : {c['p_value']}")
        print(f"    cohen's_d          : {c['cohens_d']}")

    print("\n" + "=" * 70)
    print(f"Full results written to {config.output_dir}/summary.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
