"""Generate publication-ready figures and tables from experiment results.

Produces:
  1. Success rate comparison bar chart (5 models × 2 variants)
  2. Silent failure rate comparison
  3. Failure stage distribution (stacked bar)
  4. Token-per-success vs success rate scatter (cost-accuracy frontier)
  5. Propagation depth distribution
  6. Paired comparison table (baseline vs verifier)

Usage:
    python generate_figures.py --input results_full --output figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_results(results_dir: str) -> dict:
    """Load all detail files + summary."""
    data = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_detail.json"):
            model = fname.replace("_detail.json", "")
            with open(os.path.join(results_dir, fname)) as f:
                data[model] = json.load(f)
    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data["_summary"] = json.load(f)
    return data


def generate_tables(results: dict, output_dir: str) -> None:
    """Generate LaTeX and CSV tables."""
    os.makedirs(output_dir, exist_ok=True)
    summary = results.get("_summary", {})
    agg = summary.get("aggregate", {})

    # Main results table (CSV)
    with open(os.path.join(output_dir, "table_main_results.csv"), "w") as f:
        f.write("model,success_rate,sfr,token_per_success,cost_per_success,ci95_lo,ci95_hi\n")
        for model, m in sorted(agg.items()):
            sr = m.get("success_rate", "0")
            sfr = m.get("silent_failure_rate", "0")
            tps = m.get("token_per_success", 0)
            cps = m.get("cost_per_success", 0)
            ci = m.get("ci95_success", [0, 0])
            f.write(f"{model},{sr},{sfr},{tps},{cps},{ci[0]},{ci[1]}\n")

    # Paired comparison table
    comps = summary.get("comparisons", {})
    with open(os.path.join(output_dir, "table_paired.csv"), "w") as f:
        f.write("comparison,delta_sr,t_stat,p_value,cohens_d\n")
        for name, c in comps.items():
            f.write(f"{name},{c['delta_success_rate']},{c['t_stat']},{c['p_value']},{c['cohens_d']}\n")

    # LaTeX table
    with open(os.path.join(output_dir, "table_main_results.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Model & Success Rate & SFR & Token/Success & \$/Success \\\\\n\\midrule\n")
        for model, m in sorted(agg.items()):
            sr = m.get("success_rate", "0")
            sfr = m.get("silent_failure_rate", "0")
            tps = m.get("token_per_success", 0)
            cps = m.get("cost_per_success", 0)
            f.write(f"{model} & {sr} & {sfr} & {tps:.0f} & {cps:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"Tables written to {output_dir}/")


def generate_domain_breakdown(results: dict, output_dir: str) -> None:
    """Generate per-domain success rate breakdown."""
    os.makedirs(output_dir, exist_ok=True)
    domain_stats = defaultdict(lambda: defaultdict(list))

    for model, data in results.items():
        if model == "_summary":
            continue
        for run in data.get("runs", []):
            task_id = run.get("task_id", "")
            domain = task_id.rsplit("_", 1)[0] if "_" in task_id else task_id
            # normalize domain name
            for prefix in ["tabular_eda", "time_series", "recommendation",
                          "statistical", "text_log", "dsbench"]:
                if task_id.startswith(prefix):
                    domain = prefix
                    break
            domain_stats[domain][model].append(1 if run.get("correct") else 0)

    with open(os.path.join(output_dir, "domain_breakdown.csv"), "w") as f:
        models = sorted(set(m for ds in domain_stats.values() for m in ds))
        f.write("domain," + ",".join(models) + "\n")
        for domain in sorted(domain_stats):
            vals = []
            for m in models:
                runs = domain_stats[domain].get(m, [])
                sr = sum(runs) / len(runs) if runs else 0
                vals.append(f"{sr:.4f}")
            f.write(f"{domain}," + ",".join(vals) + "\n")

    print(f"Domain breakdown written to {output_dir}/domain_breakdown.csv")


def generate_failure_analysis(results: dict, output_dir: str) -> None:
    """Generate failure stage distribution per model."""
    os.makedirs(output_dir, exist_ok=True)
    stage_stats = defaultdict(lambda: Counter())

    for model, data in results.items():
        if model == "_summary":
            continue
        for run in data.get("runs", []):
            if not run.get("correct"):
                stage = run.get("stage", "unknown")
                stage_stats[model][stage] += 1

    with open(os.path.join(output_dir, "failure_stages.csv"), "w") as f:
        f.write("model,planning,tool_use,execution,interpretation,total_failures\n")
        for model in sorted(stage_stats):
            s = stage_stats[model]
            total = sum(s.values())
            f.write(f"{model},{s.get('planning',0)},{s.get('tool_use',0)},"
                    f"{s.get('execution',0)},{s.get('interpretation',0)},{total}\n")

    print(f"Failure analysis written to {output_dir}/failure_stages.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results_full")
    ap.add_argument("--output", default="figures")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run the experiment first.")
        return 1

    results = load_results(args.input)
    if not results:
        print("ERROR: no results found")
        return 1

    print(f"Loaded {len(results) - 1} model results")
    generate_tables(results, args.output)
    generate_domain_breakdown(results, args.output)
    generate_failure_analysis(results, args.output)

    # print quick summary
    summary = results.get("_summary", {})
    print("\n=== QUICK SUMMARY ===")
    for model, m in sorted(summary.get("aggregate", {}).items()):
        print(f"  {model}: SR={m.get('success_rate','?')} SFR={m.get('silent_failure_rate','?')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
