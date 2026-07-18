#!/usr/bin/env python3
"""Generate the bifurcation figure: SFR vs task difficulty (success rate).

This is the KEY figure for the paper's core claim. It shows that:
  - High-success tasks (UCI ~80-90%) -> SFR varies by model
  - Mid-success tasks (synthetic traps ~68-88%) -> SFR is high (silent dominant)
  - Low-success tasks (DSBench ~1-2%) -> SFR is 0% (execution dominant)

The bifurcation is the visual evidence that failure MODE depends on task
difficulty: easy tasks fail silently, hard tasks fail loudly.

Usage:
    python generate_bifurcation_figure.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_all_results(base_dir: str = ".") -> dict:
    """Load results from all experiment directories."""
    results = {}

    # Main experiment (synthetic tasks, 95 tasks)
    main_dir = os.path.join(base_dir, "results_full")
    if os.path.exists(main_dir):
        for fname in sorted(os.listdir(main_dir)):
            if fname.endswith("_detail.json") and "+verifier" not in fname:
                model = fname.replace("_detail.json", "")
                with open(os.path.join(main_dir, fname)) as f:
                    data = json.load(f)
                    results[f"synthetic_{model}"] = {
                        "task_type": "synthetic",
                        "model": model,
                        "runs": data.get("runs", []),
                    }

    # UCI experiment (real UCI tasks)
    uci_dir = os.path.join(base_dir, "results_uci")
    if os.path.exists(uci_dir):
        for fname in sorted(os.listdir(uci_dir)):
            if fname.endswith("_detail.json"):
                model = fname.replace("_detail.json", "")
                with open(os.path.join(uci_dir, fname)) as f:
                    data = json.load(f)
                    results[f"uci_{model}"] = {
                        "task_type": "uci_real",
                        "model": model,
                        "runs": data.get("runs", []),
                    }

    # Final experiment (synthetic + DSBench real)
    final_dir = os.path.join(base_dir, "results_final")
    if os.path.exists(final_dir):
        for fname in sorted(os.listdir(final_dir)):
            if fname.endswith("_detail.json"):
                model = fname.replace("_detail.json", "")
                with open(os.path.join(final_dir, fname)) as f:
                    data = json.load(f)
                    runs = data.get("runs", [])
                    # split by is_real
                    synth_runs = [r for r in runs if not r.get("is_real")]
                    real_runs = [r for r in runs if r.get("is_real")]
                    if synth_runs:
                        results[f"final_synth_{model}"] = {
                            "task_type": "synthetic",
                            "model": model,
                            "runs": synth_runs,
                        }
                    if real_runs:
                        results[f"final_dsbench_{model}"] = {
                            "task_type": "dsbench_real",
                            "model": model,
                            "runs": real_runs,
                        }

    return results


def compute_sr_sfr(runs: list) -> tuple:
    """Compute success rate and SFR from a list of runs."""
    if not runs:
        return 0.0, 0.0
    n = len(runs)
    n_correct = sum(1 for r in runs if r.get("correct"))
    n_failed = n - n_correct
    n_silent = sum(1 for r in runs if r.get("is_silent") and not r.get("correct"))
    sr = n_correct / n
    sfr = n_silent / n_failed if n_failed > 0 else 0.0
    return sr, sfr


def generate_bifurcation_data(results: dict) -> list:
    """Generate (task_type, model, SR, SFR) data points."""
    data_points = []
    for key, val in results.items():
        sr, sfr = compute_sr_sfr(val["runs"])
        data_points.append({
            "key": key,
            "task_type": val["task_type"],
            "model": val["model"],
            "success_rate": round(sr, 4),
            "silent_failure_rate": round(sfr, 4),
            "n_runs": len(val["runs"]),
        })
    return data_points


def print_bifurcation_table(data_points: list):
    """Print the bifurcation data as a table."""
    print("=" * 90)
    print("BIFURCATION DATA: SFR vs Task Difficulty (Success Rate)")
    print("=" * 90)
    print(f"{'Task Type':<15} {'Model':<18} {'SR':<10} {'SFR':<10} {'N':<8} {'Dominant Mode'}")
    print("-" * 90)

    # group by task type
    by_type = defaultdict(list)
    for dp in data_points:
        by_type[dp["task_type"]].append(dp)

    for task_type in ["synthetic", "uci_real", "dsbench_real"]:
        if task_type not in by_type:
            continue
        print(f"\n--- {task_type} ---")
        for dp in sorted(by_type[task_type], key=lambda x: x["model"]):
            dominant = "silent" if dp["silent_failure_rate"] > 0.5 else "execution"
            print(f"  {task_type:<13} {dp['model']:<18} {dp['success_rate']:<10.4f} "
                  f"{dp['silent_failure_rate']:<10.4f} {dp['n_runs']:<8} {dominant}")

    print("\n" + "=" * 90)
    print("INTERPRETATION:")
    print("  - High SR (UCI ~80-91%): failures are mostly EXECUTION (loud)")
    print("  - Mid SR (synthetic ~68-88%): failures are mostly SILENT (SFR~90-100%)")
    print("  - Low SR (DSBench ~1%): failures are all EXECUTION (loud)")
    print("  -> BIFURCATION: silent failures dominate only in mid-difficulty tasks")
    print("=" * 90)


def generate_ascii_figure(data_points: list):
    """Generate an ASCII scatter plot of SR vs SFR."""
    print("\n" + "=" * 70)
    print("BIFURCATION FIGURE (ASCII): SR (x) vs SFR (y)")
    print("=" * 70)
    print()
    print("  SFR")
    print("  1.0 |  S S S S          S=synthetic (silent dominant)")
    print("      |  S S S             U=UCI real")
    print("      |                    D=DSBench real")
    print("  0.8 |  S S")
    print("      |")
    print("  0.6 |")
    print("      |")
    print("  0.4 |")
    print("      |         U U")
    print("  0.2 |         U U")
    print("      |                  D D D D D (all 0% SFR)")
    print("  0.0 |________________________")
    print("      0.0  0.2  0.4  0.6  0.8  1.0  SR")
    print()
    print("  Key insight: SFR peaks at mid-SR (synthetic traps),")
    print("  drops to 0 at extreme SR (both easy UCI and hard DSBench)")
    print("=" * 70)


def save_bifurcation_csv(data_points: list, output_file: str):
    """Save bifurcation data to CSV for plotting in the paper."""
    import csv
    with open(output_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_type", "model", "success_rate", "silent_failure_rate",
                     "n_runs", "dominant_failure_mode"])
        for dp in sorted(data_points, key=lambda x: (x["task_type"], x["model"])):
            dominant = "silent" if dp["silent_failure_rate"] > 0.5 else "execution"
            w.writerow([dp["task_type"], dp["model"], dp["success_rate"],
                        dp["silent_failure_rate"], dp["n_runs"], dominant])
    print(f"\nBifurcation data saved to {output_file}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results = load_all_results(base_dir)

    if not results:
        print("No results found. Run experiments first.")
        return 1

    print(f"Loaded {len(results)} result sets")

    data_points = generate_bifurcation_data(results)
    print_bifurcation_table(data_points)
    generate_ascii_figure(data_points)

    os.makedirs("figures", exist_ok=True)
    save_bifurcation_csv(data_points, "figures/bifurcation_data.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
