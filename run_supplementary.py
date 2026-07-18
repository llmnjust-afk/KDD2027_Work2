#!/usr/bin/env python3
"""Supplementary experiments to fill empty cells in Table 3.

1. UCI +verifier: 5 models x 23 tasks x 3 repeats = 345 runs
2. DSBench baseline: 3 models (deepseek-chat, deepseek-r1, qwen3-max) x 200 tasks x 3 repeats = 1800 runs

Output: results_supplementary/
"""
import json, os, sys, time, tempfile, traceback
sys.path.insert(0, '.')

from agentfail.benchmark.uci_tasks import build_uci_benchmark
from agentfail.benchmark.dsbench_real_adapter import build_dsbench_real_subset
from agentfail.benchmark.tasks import TaskSet
from agentfail.agent.sandbox import CodeSandbox
from agentfail.agent.react_agent import ReActAgent, AgentTrace
from agentfail.diagnosis.classifier import FailureClassifier
from agentfail.diagnosis.propagation import PropagationAnalyzer, PropagationReport
from agentfail.diagnosis.causality import CausalReplay
from agentfail.llm.openai_backend import OpenAIBackend
from agentfail.method.rule_verifier import FailureAwareVerifier
from agentfail.metrics.aggregate import compute_aggregate
import argparse

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

def run_one(model_key, task, max_steps, use_verifier=False):
    api_name, display, pin, pout = MODEL_CONFIGS[model_key]
    llm = make_backend(api_name, pin, pout)
    workdir = tempfile.mkdtemp(prefix=f"sup_{task.task_id}_")
    sandbox = CodeSandbox(workdir)
    task.prepare_data(workdir)
    verifier = FailureAwareVerifier(gt_path_keywords=task.gt_path) if use_verifier else None
    agent = ReActAgent(llm=llm, sandbox=sandbox, max_steps=max_steps, verifier=verifier)
    trace = agent.run(task.task_id, task.question)
    classified = FailureClassifier().classify(trace, task, trace.final_answer)
    propagation = PropagationAnalyzer().analyze(classified)
    causal = CausalReplay().attribute(classified, task)
    return trace, classified, propagation, causal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results_supplementary")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # === Part 1: UCI +verifier ===
    print("="*60)
    print("PART 1: UCI +verifier (5 models x 23 tasks x 3 repeats)")
    print("="*60)
    uci_tasks = build_uci_benchmark()
    uci_verifier_runs = []

    for model_key in MODEL_CONFIGS:
        _, display, pin, pout = MODEL_CONFIGS[model_key]
        print(f"\n--- UCI+verifier: {model_key} ({display}) ---")
        for task in uci_tasks:
            for rep in range(args.repeats):
                t0 = time.time()
                try:
                    trace, classified, propagation, causal = run_one(model_key, task, args.max_steps, use_verifier=True)
                    status = "OK"
                except Exception as e:
                    print(f"  ERROR {task.task_id} r{rep}: {e}")
                    trace = AgentTrace(task_id=task.task_id, model=display)
                    from agentfail.diagnosis.taxonomy import FailureStage as FS, FailureCategory as FC; from agentfail.diagnosis.classifier import FailureClassification as FCls, ClassifiedTrace as CT; classified = CT(trace=trace, classification=FCls(stage=FS.RUNTIME, category=FC.RUNTIME_EXCEPTION, step_index=0, is_silent=False, evidence="error"), task_correct=False)
                    propagation = PropagationReport(-1,-1,0,False,False,[])
                    status = "ERR"

                elapsed = time.time() - t0
                run_rec = {
                    "task_id": task.task_id, "rep": rep,
                    "model": display, "variant": f"{model_key}+verifier",
                    "domain": "uci_real", "correct": classified.task_correct,
                    "stage": classified.classification.stage.value,
                    "category": classified.classification.category.value,
                    "is_silent": classified.is_silent_failure,
                    "propagation_depth": propagation.propagation_depth,
                    "tokens": trace.total_tokens.total_tokens,
                    "elapsed_s": round(elapsed, 2),
                    "status": status,
                    "final_answer": str(trace.final_answer)[:200] if trace.final_answer else None,
                    "is_real": True,
                }
                uci_verifier_runs.append(run_rec)
                mark = "+" if classified.task_correct else ("~" if classified.is_silent_failure else "x")
                print(f"  [{mark}] {task.task_id} r{rep} | tok={run_rec['tokens']} {elapsed:.1f}s")

    # Save UCI verifier results
    with open(os.path.join(args.output, "uci_verifier_detail.json"), "w") as f:
        json.dump({"runs": uci_verifier_runs}, f, indent=2, ensure_ascii=False)
    print(f"\nUCI+verifier: {len(uci_verifier_runs)} runs saved")

    # === Part 2: DSBench baseline for 3 missing models ===
    print("\n" + "="*60)
    print("PART 2: DSBench baseline (3 models x 200 tasks x 3 repeats)")
    print("="*60)
    dsbench_tasks = build_dsbench_real_subset("/data/lab/DSBench", 200,
        preferred_task_ids=['00000043','00000033','00000038','00000010','00000035',
                           '00000030','00000016','00000005','00000006','00000007'])
    print(f"DSBench tasks: {len(dsbench_tasks)}")

    dsbench_missing_models = ["deepseek-chat", "deepseek-r1", "qwen3-max"]
    dsbench_runs = {}

    for model_key in dsbench_missing_models:
        _, display, pin, pout = MODEL_CONFIGS[model_key]
        print(f"\n--- DSBench: {model_key} ({display}) ---")
        runs = []
        for task in dsbench_tasks:
            for rep in range(args.repeats):
                t0 = time.time()
                try:
                    trace, classified, propagation, causal = run_one(model_key, task, args.max_steps, use_verifier=False)
                    status = "OK"
                except Exception as e:
                    print(f"  ERROR {task.task_id} r{rep}: {e}")
                    trace = AgentTrace(task_id=task.task_id, model=display)
                    from agentfail.diagnosis.taxonomy import FailureStage as FS, FailureCategory as FC; from agentfail.diagnosis.classifier import FailureClassification as FCls, ClassifiedTrace as CT; classified = CT(trace=trace, classification=FCls(stage=FS.RUNTIME, category=FC.RUNTIME_EXCEPTION, step_index=0, is_silent=False, evidence="error"), task_correct=False)
                    propagation = PropagationReport(-1,-1,0,False,False,[])
                    status = "ERR"

                elapsed = time.time() - t0
                run_rec = {
                    "task_id": task.task_id, "rep": rep,
                    "model": display, "variant": model_key,
                    "domain": "dsbench_real", "correct": classified.task_correct,
                    "stage": classified.classification.stage.value,
                    "category": classified.classification.category.value,
                    "is_silent": classified.is_silent_failure,
                    "propagation_depth": propagation.propagation_depth,
                    "tokens": trace.total_tokens.total_tokens,
                    "elapsed_s": round(elapsed, 2),
                    "status": status,
                    "final_answer": str(trace.final_answer)[:200] if trace.final_answer else None,
                    "is_real": True,
                }
                runs.append(run_rec)
                mark = "+" if classified.task_correct else ("~" if classified.is_silent_failure else "x")
                print(f"  [{mark}] {task.task_id} r{rep} | tok={run_rec['tokens']} {elapsed:.1f}s")

        dsbench_runs[model_key] = runs
        with open(os.path.join(args.output, f"dsbench_{model_key}_detail.json"), "w") as f:
            json.dump({"runs": runs}, f, indent=2, ensure_ascii=False)

    total_dsbench = sum(len(r) for r in dsbench_runs.values())
    print(f"\nDSBench: {total_dsbench} runs saved")

    # Auto-push
    print("\n=== Auto-pushing supplementary results ===")
    import subprocess
    from pathlib import Path
    script = Path(__file__).parent / "scripts" / "auto_push_results.sh"
    try:
        subprocess.run(["bash", str(script), args.output,
                       "Add supplementary experiment results (UCI verifier + DSBench 3 models)"],
                       check=True)
        print("Push successful.")
    except Exception as e:
        print(f"Push failed: {e}")

    print("\n=== SUPPLEMENTARY EXPERIMENTS COMPLETE ===")
    print(f"UCI+verifier: {len(uci_verifier_runs)} runs")
    print(f"DSBench: {total_dsbench} runs")
    print(f"Total: {len(uci_verifier_runs) + total_dsbench} runs")

if __name__ == "__main__":
    main()
