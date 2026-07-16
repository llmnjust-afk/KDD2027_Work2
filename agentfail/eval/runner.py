"""End-to-end evaluation pipeline.

For each (model, task, repeat):
  1. build the LLM backend (mock or real)
  2. run the ReAct agent, recording a full AgentTrace
  3. classify the failure (4-stage taxonomy)
  4. compute propagation depth
  5. run counterfactual causal replay
  6. collect token-economics
Then aggregate per-model across repeats with mean+/-std, bootstrap CI, and
paired t-tests between baseline and verifier-augmented variants.

The whole pipeline runs deterministically with MockLLM (no API key). Swapping
to real models is a one-line config change (see eval/config.py).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List

from ..agent.react_agent import ReActAgent, AgentTrace
from ..agent.sandbox import CodeSandbox
from ..benchmark.tasks import TaskSet, Task, build_default_benchmark
from ..diagnosis.classifier import FailureClassifier, ClassifiedTrace
from ..diagnosis.propagation import PropagationAnalyzer, PropagationReport
from ..diagnosis.causality import CausalReplay, CausalAttribution
from ..llm.base import LLMBackend
from ..llm.mock import MockLLM, MockSkill
from ..llm.openai_backend import OpenAIBackend
from ..method import FailureAwareVerifier
from ..metrics.failure_metrics import compute_failure_metrics, FailureMetrics
from ..metrics.economics import compute_economics, EconomicsMetrics
from ..metrics.aggregate import compute_aggregate, AggregateReport, paired_ttest, cohens_d
from .config import MODEL_GRID, ModelSpec


@dataclass
class RunConfig:
    models: List[str] = field(default_factory=lambda: [
        "mock-weak", "mock-medium", "mock-strong",
        "mock-medium+verifier", "mock-strong+verifier",
    ])
    n_repeats: int = 3
    max_steps: int = 6
    output_dir: str = "results"


def _make_llm(spec: ModelSpec) -> LLMBackend:
    if spec.backend == "mock":
        skill = MockSkill(spec.skill)
        return MockLLM(skill=skill, price_in=spec.price_in, price_out=spec.price_out)
    return OpenAIBackend(
        model=spec.name, api_key=spec.api_key, base_url=spec.base_url,
        price_in=spec.price_in, price_out=spec.price_out,
    )


def _spec_for(model_name: str) -> ModelSpec:
    if model_name in MODEL_GRID:
        return MODEL_GRID[model_name]
    # parse "+verifier" suffix
    base = model_name.replace("+verifier", "")
    if base in MODEL_GRID:
        s = MODEL_GRID[base]
        return ModelSpec(model_name, s.backend, s.skill, s.price_in, s.price_out)
    raise KeyError(f"unknown model: {model_name}")


def _run_one(
    model_name: str, task: Task, max_steps: int, use_verifier: bool
):
    spec = _spec_for(model_name)
    llm = _make_llm(spec)
    workdir = tempfile.mkdtemp(prefix=f"agentfail_{task.task_id}_")
    sandbox = CodeSandbox(workdir)
    task.prepare_data(workdir)

    verifier = None
    if use_verifier:
        verifier = FailureAwareVerifier(gt_path_keywords=task.gt_path)

    agent = ReActAgent(llm=llm, sandbox=sandbox, max_steps=max_steps, verifier=verifier)
    trace = agent.run(task.task_id, task.question)

    classifier = FailureClassifier()
    classified = classifier.classify(trace, task, trace.final_answer)

    propagator = PropagationAnalyzer()
    propagation = propagator.analyze(classified)

    causal = CausalReplay()
    attribution = causal.attribute(classified, task)

    return trace, classified, propagation, attribution, llm


def run_experiment(config: RunConfig = None) -> Dict:
    config = config or RunConfig()
    taskset = build_default_benchmark()
    os.makedirs(config.output_dir, exist_ok=True)

    # model_name -> list of per-run dicts
    per_model_runs: Dict[str, List[dict]] = {}
    # model_name -> aggregate over repeats
    per_model_aggregate: Dict[str, AggregateReport] = {}

    for model_name in config.models:
        use_verifier = model_name.endswith("+verifier")
        runs: List[dict] = []
        all_classified: List[ClassifiedTrace] = []
        all_propagations: List[PropagationReport] = []
        all_traces: List[AgentTrace] = []
        all_correct: List[bool] = []
        all_causal: List[bool] = []
        invalid_tokens: List[int] = []

        spec = _spec_for(model_name)

        for task in taskset:
            for rep in range(config.n_repeats):
                trace, classified, propagation, attribution, llm = _run_one(
                    model_name, task, config.max_steps, use_verifier
                )
                all_traces.append(trace)
                all_correct.append(classified.task_correct)
                all_classified.append(classified)
                all_propagations.append(propagation)
                all_causal.append(attribution.replay_succeeded)
                invalid_tokens.append(getattr(llm, "invalid_token_count", 0))

                runs.append({
                    "task_id": task.task_id,
                    "rep": rep,
                    "correct": classified.task_correct,
                    "stage": classified.classification.stage.value,
                    "category": classified.classification.category.value,
                    "is_silent": classified.is_silent_failure,
                    "propagation_depth": propagation.propagation_depth,
                    "recovered": propagation.was_recovered,
                    "causal_attributed": attribution.replay_succeeded,
                    "tokens": trace.total_tokens.total_tokens,
                })

        # per-run metrics (one summary row per repeat)
        per_run_metrics = []
        for rep in range(config.n_repeats):
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
                "cost_per_success": (tok * spec.price_in / 1e6) / nc if nc else float("inf"),
            })

        agg = compute_aggregate(
            model_name, per_run_metrics,
            price_in=spec.price_in, price_out=spec.price_out,
        )
        per_model_aggregate[model_name] = agg
        per_model_runs[model_name] = runs

        # full failure + economics metrics (across all repeats)
        fm = compute_failure_metrics(all_classified, all_propagations, all_causal)
        em = compute_economics(
            all_traces, all_correct, spec.price_in, spec.price_out, invalid_tokens
        )

        # save detailed results
        with open(os.path.join(config.output_dir, f"{model_name}_detail.json"), "w") as f:
            json.dump({"runs": runs, "failure_metrics": fm.as_dict(),
                       "economics": em.as_dict(), "aggregate": agg.as_dict()},
                      f, indent=2)

    # paired comparisons: baseline vs +verifier
    comparisons = {}
    for model_name in config.models:
        if model_name.endswith("+verifier"):
            base = model_name.replace("+verifier", "")
            if base in per_model_aggregate:
                # build per-task success series for paired test
                base_series = _per_task_success(per_model_runs[base])
                ver_series = _per_task_success(per_model_runs[model_name])
                t, p = paired_ttest(base_series, ver_series)
                d = cohens_d(base_series, ver_series)
                comparisons[f"{base}_vs_{model_name}"] = {
                    "delta_success_rate": (sum(ver_series) - sum(base_series)) / len(base_series)
                        if base_series else 0,
                    "t_stat": round(t, 4),
                    "p_value": round(p, 6),
                    "cohens_d": round(d, 4),
                }

    summary = {
        "config": {"n_repeats": config.n_repeats, "max_steps": config.max_steps,
                   "n_tasks": len(taskset)},
        "aggregate": {m: a.as_dict() for m, a in per_model_aggregate.items()},
        "comparisons": comparisons,
    }
    with open(os.path.join(config.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _per_task_success(runs: List[dict]) -> List[float]:
    """Average success per task across repeats -> one series for paired test."""
    by_task: Dict[str, List[int]] = {}
    for r in runs:
        by_task.setdefault(r["task_id"], []).append(1 if r["correct"] else 0)
    return [sum(v) / len(v) for v in by_task.values()]
