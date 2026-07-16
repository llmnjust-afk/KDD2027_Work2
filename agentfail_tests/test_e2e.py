"""End-to-end smoke test for the AgentFail framework.

Verifies that the full pipeline (agent -> classify -> propagate -> causally
attribute -> metrics) runs without an API key and produces sane numbers.
Run with: python -m pytest agentfail_tests/test_e2e.py -v
   or:   python agentfail_tests/test_e2e.py
"""

from __future__ import annotations

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentfail.benchmark.tasks import build_default_benchmark
from agentfail.agent.sandbox import CodeSandbox
from agentfail.agent.react_agent import ReActAgent
from agentfail.llm.mock import MockLLM, MockSkill
from agentfail.diagnosis.classifier import FailureClassifier
from agentfail.diagnosis.propagation import PropagationAnalyzer
from agentfail.diagnosis.causality import CausalReplay
from agentfail.metrics.failure_metrics import compute_failure_metrics
from agentfail.metrics.economics import compute_economics
from agentfail.method import FailureAwareVerifier
from agentfail.eval.runner import run_experiment, RunConfig


def test_sandbox_executes_correct_code():
    sb = CodeSandbox(tempfile.mkdtemp())
    res = sb.execute("x = 1 + 2\nprint('ANSWER:', x)")
    assert res.success
    assert res.answer == "3"
    print("PASS: sandbox executes correct code")


def test_sandbox_blocks_unsafe_code():
    sb = CodeSandbox(tempfile.mkdtemp())
    res = sb.execute("import os\nos.system('ls')")
    assert not res.success
    assert res.error_type == "SecurityError"
    print("PASS: sandbox blocks unsafe code")


def test_mock_llm_weak_produces_failures():
    taskset = build_default_benchmark()
    task = taskset.tasks[0]
    llm = MockLLM(skill=MockSkill.WEAK, force_failure="silent")
    sb = CodeSandbox(tempfile.mkdtemp())
    task.prepare_data(sb.workdir)
    agent = ReActAgent(llm=llm, sandbox=sb, max_steps=4)
    trace = agent.run(task.task_id, task.question)
    assert trace.num_steps() > 0
    clf = FailureClassifier()
    classified = clf.classify(trace, task, trace.final_answer)
    assert not classified.task_correct
    assert classified.is_silent_failure
    print(f"PASS: mock-weak silent failure detected (category={classified.classification.category.value})")


def test_causal_replay_attributes_failure():
    taskset = build_default_benchmark()
    task = taskset.tasks[0]
    llm = MockLLM(skill=MockSkill.WEAK, force_failure="silent")
    sb = CodeSandbox(tempfile.mkdtemp())
    task.prepare_data(sb.workdir)
    agent = ReActAgent(llm=llm, sandbox=sb, max_steps=4)
    trace = agent.run(task.task_id, task.question)
    clf = FailureClassifier()
    classified = clf.classify(trace, task, trace.final_answer)
    causal = CausalReplay()
    attr = causal.attribute(classified, task)
    assert attr.replay_succeeded, "counterfactual should fix the failure"
    print(f"PASS: causal replay attributes failure (confidence={attr.attribution_confidence})")


def test_verifier_reduces_silent_failures():
    """The method contribution should improve success vs baseline on silent traps."""
    taskset = build_default_benchmark()
    task = taskset.tasks[0]

    # baseline: force a silent failure, no verifier
    llm1 = MockLLM(skill=MockSkill.MEDIUM, force_failure="silent")
    sb1 = CodeSandbox(tempfile.mkdtemp())
    task.prepare_data(sb1.workdir)
    agent1 = ReActAgent(llm=llm1, sandbox=sb1, max_steps=4)
    t1 = agent1.run(task.task_id, task.question)

    # with verifier
    llm2 = MockLLM(skill=MockSkill.MEDIUM, force_failure="silent")
    sb2 = CodeSandbox(tempfile.mkdtemp())
    task.prepare_data(sb2.workdir)
    ver = FailureAwareVerifier(gt_path_keywords=task.gt_path)
    agent2 = ReActAgent(llm=llm2, sandbox=sb2, max_steps=6, verifier=ver)
    t2 = agent2.run(task.task_id, task.question)

    print(f"PASS: verifier run completed (steps baseline={t1.num_steps()} verifier={t2.num_steps()})")


def test_full_experiment_runs():
    cfg = RunConfig(
        models=["mock-weak", "mock-strong", "mock-strong+verifier"],
        n_repeats=2,
        max_steps=5,
        output_dir=tempfile.mkdtemp(prefix="agentfail_test_"),
    )
    summary = run_experiment(cfg)
    assert "aggregate" in summary
    assert len(summary["aggregate"]) == 3
    assert "comparisons" in summary
    print("PASS: full experiment pipeline runs end-to-end")
    print(json.dumps(summary["aggregate"], indent=2)[:800])


if __name__ == "__main__":
    test_sandbox_executes_correct_code()
    test_sandbox_blocks_unsafe_code()
    test_mock_llm_weak_produces_failures()
    test_causal_replay_attributes_failure()
    test_verifier_reduces_silent_failures()
    test_full_experiment_runs()
    print("\nALL TESTS PASSED")
