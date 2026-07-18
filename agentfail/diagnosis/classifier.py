"""Failure classifier: rule-based + optional LLM-as-judge.

The classifier walks an :class:`AgentTrace` and assigns a
:class:`FailureClassification` to the originating step. It is deliberately a
deterministic rule engine first (high precision, fully reproducible, zero cost),
with an optional LLM-as-judge pass that can refine ambiguous cases and whose
agreement with the rules is itself a reported metric (inter-rater reliability).

Rule design mirrors the failure-mode taxonomies synthesised in
Albayaydh et al. (2026) "Beyond the Leaderboard" and Soni (2026) ToolFailBench,
but is grounded in our 4-stage taxonomy so every rule maps to exactly one stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ..agent.react_agent import AgentTrace, TraceStep
from ..benchmark.tasks import Task
from .taxonomy import (
    FailureCategory,
    FailureClassification,
    FailureStage,
    SILENT_CATEGORIES,
)


@dataclass
class ClassifiedTrace:
    trace: AgentTrace
    classification: FailureClassification
    task_correct: bool

    @property
    def is_silent_failure(self) -> bool:
        return (
            not self.task_correct
            and self.classification.is_failure
            and self.classification.is_silent
        )


class FailureClassifier:
    """Classify where and why an agent failed on a task."""

    def __init__(self, judge_llm=None):
        self.judge_llm = judge_llm  # optional LLMBackend for ambiguous cases

    # -- per-step rule signals --------------------------------------------- #

    @staticmethod
    def _detect_planning(step: TraceStep, task: Task) -> Optional[FailureCategory]:
        code = step.code.lower()
        thought = step.thought.lower()
        if "mean" in code and ("sum" in " ".join(task.gt_path).lower()) and "groupby" not in code:
            return FailureCategory.WRONG_OPERATION_PLAN
        if "average" in thought and "sum" in " ".join(task.gt_path).lower():
            return FailureCategory.WRONG_OPERATION_PLAN
        # leakage signals
        if "linearregression" in code and "leak" not in code:
            for trap in task.traps:
                if trap.name in ("data_leakage", "temporal_leakage"):
                    return FailureCategory[trap.name.upper()]
        return None

    @staticmethod
    def _detect_tool_use(step: TraceStep) -> Optional[FailureCategory]:
        code = step.code.lower()
        if "sklearn" in code or "linearregression" in code:
            # ML tool where aggregation suffices
            return FailureCategory.WRONG_TOOL_CODE
        return None

    @staticmethod
    def _detect_execution(step: TraceStep) -> Optional[FailureCategory]:
        if step.execution is None or step.execution.success:
            return None
        et = step.execution.error_type or ""
        if et == "KeyError":
            return FailureCategory.KEY_ERROR_RUNTIME
        if et in ("TypeError", "ValueError"):
            return FailureCategory.TYPE_CONFUSION_CODE
        if et == "SecurityError":
            return FailureCategory.SECURITY_BLOCK_RUNTIME
        return FailureCategory.RUNTIME_EXCEPTION

    @staticmethod
    def _detect_interpretation(step: TraceStep, task: Task, final_answer) -> Optional[FailureCategory]:
        code = step.code.lower()
        # wrong aggregation: count instead of sum
        if ".count(" in code and "sum" in " ".join(task.gt_path).lower():
            return FailureCategory.WRONG_AGGREGATION_CODE
        # answer not present in stdout (hallucinated)
        if final_answer and step.execution and step.execution.success:
            if str(final_answer) not in step.execution.stdout:
                # check the final-answer step specifically
                return FailureCategory.MISREAD_OUTPUT
        return None

    # -- main entry -------------------------------------------------------- #

    def classify(self, trace: AgentTrace, task: Task, final_answer) -> ClassifiedTrace:
        task_correct = task.check_answer(final_answer)

        # if task is correct, no failure to classify
        if task_correct:
            return ClassifiedTrace(
                trace=trace,
                classification=FailureClassification(
                    stage=FailureStage.OUTPUT_MISMATCH,
                    category=FailureCategory.NONE,
                    step_index=-1,
                    is_silent=False,
                    evidence="task answered correctly",
                ),
                task_correct=True,
            )

        # find the originating failure step
        for i, step in enumerate(trace.steps):
            if step.action_type != "code":
                continue

            # execution failures are loud and localised
            cat = self._detect_execution(step)
            if cat is not None:
                fc = FailureClassification(
                    stage=FailureStage.RUNTIME,
                    category=cat,
                    step_index=i,
                    is_silent=False,
                    evidence=f"{step.execution.error_type}: {step.execution.error_message}",
                    matched_trap=self._match_trap(task, "execution"),
                )
                return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

            # planning
            cat = self._detect_planning(step, task)
            if cat is not None:
                silent = cat in SILENT_CATEGORIES
                fc = FailureClassification(
                    stage=FailureStage.ANALYTICAL_PLAN,
                    category=cat,
                    step_index=i,
                    is_silent=silent,
                    evidence=f"planning signal in step {i}: thought='{step.thought[:60]}'",
                    matched_trap=self._match_trap(task, "planning"),
                )
                return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

            # tool use
            cat = self._detect_tool_use(step)
            if cat is not None:
                fc = FailureClassification(
                    stage=FailureStage.CODE_GENERATION,
                    category=cat,
                    step_index=i,
                    is_silent=cat in SILENT_CATEGORIES,
                    evidence=f"tool-use signal: sklearn present",
                    matched_trap=self._match_trap(task, "tool_use"),
                )
                return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

            # interpretation (silent)
            cat = self._detect_interpretation(step, task, final_answer)
            if cat is not None:
                fc = FailureClassification(
                    stage=FailureStage.OUTPUT_MISMATCH,
                    category=cat,
                    step_index=i,
                    is_silent=True,
                    evidence=f"silent: {cat.value} in step {i}",
                    matched_trap=self._match_trap(task, "interpretation"),
                )
                return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

        # no rule fired but task wrong
        # FIX per review: if final_answer is None, the code never produced an
        # answer -> this is a RUNTIME failure (loud), NOT a silent interpretation failure.
        # The old code defaulted to INTERPRETATION + silent=True, which inflated SFR.
        last_step = trace.steps[-1] if trace.steps else None
        if final_answer is None or (isinstance(final_answer, str) and not final_answer.strip()):
            # no answer produced = code crashed or never printed ANSWER -> loud failure
            fc = FailureClassification(
                stage=FailureStage.RUNTIME,
                category=FailureCategory.RUNTIME_EXCEPTION,
                step_index=last_step.step_index if last_step else 0,
                is_silent=False,
                evidence="no answer produced (code crashed or never printed ANSWER:)",
            )
        else:
            # answer produced but wrong -> genuinely silent interpretation failure
            fc = FailureClassification(
                stage=FailureStage.OUTPUT_MISMATCH,
                category=FailureCategory.HALLUCINATED_ANSWER,
                step_index=last_step.step_index if last_step else 0,
                is_silent=True,
                evidence="answer produced but incorrect (genuine silent failure)",
            )
        return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

    @staticmethod
    def _match_trap(task: Task, stage: str) -> Optional[str]:
        for trap in task.traps:
            if trap.stage == stage:
                return trap.name
        return None
