"""Failure classifier: rule-based + optional LLM-as-judge (v2 taxonomy).

The classifier walks an :class:`AgentTrace` and assigns a
:class:`FailureClassification` to the originating step. It is a deterministic
rule engine first (high precision, fully reproducible, zero cost), with an
optional LLM-as-judge pass for ambiguous cases.

v2 taxonomy stages (mutually exclusive by detection point):
  ANALYTICAL_PLAN  -- the Thought text reveals a wrong approach BEFORE code runs.
  CODE_GENERATION  -- the plan is correct but the generated code uses the wrong
                      operation (detectable by comparing code to gt_path).
  RUNTIME          -- the code raises an exception. (loud)
  OUTPUT_MISMATCH  -- the code runs without error but the agent extracts /
                      reports a wrong answer. Observability is separate.
  ANSWER_ERROR     -- the code runs, output is consistent, but the answer is
                      wrong because the approach itself was flawed. (silent)

Disambiguation rule: assign the FIRST stage at which the error becomes
detectable, checking in order: RUNTIME > ANALYTICAL_PLAN > CODE_GENERATION >
OUTPUT_MISMATCH > ANSWER_ERROR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..agent.react_agent import AgentTrace, TraceStep
from ..benchmark.tasks import Task
from .taxonomy import (
    FailureCategory,
    FailureClassification,
    FailureStage,
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
    """Classify where and why an agent failed on a task (v2 taxonomy)."""

    def __init__(self, judge_llm=None):
        self.judge_llm = judge_llm

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _gt_ops(task: Task) -> set:
        """Extract expected operations from gt_path as a lowercase token set."""
        gt = " ".join(task.gt_path).lower()
        ops = set()
        if "sum()" in gt or ".sum(" in gt:
            ops.add("sum")
        if "count()" in gt or ".count(" in gt:
            ops.add("count")
        if "mean()" in gt or ".mean(" in gt or "average" in gt:
            ops.add("mean")
        if "groupby" in gt:
            ops.add("groupby")
        if "idxmax" in gt:
            ops.add("idxmax")
        if "idxmin" in gt:
            ops.add("idxmin")
        if "max(" in gt:
            ops.add("max")
        if "min(" in gt:
            ops.add("min")
        if "filter" in gt:
            ops.add("filter")
        if "dropna" in gt or "fillna" in gt:
            ops.add("nullhandle")
        if "sort" in gt:
            ops.add("sort")
        if "nunique" in gt or "unique" in gt:
            ops.add("unique")
        if "value_counts" in gt:
            ops.add("value_counts")
        return ops

    @staticmethod
    def _explicit_failure_message(trace: AgentTrace, final_answer) -> bool:
        """Whether the agent explicitly exposes failure to the user.

        This is intentionally independent of failure stage. Handled errors and
        abstentions can execute successfully while remaining observable.
        """
        text_parts = [str(final_answer or "")]
        for step in trace.steps:
            if step.execution and step.execution.stdout:
                text_parts.append(step.execution.stdout)
        text = "\n".join(text_parts).lower()
        patterns = (
            r"\bno data (?:file|files|available|found)",
            r"\bdata (?:file )?not found\b",
            r"\bcolumn not found\b",
            r"\bfile does not exist\b",
            r"\bunable to\b",
            r"\bcannot (?:load|find|access|compute|determine)\b",
            r"\berror:\s*(?:no|file|column|data)",
            r"\bnot available\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_silent(trace: AgentTrace, final_answer) -> bool:
        return not FailureClassifier._explicit_failure_message(trace, final_answer)

    @staticmethod
    def _expected_group_column(task: Task) -> Optional[str]:
        gt = " ".join(task.gt_path).lower()
        match = re.search(r"groupby\(([^)]+)\)", gt)
        return match.group(1).strip(" '\"[]") if match else None

    @staticmethod
    def _trace_code_generation(
        trace: AgentTrace, task: Task, final_answer
    ) -> Optional[Tuple[FailureCategory, int, str]]:
        """Detect implementation failures requiring whole-trace context."""
        code_steps = [step for step in trace.steps if step.action_type == "code"]
        gt_ops = FailureClassifier._gt_ops(task)
        gt = " ".join(task.gt_path).lower()

        # Correct plan followed directly by an unresolved placeholder answer.
        if not code_steps and final_answer and re.search(r"\{[^{}]+\}", str(final_answer)):
            return (
                FailureCategory.INCOMPLETE_ANALYSIS,
                0,
                "correct plan was not implemented; unresolved placeholder answer",
            )

        expected_group = FailureClassifier._expected_group_column(task)
        for step in code_steps:
            if step.execution is None or not step.execution.success:
                continue
            code = step.code.lower()
            stdout = (step.execution.stdout or "").lower()

            # Dynamic column-selection code can expose the chosen column in stdout.
            if expected_group:
                selected = re.search(
                    r"using\s+(?:user|item|category|group)\s+column:\s*[\"']?([a-z_][a-z0-9_]*)",
                    stdout,
                )
                if selected and selected.group(1) != expected_group:
                    return (
                        FailureCategory.WRONG_COLUMN_CODE,
                        step.step_index,
                        f"selected grouping column {selected.group(1)!r}; expected {expected_group!r}",
                    )

            # A frequent implementation bug uses a row maximum instead of the
            # grouped sum required by the reference path.
            answer_bearing = "answer:" in stdout or step.execution.answer is not None
            if (
                answer_bearing
                and "groupby" in gt_ops
                and "sum" in gt_ops
                and "groupby" not in code
                and ("idxmax" in code or ".max(" in code)
            ):
                return (
                    FailureCategory.WRONG_AGGREGATION_CODE,
                    step.step_index,
                    "row-level maximum used instead of required grouped sum",
                )

            # Expected singular columns are sometimes implemented as a
            # nonexistent plural name and handled without raising.
            if "click" in gt and re.search(r"[\"']clicks[\"']", code):
                return (
                    FailureCategory.WRONG_COLUMN_CODE,
                    step.step_index,
                    "code references 'clicks' while reference path requires 'click'",
                )

        if code_steps and (final_answer is None or not str(final_answer).strip()):
            last = code_steps[-1]
            if last.execution and last.execution.success:
                code = last.code.lower()
                thought = last.thought.lower()
                stdout = (last.execution.stdout or "").lower()
                promised = any(op in thought for op in gt_ops)
                promised = promised or (
                    "value_counts" in gt_ops and "most common" in thought
                )
                promised = promised or (
                    "count" in gt_ops and re.search(r"\bcount\b", thought) is not None
                )
                promised = promised or (
                    "filter" in gt_ops and re.search(r"\bfilter\b", thought) is not None
                )
                implemented = any(
                    token in code
                    for token in ("groupby", ".sum(", ".count(", ".mean(", "value_counts", "idxmax")
                )
                scalar_mode_count_unprinted = (
                    "value_counts" in gt_ops
                    and ".mode()" in code
                    and ".sum()" in code
                    and not stdout.strip()
                    and "print(" not in code
                )
                if (promised and not implemented) or scalar_mode_count_unprinted:
                    return (
                        FailureCategory.INCOMPLETE_ANALYSIS,
                        last.step_index,
                        "final code did not implement or print the operation promised in Thought",
                    )
        return None

    @staticmethod
    def _trace_output_mismatch(
        trace: AgentTrace, final_answer
    ) -> Optional[Tuple[FailureCategory, int, str]]:
        """Detect output interpretation using whole-trace evidence."""
        for index in range(1, len(trace.steps)):
            previous = trace.steps[index - 1]
            current = trace.steps[index]
            if not previous.execution or not previous.execution.success:
                continue
            if (previous.execution.stdout or "").strip():
                continue
            thought = current.thought.lower()
            if re.search(
                r"(?:no|without|didn't receive|did not receive|missing)\s+(?:visible\s+)?output|"
                r"no output (?:printed|shown|indicating)|"
                r"(?:file|data\.csv).*(?:does not exist|not exist|missing)",
                thought,
            ):
                return (
                    FailureCategory.MISREAD_OUTPUT,
                    current.step_index,
                    "empty stdout from an unprinted expression was interpreted as negative evidence",
                )

        successful_stdout = "\n".join(
            step.execution.stdout or ""
            for step in trace.steps
            if step.execution and step.execution.success
        )
        if final_answer and str(final_answer).strip():
            answer = str(final_answer).strip()
            if answer not in successful_stdout:
                return (
                    FailureCategory.HALLUCINATED_ANSWER,
                    trace.steps[-1].step_index if trace.steps else 0,
                    "final answer is not supported by successful execution output",
                )
        return None

    # -- per-step rule signals (v2) ---------------------------------------- #

    @staticmethod
    def _detect_runtime(step: TraceStep) -> Optional[FailureCategory]:
        """RUNTIME: code crashed (loud, detectable from traceback)."""
        if step.execution is None or step.execution.success:
            return None
        et = step.execution.error_type or ""
        if et == "KeyError":
            return FailureCategory.KEY_ERROR_RUNTIME
        if et == "SecurityError":
            return FailureCategory.SECURITY_BLOCK_RUNTIME
        return FailureCategory.RUNTIME_EXCEPTION

    @staticmethod
    def _detect_analytical_plan(step: TraceStep, task: Task) -> Optional[FailureCategory]:
        """ANALYTICAL_PLAN: the Thought text reveals a wrong approach."""
        thought = step.thought.lower()
        gt = " ".join(task.gt_path).lower()
        # thought says mean/average but gt requires sum
        if re.search(r"\b(?:compute|calculate|use|take)\s+(?:the\s+)?(?:mean|average)\b", thought) and "sum" in gt and "mean" not in gt:
            return FailureCategory.WRONG_OPERATION_PLAN
        # thought says count but gt requires sum
        if re.search(r"\b(?:compute|calculate|use|take)\s+(?:the\s+)?count\b", thought) and "sum" in gt and "count" not in gt:
            return FailureCategory.WRONG_OPERATION_PLAN
        # thought says sum but gt requires count
        if re.search(r"\b(?:compute|calculate|use|take)\s+(?:the\s+)?sum\b", thought) and "count" in gt and "sum" not in gt:
            return FailureCategory.WRONG_OPERATION_PLAN
        # leakage signals in thought
        if "linearregression" in thought and "leak" not in thought:
            for trap in task.traps:
                if trap.name in ("data_leakage", "temporal_leakage"):
                    return FailureCategory.LEAKAGE_PLAN
        return None

    @staticmethod
    def _detect_code_generation(step: TraceStep, task: Task) -> Optional[FailureCategory]:
        """CODE_GENERATION: code uses wrong operation vs gt_path (detectable
        by comparing code to reference solution)."""
        if step.execution is None or not step.execution.success:
            return None  # only check code that ran successfully
        code = step.code.lower()
        gt_ops = FailureClassifier._gt_ops(task)
        gt = " ".join(task.gt_path).lower()

        # wrong aggregation: code uses count but gt needs sum
        if "sum" in gt_ops and ".count(" in code and ".sum(" not in code:
            return FailureCategory.WRONG_AGGREGATION_CODE
        # wrong aggregation: code uses sum but gt needs count
        if "count" in gt_ops and ".sum(" in code and ".count(" not in code:
            return FailureCategory.WRONG_AGGREGATION_CODE
        # wrong aggregation: code uses mean but gt needs sum
        if "sum" in gt_ops and ".mean(" in code and ".sum(" not in code:
            return FailureCategory.WRONG_AGGREGATION_CODE
        # wrong aggregation: code uses sum but gt needs mean
        if "mean" in gt_ops and ".sum(" in code and ".mean(" not in code:
            return FailureCategory.WRONG_AGGREGATION_CODE
        # wrong tool: sklearn/ML where aggregation suffices
        if ("sklearn" in code or "linearregression" in code) and "sklearn" not in gt:
            return FailureCategory.WRONG_TOOL_CODE
        return None

    @staticmethod
    def _detect_output_mismatch(step: TraceStep, task: Task, final_answer) -> Optional[FailureCategory]:
        """OUTPUT_MISMATCH: code runs correctly but agent extracts wrong answer."""
        if step.execution is None or not step.execution.success:
            return None
        stdout = step.execution.stdout or ""
        # code ran but never printed ANSWER
        if "ANSWER:" not in stdout and final_answer is None:
            return FailureCategory.NO_ANSWER_PRINTED
        # answer not in stdout (hallucinated)
        if final_answer and str(final_answer).strip():
            if str(final_answer).strip() not in stdout:
                return FailureCategory.HALLUCINATED_ANSWER
        return None

    # -- main entry -------------------------------------------------------- #

    def classify(self, trace: AgentTrace, task: Task, final_answer) -> ClassifiedTrace:
        task_correct = task.check_answer(final_answer)

        if task_correct:
            return ClassifiedTrace(
                trace=trace,
                classification=FailureClassification(
                    stage=FailureStage.NONE,
                    category=FailureCategory.NONE,
                    step_index=-1,
                    is_silent=False,
                    evidence="task answered correctly",
                ),
                task_correct=True,
            )

        # Runtime is globally prior: an early exploratory step must not mask a
        # later explicit exception.
        for i, step in enumerate(trace.steps):
            if step.action_type != "code":
                continue
            cat = self._detect_runtime(step)
            if cat is not None:
                fc = FailureClassification(
                    stage=FailureStage.RUNTIME,
                    category=cat,
                    step_index=i,
                    is_silent=False,
                    evidence=f"runtime: {step.execution.error_type}: {step.execution.error_message}",
                    matched_trap=self._match_trap(task, "execution"),
                )
                return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

        trace_codegen = self._trace_code_generation(trace, task, final_answer)
        if trace_codegen is not None:
            cat, step_index, evidence = trace_codegen
            fc = FailureClassification(
                stage=FailureStage.CODE_GENERATION,
                category=cat,
                step_index=step_index,
                is_silent=self._is_silent(trace, final_answer),
                evidence=f"code_generation: {evidence}",
                matched_trap=self._match_trap(task, "tool_use"),
            )
            return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

        # Walk steps for conservative local planning and operation mismatches.
        for i, step in enumerate(trace.steps):
            if step.action_type != "code":
                continue

            # Analytical-plan rules are deliberately conservative and only
            # apply before any execution evidence exists.
            cat = self._detect_analytical_plan(step, task) if i == 0 else None
            if cat is not None:
                fc = FailureClassification(
                    stage=FailureStage.ANALYTICAL_PLAN,
                    category=cat,
                    step_index=i,
                    is_silent=self._is_silent(trace, final_answer),
                    evidence=f"analytical_plan: thought='{step.thought[:60]}'",
                    matched_trap=self._match_trap(task, "planning"),
                )
                return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

            # 3. CODE_GENERATION (code uses wrong operation vs gt_path)
            cat = self._detect_code_generation(step, task)
            if cat is not None:
                fc = FailureClassification(
                    stage=FailureStage.CODE_GENERATION,
                    category=cat,
                    step_index=i,
                    is_silent=self._is_silent(trace, final_answer),
                    evidence=f"code_generation: {cat.value} in step {i}",
                    matched_trap=self._match_trap(task, "tool_use"),
                )
                return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

        trace_output = self._trace_output_mismatch(trace, final_answer)
        if trace_output is not None:
            cat, step_index, evidence = trace_output
            fc = FailureClassification(
                stage=FailureStage.OUTPUT_MISMATCH,
                category=cat,
                step_index=step_index,
                is_silent=self._is_silent(trace, final_answer),
                evidence=f"output_mismatch: {evidence}",
                matched_trap=self._match_trap(task, "interpretation"),
            )
            return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

        # no rule fired but task wrong -> distinguish ANSWER_ERROR vs fallback
        last_step = trace.steps[-1] if trace.steps else None
        if final_answer is None or (isinstance(final_answer, str) and not final_answer.strip()):
            # A missing answer without an exception is an observable output
            # failure, not a fabricated runtime exception.
            fc = FailureClassification(
                stage=FailureStage.OUTPUT_MISMATCH,
                category=FailureCategory.NO_ANSWER_PRINTED,
                step_index=last_step.step_index if last_step else 0,
                is_silent=self._is_silent(trace, final_answer),
                evidence="no answer produced and no runtime exception observed",
            )
        else:
            # answer produced but wrong, no code-level error detected -> ANSWER_ERROR
            fc = FailureClassification(
                stage=FailureStage.ANSWER_ERROR,
                category=FailureCategory.WRONG_RESULT,
                step_index=last_step.step_index if last_step else 0,
                is_silent=self._is_silent(trace, final_answer),
                evidence="answer produced but incorrect (approach-level error)",
            )
        return ClassifiedTrace(trace=trace, classification=fc, task_correct=False)

    @staticmethod
    def _match_trap(task: Task, stage: str) -> Optional[str]:
        for trap in task.traps:
            if trap.stage == stage:
                return trap.name
        return None
