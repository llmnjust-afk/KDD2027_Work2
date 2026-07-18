"""The 5-stage failure taxonomy (v2, revised per reviewer feedback).

FIX per review: The original 4-stage taxonomy had overlapping categories:
Planning's `wrong_operation` (mean instead of sum) and Interpretation's
`wrong_aggregation` (count instead of sum) could describe the same error.
The reviewer correctly noted this created circular classification.

v2 redesign: stages are now defined by WHERE the error is detectable in the
agent loop, not by what the error conceptually is. This makes stages mutually
exclusive by construction:

  ANALYTICAL_PLAN   -- the agent's stated plan (Thought) is wrong BEFORE any
                        code is written. Detectable from the thought text alone.
  CODE_GENERATION   -- the plan is correct but the generated code does not
                        implement it. Detectable by comparing thought to code.
  RUNTIME           -- the code raises an exception. Detectable from the
                        traceback. (loud failure)
  OUTPUT_MISMATCH   -- the code runs without error but the printed answer does
                        not match the computed result. Detectable by comparing
                        stdout to the final ANSWER. (silent, agent's fault)
  ANSWER_ERROR      -- the code runs, the output is internally consistent, but
                        the answer is wrong because the approach itself was
                        flawed (e.g., correct code for the wrong question).
                        Detectable only by external checking. (silent, approach's fault)

Key change: ANALYTICAL_PLAN vs ANSWER_ERROR are now distinguished by WHETHER
the agent's Thought reveals the error. If the thought says "compute the mean"
but the task asks for sum, that's ANALYTICAL_PLAN (visible in thought). If the
thought says "compute the sum" and the code computes the sum but the answer is
still wrong (because the data needs filtering first), that's ANSWER_ERROR
(invisible without external ground truth).

This removes the overlap: a count-vs-sum error is classified by looking at the
Thought. If the Thought says "count the items" (wrong plan), it's
ANALYTICAL_PLAN. If the Thought says "sum the values" but the code uses
.count() (code doesn't match plan), it's CODE_GENERATION.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class FailureStage(Enum):
    """v2 stages, mutually exclusive by detection point."""
    ANALYTICAL_PLAN = "analytical_plan"   # thought reveals wrong approach
    CODE_GENERATION = "code_generation"   # code doesn't match correct thought
    RUNTIME = "runtime"                   # code crashes (loud)
    OUTPUT_MISMATCH = "output_mismatch"   # answer extracted wrong from output (silent)
    ANSWER_ERROR = "answer_error"         # code runs, output consistent, answer wrong (silent)
    NONE = "none"

    @classmethod
    def from_trap(cls, trap_stage: str) -> "FailureStage":
        """Map old trap stages to new taxonomy for backward compatibility."""
        mapping = {
            "planning": cls.ANALYTICAL_PLAN,
            "tool_use": cls.CODE_GENERATION,
            "execution": cls.RUNTIME,
            "interpretation": cls.OUTPUT_MISMATCH,
        }
        return mapping.get(trap_stage, cls.ANSWER_ERROR)


class FailureCategory(Enum):
    """Concrete failure sub-types within each stage (v2, mutually exclusive)."""

    # ANALYTICAL_PLAN: the Thought text reveals a wrong approach
    WRONG_OPERATION_PLAN = "wrong_operation_plan"  # thought says mean, task needs sum
    WRONG_FILTER_PLAN = "wrong_filter_plan"        # thought says filter X, should filter Y
    LEAKAGE_PLAN = "leakage_plan"                  # thought reveals using future/target data

    # CODE_GENERATION: thought is correct but code doesn't implement it
    WRONG_AGGREGATION_CODE = "wrong_aggregation_code"  # thought says sum, code uses count
    WRONG_COLUMN_CODE = "wrong_column_code"            # thought says col A, code uses col B
    TYPE_CONFUSION_CODE = "type_confusion_code"        # thought says numeric, code treats as string
    WRONG_TOOL_CODE = "wrong_tool_code"                # thought says groupby, code uses sklearn

    # RUNTIME: code crashes (loud, immediately detectable)
    RUNTIME_EXCEPTION = "runtime_exception"
    KEY_ERROR_RUNTIME = "key_error_runtime"
    SECURITY_BLOCK_RUNTIME = "security_block_runtime"

    # OUTPUT_MISMATCH: code runs but agent extracts wrong answer from output (silent)
    MISREAD_OUTPUT = "misread_output"              # output has answer but agent reports different
    HALLUCINATED_ANSWER = "hallucinated_answer"    # answer not in output at all
    NO_ANSWER_PRINTED = "no_answer_printed"        # code ran but never printed ANSWER:

    # ANSWER_ERROR: code runs, output is consistent, but approach was wrong (silent)
    WRONG_RESULT = "wrong_result"                  # correct code execution, wrong final value
    INCOMPLETE_ANALYSIS = "incomplete_analysis"    # missing a step (e.g., forgot to filter)

    NONE = "none"


# Silent = code ran without error but answer is wrong
SILENT_STAGES = {FailureStage.OUTPUT_MISMATCH, FailureStage.ANSWER_ERROR}
SILENT_CATEGORIES = {
    FailureCategory.MISREAD_OUTPUT,
    FailureCategory.HALLUCINATED_ANSWER,
    FailureCategory.NO_ANSWER_PRINTED,
    FailureCategory.WRONG_RESULT,
    FailureCategory.INCOMPLETE_ANALYSIS,
    # plan errors that produce running code are also silent
    FailureCategory.WRONG_OPERATION_PLAN,
    FailureCategory.WRONG_FILTER_PLAN,
    FailureCategory.LEAKAGE_PLAN,
}


@dataclass
class FailureClassification:
    """Result of classifying one trace (or one step)."""

    stage: FailureStage
    category: FailureCategory
    step_index: int
    is_silent: bool
    confidence: float = 1.0
    evidence: str = ""
    matched_trap: Optional[str] = None
    propagated_to: List[int] = field(default_factory=list)

    @property
    def is_failure(self) -> bool:
        return self.category != FailureCategory.NONE
