"""The 4-stage failure taxonomy.

This taxonomy is the conceptual core of the benchmark. It decomposes every
agent failure into *which stage of the agent loop it originated in*:

  PLANNING      -- the agent chose the wrong approach / operation
  TOOL_USE      -- the agent selected or parameterised a tool incorrectly
  EXECUTION     -- the generated code raised an exception (loud failure)
  INTERPRETATION-- the code ran but the agent extracted/understood the wrong
                   answer (silent failure)

The EXECUTION/INTERPRETATION split is what operationalises the "silent failure"
concept from Tree-Notebook (Qiu 2026) and Wu (2026): a loud failure is visible
to the agent (it can retry), a silent failure is not (it propagates undetected).

Each category carries structured fields so the classifier output is machine-
readable and the metrics layer can aggregate without re-parsing free text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class FailureStage(Enum):
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    EXECUTION = "execution"
    INTERPRETATION = "interpretation"

    @classmethod
    def from_trap(cls, trap_stage: str) -> "FailureStage":
        mapping = {
            "planning": cls.PLANNING,
            "tool_use": cls.TOOL_USE,
            "execution": cls.EXECUTION,
            "interpretation": cls.INTERPRETATION,
        }
        return mapping.get(trap_stage, cls.INTERPRETATION)


class FailureCategory(Enum):
    """Concrete failure sub-types within each stage."""

    # planning
    WRONG_OPERATION = "wrong_operation"        # mean instead of sum
    WRONG_DECOMPOSITION = "wrong_decomposition"
    TEMPORAL_LEAKAGE = "temporal_leakage"      # uses future/test data
    DATA_LEAKAGE = "data_leakage"              # target as feature

    # tool_use
    WRONG_TOOL = "wrong_tool"                  # ML model where pandas suffices
    WRONG_PARAMS = "wrong_params"
    OVER_PRIVILEGED = "over_privileged"        # higher-privilege tool than needed

    # execution
    RUNTIME_ERROR = "runtime_error"
    TYPE_ERROR = "type_error"
    KEY_ERROR = "key_error"
    SECURITY_BLOCK = "security_block"

    # interpretation
    WRONG_AGGREGATION = "wrong_aggregation"    # count vs sum (silent)
    WRONG_INDEX = "wrong_index"
    MISREAD_OUTPUT = "misread_output"
    HALLUCINATED_ANSWER = "hallucinated_answer"  # answer not in output

    NONE = "none"  # no failure


# Which categories are "silent" (code runs, wrong answer)
SILENT_CATEGORIES = {
    FailureCategory.WRONG_AGGREGATION,
    FailureCategory.WRONG_INDEX,
    FailureCategory.MISREAD_OUTPUT,
    FailureCategory.HALLUCINATED_ANSWER,
    FailureCategory.TEMPORAL_LEAKAGE,
    FailureCategory.DATA_LEAKAGE,
    FailureCategory.WRONG_OPERATION,  # wrong op may still run
}


@dataclass
class FailureClassification:
    """Result of classifying one trace (or one step)."""

    stage: FailureStage
    category: FailureCategory
    step_index: int                      # where the failure originated
    is_silent: bool
    confidence: float = 1.0              # rule-based = 1.0; LLM-judge < 1.0
    evidence: str = ""                   # why this classification
    matched_trap: Optional[str] = None   # name of the task trap it matches
    propagated_to: List[int] = field(default_factory=list)  # later steps affected

    @property
    def is_failure(self) -> bool:
        return self.category != FailureCategory.NONE
