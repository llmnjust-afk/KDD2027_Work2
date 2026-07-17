"""Method contribution: failure-aware adaptive verification (rule-based).

This is the ORIGINAL rule-based verifier (v1). It uses ground-truth path
keywords to detect aggregation mismatches and over-tooling. It has HIGH false
positives (see experiment results: -5.3% success rate on GPT-4o-mini) because
it assumes the ground-truth path is the only correct path.

Kept for ablation comparison. The improved version is
:class:`ExecutionConsistencyVerifier` in :mod:`consistency_verifier`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..agent.react_agent import TraceStep, AgentTrace
from ..agent.sandbox import ExecutionResult


@dataclass
class VerifierVerdict:
    needs_retry: bool
    reason: str
    severity: str = "none"  # none | low | high


class FailureAwareVerifier:
    """Lightweight silent-failure detector that triggers adaptive retries."""

    def __init__(self, gt_path_keywords: Optional[list] = None, max_retries: int = 1):
        self.gt_keywords = [k.lower() for k in (gt_path_keywords or [])]
        self.max_retries = max_retries
        self._retries_used = 0

    def check(
        self, step: TraceStep, exec_res: ExecutionResult, trace: AgentTrace
    ) -> VerifierVerdict:
        if not exec_res.success:
            return VerifierVerdict(needs_retry=False, reason="loud failure, agent handles")

        code = step.code.lower()

        if self.gt_keywords:
            if "sum" in self.gt_keywords and ".count(" in code and ".sum(" not in code:
                self._retries_used += 1
                return VerifierVerdict(
                    needs_retry=True,
                    reason="aggregation mismatch: gt expects sum but code uses count",
                    severity="high",
                )
            if "sum" in self.gt_keywords and ".mean(" in code and ".sum(" not in code:
                self._retries_used += 1
                return VerifierVerdict(
                    needs_retry=True,
                    reason="aggregation mismatch: gt expects sum but code uses mean",
                    severity="high",
                )

        if ".sum(" in code and "to_numeric" not in code and "astype" not in code:
            if re.search(r"read_csv", code) and not re.search(r"dtypes|info\(\)", code):
                return VerifierVerdict(
                    needs_retry=False, reason="possible type issue (advisory)", severity="low"
                )

        if "linearregression" in code or "fit(" in code:
            if self.gt_keywords and all(
                k not in code for k in ["sum", "mean", "count", "len"]
            ):
                self._retries_used += 1
                return VerifierVerdict(
                    needs_retry=True,
                    reason="possible over-tooling: ML model where aggregation suffices",
                    severity="medium",
                )

        if exec_res.answer is None and step.action_type == "code":
            return VerifierVerdict(
                needs_retry=False,
                reason="no ANSWER marker (will continue loop)",
                severity="low",
            )

        return VerifierVerdict(needs_retry=False, reason="step looks consistent with gt path")
