"""B3: Execution-consistency verifier for silent-failure detection.

Replaces the fragile rule-based verifier with a self-consistency mechanism:
  1. Run the agent once (primary run).
  2. If the primary run produces an answer, run it AGAIN with a different
     seed/perturbation (secondary run).
  3. Compare the two answers. If they disagree -> likely silent failure ->
     trigger a retry that asks the agent to reconcile.
  4. If they agree -> high confidence, accept.

This is sound because silent failures are stochastic: the same model on the
same task will often produce DIFFERENT wrong answers across runs (e.g., one
run picks count, another picks sum). A correct answer is more likely to be
reproduced. This does NOT depend on ground-truth path keywords, eliminating
the false-positive problem that crippled the rule-based verifier.

The cost is roughly 1.4-1.6x tokens (secondary run only when primary succeeds),
not 2x, because failed primary runs skip the secondary check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..agent.react_agent import AgentTrace, TraceStep
from ..agent.sandbox import ExecutionResult


ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


@dataclass
class ConsistencyVerdict:
    needs_retry: bool
    reason: str
    severity: str = "none"  # none | low | high
    primary_answer: Optional[str] = None
    secondary_answer: Optional[str] = None


class ExecutionConsistencyVerifier:
    """Self-consistency-based silent-failure detector.

    Unlike FailureAwareVerifier (rule-based, high false positives), this
    verifier does NOT inspect code or use ground-truth path keywords. It
    only compares two independent executions' answers. A disagreement is
    strong evidence of a silent failure.
    """

    def __init__(self, max_retries: int = 1, similarity_fn=None):
        self.max_retries = max_retries
        self._retries_used = 0
        # optional custom answer-similarity function; default = exact match
        self.similarity_fn = similarity_fn or self._default_similarity

    @staticmethod
    def _default_similarity(a: Optional[str], b: Optional[str]) -> bool:
        """Two answers are 'consistent' if they match after normalisation."""
        if a is None or b is None:
            return False
        na = re.sub(r"\s+", " ", a.strip().lower())
        nb = re.sub(r"\s+", " ", b.strip().lower())
        return na == nb

    def check(
        self,
        primary_trace: AgentTrace,
        secondary_trace: Optional[AgentTrace],
    ) -> ConsistencyVerdict:
        """Compare primary and secondary run answers.

        Parameters
        ----------
        primary_trace : AgentTrace
            The first agent run.
        secondary_trace : AgentTrace or None
            The second agent run. If None, consistency cannot be checked and
            the verdict is no-retry (the caller should run a secondary first).
        """
        pa = primary_trace.final_answer
        if pa is None:
            # primary produced no answer -> loud failure, let agent loop handle
            return ConsistencyVerdict(
                needs_retry=False, reason="no primary answer (loud failure)",
                primary_answer=pa,
            )

        if secondary_trace is None:
            # signal that a secondary run is needed
            return ConsistencyVerdict(
                needs_retry=False, reason="secondary run required",
                primary_answer=pa,
            )

        sa = secondary_trace.final_answer
        if self.similarity_fn(pa, sa):
            return ConsistencyVerdict(
                needs_retry=False, reason="answers consistent across 2 runs",
                primary_answer=pa, secondary_answer=sa,
            )

        # disagreement -> likely silent failure
        self._retries_used += 1
        return ConsistencyVerdict(
            needs_retry=True,
            reason=f"answer inconsistency: primary='{pa[:50]}' vs secondary='{sa[:50] if sa else None}'",
            severity="high",
            primary_answer=pa, secondary_answer=sa,
        )
