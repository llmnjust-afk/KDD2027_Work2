"""Failure-propagation-depth analysis.

Once a failure originates at step ``i``, it may or may not be *detected* by the
agent. A loud (execution) failure is immediately visible and the agent can
retry; a silent failure is invisible and *propagates* -- the agent keeps building
on the wrong intermediate result. The propagation depth is the number of steps
between the originating failure and the step where it is (or isn't) corrected.

This directly addresses the LongDS-Bench (Xu 2026) gap: they show long-horizon
tasks fail, but do not measure *how far* a failure propagates before detection.
Long propagation depth is the most dangerous failure mode because it wastes the
most tokens and is hardest to recover from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..agent.react_agent import AgentTrace
from .classifier import ClassifiedTrace


@dataclass
class PropagationReport:
    origin_step: int
    detection_step: int          # step where failure was detected, or -1 if never
    propagation_depth: int       # detection - origin (or num_steps - origin if never)
    was_detected: bool
    was_recovered: bool          # detected AND corrected to a correct answer
    recovery_steps: List[int]    # steps that were recovery attempts

    @property
    def is_long_horizon(self) -> bool:
        """A propagation depth >= 2 indicates long-horizon failure spread."""
        return self.propagation_depth >= 2


class PropagationAnalyzer:
    """Compute how far a failure propagated before detection/recovery."""

    def analyze(self, classified: ClassifiedTrace) -> PropagationReport:
        trace = classified.trace
        cls = classified.classification
        origin = cls.step_index
        n = trace.num_steps()

        if not cls.is_failure:
            return PropagationReport(
                origin_step=-1, detection_step=-1, propagation_depth=0,
                was_detected=False, was_recovered=True, recovery_steps=[],
            )

        # Loud failures (execution) are detected at origin
        if not cls.is_silent:
            detection = origin
            recovered = classified.task_correct
            recovery_steps = [
                s.step_index for s in trace.steps
                if s.is_recovery_attempt and s.step_index > origin
            ]
            return PropagationReport(
                origin_step=origin,
                detection_step=detection,
                propagation_depth=0,
                was_detected=True,
                was_recovered=recovered,
                recovery_steps=recovery_steps,
            )

        # Silent failures: scan forward for a recovery attempt or correction
        detection = -1
        recovery_steps: List[int] = []
        for s in trace.steps:
            if s.step_index <= origin:
                continue
            if s.is_recovery_attempt:
                recovery_steps.append(s.step_index)
            # if a later step produced the correct answer, that's detection
            if s.execution and s.execution.answer is not None:
                # cannot know correctness without task; approximate: any later
                # successful answer change counts as a detection attempt
                detection = s.step_index
                break

        was_detected = detection != -1
        depth = (detection - origin) if was_detected else (n - 1 - origin)
        return PropagationReport(
            origin_step=origin,
            detection_step=detection,
            propagation_depth=max(depth, 0),
            was_detected=was_detected,
            was_recovered=classified.task_correct,
            recovery_steps=recovery_steps,
        )
