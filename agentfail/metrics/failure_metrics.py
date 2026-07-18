"""Failure-diagnosis metrics (Layer 2 of the metric stack).

These are the novel metrics that DSBench/DataSciBench do not report:
  - SFR: silent failure rate
  - stage distribution: where failures originate
  - propagation depth: how far failures spread
  - recovery rate: how often agents self-correct
  - tool-misuse breakdown + over-privilege rate

All metrics are computed from ClassifiedTrace + PropagationReport objects so
they are deterministic and auditable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from ..diagnosis.classifier import ClassifiedTrace
from ..diagnosis.propagation import PropagationReport
from ..diagnosis.taxonomy import FailureStage


@dataclass
class FailureMetrics:
    n_total: int = 0
    n_correct: int = 0
    n_failed: int = 0
    n_silent: int = 0
    n_loud: int = 0
    silent_failure_rate: float = 0.0          # SFR
    stage_distribution: Dict[str, float] = field(default_factory=dict)
    avg_propagation_depth: float = 0.0
    max_propagation_depth: int = 0
    long_horizon_failure_rate: float = 0.0    # proportion of failures with depth>=2
    recovery_rate: float = 0.0                # detected failures that were recovered
    over_privilege_rate: float = 0.0
    tool_misuse_breakdown: Dict[str, int] = field(default_factory=dict)
    oracle_repair_rate: float = 0.0       # renamed from causal_attribution_rate (per review)

    @property
    def success_rate(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0

    def as_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "success_rate": round(self.success_rate, 4),
            "n_failed": self.n_failed,
            "silent_failure_rate": round(self.silent_failure_rate, 4),
            "stage_distribution": {k: round(v, 4) for k, v in self.stage_distribution.items()},
            "avg_propagation_depth": round(self.avg_propagation_depth, 4),
            "max_propagation_depth": self.max_propagation_depth,
            "long_horizon_failure_rate": round(self.long_horizon_failure_rate, 4),
            "recovery_rate": round(self.recovery_rate, 4),
            "over_privilege_rate": round(self.over_privilege_rate, 4),
            "tool_misuse_breakdown": self.tool_misuse_breakdown,
            "oracle_repair_rate": round(self.oracle_repair_rate, 4),
        }


def compute_failure_metrics(
    classified: List[ClassifiedTrace],
    propagations: List[PropagationReport],
    causal_attributions: List[bool] = None,
) -> FailureMetrics:
    m = FailureMetrics()
    m.n_total = len(classified)
    m.n_correct = sum(1 for c in classified if c.task_correct)
    m.n_failed = m.n_total - m.n_correct

    stages = Counter()
    tool_misuse = Counter()
    n_over_priv = 0
    n_silent = 0
    n_loud = 0
    depths = []
    n_long = 0
    n_detected = 0
    n_recovered = 0

    for c in classified:
        if c.task_correct:
            continue
        cls = c.classification
        stages[cls.stage.value] += 1
        if c.is_silent_failure:
            n_silent += 1
        else:
            n_loud += 1
        if cls.category.value == "over_privileged":
            n_over_priv += 1
        if cls.stage == FailureStage.TOOL_USE:
            tool_misuse[cls.category.value] += 1

    for p in propagations:
        depths.append(p.propagation_depth)
        if p.is_long_horizon:
            n_long += 1
        if p.was_detected:
            n_detected += 1
            if p.was_recovered:
                n_recovered += 1

    m.n_silent = n_silent
    m.n_loud = n_loud
    m.silent_failure_rate = n_silent / m.n_failed if m.n_failed else 0.0
    m.stage_distribution = {
        s: stages[s] / m.n_failed for s in stages
    } if m.n_failed else {}
    m.avg_propagation_depth = sum(depths) / len(depths) if depths else 0.0
    m.max_propagation_depth = max(depths) if depths else 0
    m.long_horizon_failure_rate = n_long / m.n_failed if m.n_failed else 0.0
    m.recovery_rate = n_recovered / n_detected if n_detected else 0.0
    m.over_privilege_rate = n_over_priv / m.n_failed if m.n_failed else 0.0
    m.tool_misuse_breakdown = dict(tool_misuse)
    if causal_attributions:
        # FIX: only compute on FAILED traces (not successful ones)
        # causal_attributions is a list of bools aligned with classified list
        failed_causal = []
        for c, ca in zip(classified, causal_attributions):
            if not c.task_correct:
                failed_causal.append(ca)
        m.oracle_repair_rate = sum(1 for a in failed_causal if a) / len(failed_causal) if failed_causal else 0.0
    return m
