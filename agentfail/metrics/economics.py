"""Token-economics metrics (Layer 3 of the metric stack).

Quantifies the cost dimension that Moghadasi & Ghaderi (2026) found
under-reported across 12 agent benchmarks:
  - token-per-success: tokens spent per correctly-solved task
  - cost-accuracy frontier: Pareto points of ($, success_rate)
  - invalid-token ratio: share of tokens wasted on retries/dead-ends
  - $/task: dollar cost per task

These turn "the agent is accurate" into "the agent is accurate at $X per
correct answer", which is the comparison reviewers (and practitioners) actually
need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ..agent.react_agent import AgentTrace


@dataclass
class EconomicsMetrics:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    n_success: int = 0
    n_total: int = 0
    token_per_success: float = 0.0
    invalid_token_ratio: float = 0.0
    total_cost: float = 0.0
    cost_per_task: float = 0.0
    cost_per_success: float = 0.0
    # Pareto frontier points: (cost, cumulative_success_rate)
    frontier: List[Tuple[float, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "n_success": self.n_success,
            "token_per_success": round(self.token_per_success, 2),
            "invalid_token_ratio": round(self.invalid_token_ratio, 4),
            "total_cost": round(self.total_cost, 4),
            "cost_per_task": round(self.cost_per_task, 4),
            "cost_per_success": round(self.cost_per_success, 4),
            "frontier": [(round(c, 4), round(s, 4)) for c, s in self.frontier],
        }


def compute_economics(
    traces: List[AgentTrace],
    correct_flags: List[bool],
    price_in: float = 0.0,
    price_out: float = 0.0,
    invalid_token_counts: List[int] = None,
) -> EconomicsMetrics:
    m = EconomicsMetrics()
    m.n_total = len(traces)
    m.n_success = sum(1 for c in correct_flags if c)

    total_in = sum(t.total_prompt_tokens() for t in traces)
    total_out = sum(t.total_completion_tokens() for t in traces)
    m.total_prompt_tokens = total_in
    m.total_completion_tokens = total_out
    m.total_tokens = total_in + total_out
    m.token_per_success = m.total_tokens / m.n_success if m.n_success else float("inf")

    if invalid_token_counts:
        tot_invalid = sum(invalid_token_counts)
        m.invalid_token_ratio = tot_invalid / m.total_tokens if m.total_tokens else 0.0

    m.total_cost = total_in * price_in / 1_000_000 + total_out * price_out / 1_000_000
    m.cost_per_task = m.total_cost / m.n_total if m.n_total else 0.0
    m.cost_per_success = m.total_cost / m.n_success if m.n_success else float("inf")

    # build cost-accuracy frontier: sort by cost, compute cumulative success rate
    per_task = sorted(
        zip(
            [t.total_tokens.cost(price_in, price_out) for t in traces],
            correct_flags,
        ),
        key=lambda x: x[0],
    )
    cum_cost = 0.0
    cum_success = 0
    for i, (c, ok) in enumerate(per_task, 1):
        cum_cost += c
        if ok:
            cum_success += 1
        m.frontier.append((cum_cost, cum_success / i))
    return m
