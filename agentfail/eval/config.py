"""Experiment configuration.

The model grid spans the price/skill spectrum so the cost-accuracy frontier is
populated. For the zero-cost demo we use MockLLM at three skill levels; for
real experiments the same grid maps to OpenAI-compatible backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelSpec:
    name: str
    backend: str  # "mock" | "openai"
    skill: str = "medium"  # for mock
    price_in: float = 0.0
    price_out: float = 0.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None


# Mock grid (zero-cost, fully reproducible)
MODEL_GRID: Dict[str, ModelSpec] = {
    "mock-weak": ModelSpec("mock-weak", "mock", "weak", 0.15, 0.60),
    "mock-medium": ModelSpec("mock-medium", "mock", "medium", 0.5, 1.5),
    "mock-strong": ModelSpec("mock-strong", "mock", "strong", 2.5, 10.0),
    "mock-medium+verifier": ModelSpec("mock-medium+verifier", "mock", "medium", 0.5, 1.5),
    "mock-strong+verifier": ModelSpec("mock-strong+verifier", "mock", "strong", 2.5, 10.0),
}

# Real-model grid (set env vars for keys)
REAL_MODEL_GRID: Dict[str, ModelSpec] = {
    "gpt-4o": ModelSpec("gpt-4o", "openai", price_in=2.5, price_out=10.0),
    "gpt-4o-mini": ModelSpec("gpt-4o-mini", "openai", price_in=0.15, price_out=0.60),
    "deepseek-v3": ModelSpec("deepseek-v3", "openai", price_in=0.14, price_out=0.28,
                             base_url="https://api.deepseek.com"),
    "claude-3-5-sonnet": ModelSpec("claude-3-5-sonnet", "openai", price_in=3.0, price_out=15.0,
                                   base_url="https://api.anthropic.com/v1"),
}


@dataclass
class DEFAULT_CONFIG:
    n_repeats: int = 3
    max_steps: int = 6
    use_verifier: bool = False
    output_dir: str = "results"
    models: List[str] = field(default_factory=lambda: [
        "mock-weak", "mock-medium", "mock-strong",
        "mock-medium+verifier", "mock-strong+verifier",
    ])
    seed: int = 0
