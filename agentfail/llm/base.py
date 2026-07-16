"""LLM backend abstraction with token-usage accounting.

Token accounting is first-class because token economics is one of the core
innovation axes. Every backend call returns a :class:`LLMResponse` that carries
the generated text and a :class:`TokenUsage` record, so downstream metrics can
compute token-per-success, invalid-token ratio, and the cost-accuracy frontier
without instrumenting call sites individually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenUsage:
    """Token consumption for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    def cost(self, price_in: float, price_out: float) -> float:
        """Dollar cost given per-million-token prices."""
        return (
            self.prompt_tokens * price_in / 1_000_000
            + self.completion_tokens * price_out / 1_000_000
        )


@dataclass
class LLMResponse:
    """Result of one LLM generation."""

    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Optional[Any] = None
    model: str = ""
    finish_reason: str = "stop"
    meta: Dict[str, Any] = field(default_factory=dict)


class LLMBackend:
    """Abstract LLM backend.

    Subclasses implement :meth:`_generate`. The public :meth:`generate` wrapper
    normalizes pricing/config and records usage. Pricing is kept on the instance
    so the economics layer can convert token usage to dollar cost consistently.
    """

    def __init__(
        self,
        model: str,
        price_in: float = 0.0,
        price_out: float = 0.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.price_in = price_in
        self.price_out = price_out
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_usage = TokenUsage()
        self.call_count = 0
        self.invalid_token_count = 0

    def _generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        raise NotImplementedError

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs
    ) -> LLMResponse:
        """Generate a completion and account for token usage."""
        resp = self._generate(
            messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )
        resp.model = resp.model or self.model
        self.total_usage = self.total_usage + resp.usage
        self.call_count += 1
        return resp

    def unit_cost(self) -> float:
        return self.total_usage.cost(self.price_in, self.price_out)

    def register_invalid_tokens(self, n: int) -> None:
        """Mark ``n`` completion tokens as wasted (retries / dead-end reasoning)."""
        self.invalid_token_count += n
