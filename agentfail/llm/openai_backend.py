"""OpenAI-compatible backend for real experiments.

Thin wrapper around the ``openai`` client. Pricing defaults reflect 2026-era
public list prices and can be overridden per experiment. The wrapper preserves
the exact same :class:`LLMResponse` / :class:`TokenUsage` contract as the mock,
so swapping backends requires no changes to the agent, diagnosis, or metrics
layers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import LLMBackend, LLMResponse, TokenUsage

# Default per-million-token prices (USD). Override per experiment.
DEFAULT_PRICES = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet": (3.0, 15.0),
    "deepseek-v3": (0.14, 0.28),
    "qwen2.5-72b": (0.5, 1.5),
}


class OpenAIBackend(LLMBackend):
    """Calls an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        price_in: Optional[float] = None,
        price_out: Optional[float] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        pin, pout = DEFAULT_PRICES.get(model, (0.0, 0.0))
        super().__init__(
            model=model,
            price_in=price_in if price_in is not None else pin,
            price_out=price_out if price_out is not None else pout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._client = None
        self._api_key = api_key
        self._base_url = base_url

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise ImportError(
                    "openai package is required for OpenAIBackend. "
                    "Install with: pip install openai"
                ) from exc
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _generate(
        self, messages: List[Dict[str, str]], max_tokens: int = 1024, **kwargs
    ) -> LLMResponse:  # pragma: no cover - network call
        client = self._ensure_client()
        temperature = kwargs.get("temperature", self.temperature)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = TokenUsage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return LLMResponse(
            text=text,
            usage=usage,
            raw=resp,
            model=self.model,
            finish_reason=choice.finish_reason or "stop",
        )
