"""Deterministic mock LLM.

This is the key enabler for reproducible, zero-cost end-to-end runs. Instead of
calling a real API, the mock inspects the conversation and returns a scripted
response that emulates a given :class:`MockSkill` level. Crucially, it can be
seeded to trigger *realistic failure modes* (planning errors, wrong-tool
selection, silent misinterpretation) so the failure-diagnosis pipeline has
genuine failures to classify -- without spending a single API token.

In production experiments the mock is swapped for :class:`OpenAIBackend` by
changing one line in the runner config; everything downstream is identical.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import LLMBackend, LLMResponse, TokenUsage


class MockSkill(Enum):
    """Simulated competence levels.

    Each level controls the probability that the mock produces correct code,
    correct interpretation, and whether it injects a deliberate failure. This
    lets the framework exercise the full spectrum from "always fails planning"
    to "near-perfect", which is exactly what a failure-diagnosis benchmark
    needs in order to populate every cell of the failure taxonomy.
    """

    WEAK = "weak"        # high failure rate across all stages
    MEDIUM = "medium"    # intermittent failures, mostly execution/interpretation
    STRONG = "strong"    # rare failures, mostly silent interpretation errors
    ADVERSARIAL = "adversarial"  # deterministic targeted failures for testing


@dataclass
class _FailureProfile:
    p_planning_error: float
    p_tool_error: float
    p_exec_error: float
    p_interpret_error: float
    p_silent: float  # probability that an error is silent (code runs, wrong answer)


_PROFILES = {
    MockSkill.WEAK: _FailureProfile(0.30, 0.25, 0.30, 0.35, 0.55),
    MockSkill.MEDIUM: _FailureProfile(0.12, 0.10, 0.15, 0.22, 0.65),
    MockSkill.STRONG: _FailureProfile(0.04, 0.03, 0.05, 0.12, 0.80),
    MockSkill.ADVERSARIAL: _FailureProfile(0.0, 0.0, 0.0, 0.0, 0.0),  # scripted
}


_CODE_OK = (
    "import pandas as pd\n"
    "df = pd.read_csv('data.csv')\n"
    "result = df.groupby('category')['value'].sum().sort_values(ascending=False)\n"
    "print('ANSWER:', result.idxmax(), result.max())\n"
)

_CODE_PLANNING_FAIL = (
    "# plan: load and compute mean instead of required groupby sum\n"
    "import pandas as pd\n"
    "df = pd.read_csv('data.csv')\n"
    "result = df['value'].mean()\n"
    "print('ANSWER:', result)\n"
)

_CODE_TOOL_FAIL = (
    "# calling the wrong tool: tries a sklearn model where plain pandas suffices\n"
    "from sklearn.linear_model import LinearRegression\n"
    "import pandas as pd\n"
    "df = pd.read_csv('data.csv')\n"
    "model = LinearRegression()\n"
    "model.fit(df[['value']], df['value'])\n"
    "print('ANSWER:', model.coef_[0])\n"
)

_CODE_EXEC_FAIL = (
    "import pandas as pd\n"
    "df = pd.read_csv('data.csv')\n"
    "result = df.groupby('cat')['value'].sum()  # wrong column name 'cat'\n"
    "print('ANSWER:', result)\n"
)

_CODE_SILENT_FAIL = (
    "# looks correct but uses wrong aggregation (count vs sum) -> silent failure\n"
    "import pandas as pd\n"
    "df = pd.read_csv('data.csv')\n"
    "result = df.groupby('category')['value'].count().sort_values(ascending=False)\n"
    "print('ANSWER:', result.idxmax(), result.max())\n"
)


def _stable_rand(seed_key: str, salt: str = "") -> float:
    """Deterministic pseudo-random in [0,1) from a string seed.

    Determinism is essential: the same (model, task, step) always yields the
    same outcome, so runs are reproducible and the failure taxonomy can be
    validated against fixed expectations.
    """
    h = hashlib.sha256(f"{seed_key}|{salt}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class MockLLM(LLMBackend):
    """A scripted, deterministic LLM for zero-cost reproducible runs.

    The mock recognises the agent's ReAct phases (``thought``/``action``/code
    blocks / final ``ANSWER:``) and responds accordingly. The failure profile
    of the chosen skill decides, per step, whether to emit a correct or failed
    code/action, and whether the failure will be silent.
    """

    def __init__(
        self,
        skill: MockSkill = MockSkill.MEDIUM,
        model: Optional[str] = None,
        price_in: float = 0.0,
        price_out: float = 0.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        force_failure: Optional[str] = None,
    ):
        super().__init__(
            model=model or f"mock-{skill.value}",
            price_in=price_in,
            price_out=price_out,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.skill = skill
        self.profile = _PROFILES[skill]
        self.force_failure = force_failure  # one of planning/tool/exec/silent/None
        self._step = 0

    def _pick_outcome(self, seed_key: str) -> str:
        """Decide which outcome to emit, deterministically.

        Returns one of: ``ok``, ``planning``, ``tool``, ``exec``, ``silent``.
        """
        if self.force_failure is not None:
            return self.force_failure

        r = _stable_rand(seed_key, str(self._step))
        p = self.profile
        # cumulative thresholds
        cum = 0.0
        cum += p.p_planning_error
        if r < cum:
            return "planning"
        cum += p.p_tool_error
        if r < cum:
            return "tool"
        cum += p.p_exec_error
        if r < cum:
            return "exec"
        # remaining errors are interpretation; decide silent vs loud
        cum += p.p_interpret_error
        if r < cum:
            return "silent" if _stable_rand(seed_key, "silent") < p.p_silent else "exec"
        return "ok"

    def _generate(
        self, messages: List[Dict[str, str]], max_tokens: int = 1024, **kwargs
    ) -> LLMResponse:
        self._step += 1
        user_text = messages[-1].get("content", "") if messages else ""
        seed_key = user_text[:120]

        outcome = self._pick_outcome(seed_key)

        # If the prompt asks for a final answer (interpretation phase)
        if "FINAL_ANSWER" in user_text or "summarize" in user_text.lower():
            if outcome in ("silent", "planning", "tool"):
                # silent misinterpretation: confident but wrong
                text = "ANSWER: 42\n\nThe result is clearly 42 based on the analysis."
            else:
                text = "ANSWER: electronics 1500\n\nThe largest category is electronics with total 1500."
            usage = TokenUsage(prompt_tokens=len(user_text) // 4 + 50, completion_tokens=len(text) // 4 + 10)
            return LLMResponse(text=text, usage=usage, model=self.model)

        # code-generation / action phase
        code_map = {
            "ok": _CODE_OK,
            "planning": _CODE_PLANNING_FAIL,
            "tool": _CODE_TOOL_FAIL,
            "exec": _CODE_EXEC_FAIL,
            "silent": _CODE_SILENT_FAIL,
        }
        code = code_map[outcome]
        # wrap as a ReAct-style action with an embedded code block
        if outcome == "ok":
            thought = "I need to group by category and sum the value column."
        elif outcome == "planning":
            thought = "I think the question asks for the average value overall."
        elif outcome == "tool":
            thought = "I'll train a regression model to find the answer."
        elif outcome == "exec":
            thought = "I'll group by the category column (I'll abbreviate it as 'cat')."
        else:  # silent
            thought = "I'll group by category and aggregate the value column."

        text = f"Thought: {thought}\nAction:\n```python\n{code}\n```"
        usage = TokenUsage(
            prompt_tokens=len(user_text) // 4 + 80,
            completion_tokens=len(text) // 4 + 20,
        )
        resp = LLMResponse(text=text, usage=usage, model=self.model)
        resp.meta["outcome"] = outcome  # ground-truth label for validation
        return resp
