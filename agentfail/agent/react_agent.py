"""ReAct-style data-science agent with full trace logging.

The agent runs a Thought -> Action(code) -> Observation loop, mirroring the
DSBench interactive setting. Every step is recorded in an :class:`AgentTrace`,
which is the single source of truth for the downstream failure-diagnosis layer:
the taxonomy classifier, propagation analyser, and causal-replay module all
operate on this trace, never on the raw LLM.

This explicit, structured trace is the core difference from DSBench, which only
kept a final ``response``/``summary``. Without per-step traces you cannot localise
*where* a failure happened or measure *how far* it propagated.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.base import LLMBackend, TokenUsage
from .sandbox import CodeSandbox, ExecutionResult


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?:\nAction:|$)", re.DOTALL)
ANSWER_LINE_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)


@dataclass
class TraceStep:
    """One agent step: an LLM call + its parsed action + the execution result."""

    step_index: int
    thought: str = ""
    action_type: str = ""  # "code" | "final_answer" | "none"
    code: str = ""
    raw_llm_text: str = ""
    llm_usage: TokenUsage = field(default_factory=TokenUsage)
    execution: Optional[ExecutionResult] = None
    timestamp: float = 0.0
    # ground-truth label injected by MockLLM (None for real backends)
    gt_outcome: Optional[str] = None
    # whether this step was a (self-)recovery from a prior step's failure
    is_recovery_attempt: bool = False


@dataclass
class AgentTrace:
    """Complete record of one task run."""

    task_id: str = ""
    model: str = ""
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    terminated: bool = False
    termination_reason: str = ""
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    wall_time: float = 0.0

    def total_prompt_tokens(self) -> int:
        return sum(s.llm_usage.prompt_tokens for s in self.steps)

    def total_completion_tokens(self) -> int:
        return sum(s.llm_usage.completion_tokens for s in self.steps)

    def num_steps(self) -> int:
        return len(self.steps)


class ReActAgent:
    """A ReAct agent that writes and executes code to solve data-science tasks.

    Parameters
    ----------
    llm : LLMBackend
        Any backend (mock or real) implementing the shared contract.
    sandbox : CodeSandbox
        Where generated code is executed.
    max_steps : int
        Horizon cap. Failures that propagate beyond this are counted as
        long-horizon propagation failures.
    """

    SYSTEM_PROMPT = (
        "You are a data-science agent. Solve the task by writing and executing Python code.\n"
        "The data file is in the current working directory. It may be 'data.csv' (CSV) "
        "or 'data.xlsx' (Excel with multiple sheets). Check which file exists first.\n"
        "For CSV: df = pd.read_csv('data.csv')\n"
        "For Excel: sheets = pd.read_excel('data.xlsx', sheet_name=None) "
        "to get all sheets, then inspect sheet names and columns.\n"
        "Each step: emit 'Thought: <reasoning>' then 'Action:' with a python code block.\n"
        "Your code MUST print the final result using exactly: print('ANSWER:', <value>)\n"
        "When you see the answer in the output, emit 'ANSWER: <final answer>' to finish.\n"
        "Do NOT use hypothetical data. Use the real data file."
    )

    def __init__(
        self,
        llm: LLMBackend,
        sandbox: CodeSandbox,
        max_steps: int = 8,
        verifier: Optional[Any] = None,
    ):
        self.llm = llm
        self.sandbox = sandbox
        self.max_steps = max_steps
        # optional failure-aware verifier (the method contribution)
        self.verifier = verifier

    def _parse(self, text: str) -> Dict[str, Any]:
        thought_m = THOUGHT_RE.search(text)
        thought = thought_m.group(1).strip() if thought_m else ""
        code_blocks = CODE_BLOCK_RE.findall(text)
        ans_m = ANSWER_LINE_RE.search(text)
        if ans_m and not code_blocks:
            return {"thought": thought, "action_type": "final_answer",
                    "code": "", "answer": ans_m.group(1).strip()}
        if code_blocks:
            return {"thought": thought, "action_type": "code",
                    "code": code_blocks[0], "answer": None}
        return {"thought": thought, "action_type": "none", "code": "", "answer": None}

    def _build_messages(self, task: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": (
            f"Task: {task}\n\n"
            "The data file 'data.csv' is in the working directory. "
            "Write Python code to load and analyze it. "
            "Print your final result as: print('ANSWER:', value)"
        )})
        messages.extend(history)
        return messages

    def run(self, task_id: str, task_description: str) -> AgentTrace:
        trace = AgentTrace(task_id=task_id, model=self.llm.model)
        t0 = time.time()
        history: List[Dict[str, str]] = []
        self.sandbox.reset()

        for i in range(self.max_steps):
            messages = self._build_messages(task_description, history)
            resp = self.llm.generate(messages)
            parsed = self._parse(resp.text)

            step = TraceStep(
                step_index=i,
                thought=parsed["thought"],
                action_type=parsed["action_type"],
                code=parsed["code"],
                raw_llm_text=resp.text,
                llm_usage=resp.usage,
                timestamp=time.time() - t0,
                gt_outcome=resp.meta.get("outcome"),
            )
            trace.steps.append(step)
            trace.total_tokens = trace.total_tokens + resp.usage

            # final answer path
            if parsed["action_type"] == "final_answer":
                trace.final_answer = parsed["answer"]
                trace.terminated = True
                trace.termination_reason = "final_answer"
                break

            if parsed["action_type"] == "code":
                exec_res = self.sandbox.execute(parsed["code"])
                step.execution = exec_res

                # optional failure-aware verification (method contribution)
                if self.verifier is not None:
                    verdict = self.verifier.check(step, exec_res, trace)
                    if verdict.needs_retry and i < self.max_steps - 1:
                        step.is_recovery_attempt = False
                        history.append({"role": "assistant", "content": resp.text})
                        history.append({
                            "role": "user",
                            "content": (
                                f"Observation:\n{(exec_res.stdout or '')[:2000]}\n"
                                f"Verifier flagged a possible issue: {verdict.reason}. "
                                f"Please revise your approach."
                            ),
                        })
                        continue

                # if execution produced an ANSWER marker, treat as candidate final
                if exec_res.success and exec_res.answer is not None:
                    # ask LLM to confirm/interpret the result
                    confirm_msg = messages + [
                        {"role": "assistant", "content": resp.text},
                        {"role": "user", "content":
                         f"Observation: {exec_res.stdout}\n"
                         "Based on this output, give the FINAL_ANSWER now."},
                    ]
                    final_resp = self.llm.generate(confirm_msg)
                    fa = ANSWER_LINE_RE.search(final_resp.text)
                    trace.final_answer = fa.group(1).strip() if fa else exec_res.answer
                    fstep = TraceStep(
                        step_index=i + 1,
                        thought="interpret result",
                        action_type="final_answer",
                        raw_llm_text=final_resp.text,
                        llm_usage=final_resp.usage,
                        timestamp=time.time() - t0,
                        gt_outcome=final_resp.meta.get("outcome"),
                    )
                    trace.steps.append(fstep)
                    trace.total_tokens = trace.total_tokens + final_resp.usage
                    trace.terminated = True
                    trace.termination_reason = "interpreted_answer"
                    break

                # no answer yet: feed observation back
                if exec_res.success:
                    raw_out = exec_res.stdout or ""
                    obs = raw_out[:2000] if raw_out else "(no output printed)"
                    if raw_out and "ANSWER:" not in raw_out:
                        obs += "\n\nNOTE: You did not print 'ANSWER: <value>'. Print your result explicitly."
                else:
                    obs = f"ERROR: {exec_res.error_type}: {exec_res.error_message[:500]}"
                history.append({"role": "assistant", "content": resp.text[:3000]})
                history.append({"role": "user", "content": f"Observation:\n{obs}"})

                # if execution failed, mark this as a recovery attempt opportunity
                if not exec_res.success:
                    step.is_recovery_attempt = True
            else:
                # no usable action; nudge
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content":
                                "Please provide a code action or a final ANSWER."})

        if not trace.terminated:
            trace.termination_reason = "max_steps"
        trace.wall_time = time.time() - t0
        return trace
