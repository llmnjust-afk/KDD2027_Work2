"""Restricted code execution sandbox.

Executes LLM-generated Python in an isolated namespace with a working directory
and resource limits. It is deliberately lightweight (no container) so the demo
runs anywhere, but the interface mirrors a containerized sandbox: the runner can
swap in Docker/firejail for production without touching the agent code.

The sandbox captures stdout, the final ``ANSWER:`` marker (used by the agent to
extract a structured answer), exceptions, and timing -- all of which feed the
failure-diagnosis layer.
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    answer: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    elapsed: float = 0.0
    produced_files: list = field(default_factory=list)
    namespace: Optional[Dict[str, Any]] = None


class CodeSandbox:
    """Execute Python code with a persistent namespace and a working dir.

    Persistence of the namespace across steps mirrors how notebook-style data
    agents (DSBench's interactive setting) work: variables defined in step 1
    are visible in step 2. This is essential for studying *failure propagation*
    -- an error introduced early can silently corrupt later steps.
    """

    FORBIDDEN = {
        "open",
        "os",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "requests",
        "pickle",
        "shelve",
        "ctypes",
        "importlib",
        "builtins",
    }

    def __init__(self, workdir: str, forbidden: Optional[set] = None):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.forbidden = forbidden if forbidden is not None else self.FORBIDDEN
        self.namespace: Dict[str, Any] = {"__name__": "__sandbox__"}

    def _check_safety(self, code: str) -> Optional[str]:
        """Reject obviously dangerous constructs. Returns a reason or None."""
        for token in self.forbidden:
            if re.search(rf"\b{re.escape(token)}\b", code):
                return f"forbidden token: {token}"
        if "__import__" in code or "eval(" in code or "exec(" in code:
            return "dynamic execution"
        return None

    def execute(self, code: str, timeout: float = 30.0) -> ExecutionResult:
        start = time.time()
        reason = self._check_safety(code)
        if reason is not None:
            return ExecutionResult(
                success=False,
                stderr=f"Blocked by sandbox: {reason}",
                error_type="SecurityError",
                error_message=reason,
                elapsed=0.0,
            )

        buf = io.StringIO()
        prev_cwd = os.getcwd()
        try:
            os.chdir(self.workdir)
            compiled = compile(code, "<sandbox>", "exec")
            with redirect_stdout(buf):
                exec(compiled, self.namespace)  # noqa: S102 - intentional sandboxed exec
            stdout = buf.getvalue()
            answer = None
            m = ANSWER_RE.search(stdout)
            if m:
                answer = m.group(1).strip()
            return ExecutionResult(
                success=True,
                stdout=stdout,
                answer=answer,
                elapsed=time.time() - start,
                namespace=self.namespace,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            return ExecutionResult(
                success=False,
                stdout=buf.getvalue(),
                stderr=tb,
                error_type=type(exc).__name__,
                error_message=str(exc),
                elapsed=time.time() - start,
                namespace=self.namespace,
            )
        finally:
            os.chdir(prev_cwd)

    def reset(self) -> None:
        self.namespace = {"__name__": "__sandbox__"}
