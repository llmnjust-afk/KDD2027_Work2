"""AgentFail: A Failure-Diagnosis Benchmark for Data-Science LLM Agents.

Implements a benchmark + failure-diagnosis framework that extends the
DSBench-style agent/sandbox architecture with:
  - a 4-stage failure taxonomy (Planning / Tool-use / Execution / Interpretation)
  - silent-failure detection and propagation-depth analysis
  - counterfactual causal replay for failure attribution
  - token-economics metrics (token-per-success, cost-accuracy frontier)
  - a method contribution: failure-aware adaptive retry / verification

The package runs end-to-end WITHOUT any API key via a deterministic mock LLM
that simulates different skill levels and triggers realistic failure modes,
so the entire pipeline is reproducible on a single CPU. A real OpenAI-compatible
backend can be swapped in for production experiments.
"""

__version__ = "0.1.0"
