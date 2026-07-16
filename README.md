# AgentFail: A Failure-Diagnosis Benchmark for Data-Science LLM Agents

A benchmark + failure-diagnosis framework that extends the DSBench-style
agent/sandbox architecture with structured failure localisation, causal
attribution, and token economics. Targets **KDD 2027** (Modern AI and Big Data
track).

## Why this exists

Existing data-science agent benchmarks (DSBench, DataSciBench, DSAEval,
LongDS-Bench) report only aggregate success rates. They do not answer:
- *Where* in the agent loop did the failure originate?
- Was it a **silent failure** (code ran, wrong answer) or a loud one?
- *How far* did the failure propagate before detection?
- *Which step* causally caused the wrong answer?
- How many **tokens per correct answer** did the agent cost?

AgentFail fills exactly these gaps.

## Innovation points

1. **4-stage failure taxonomy** (Planning / Tool-use / Execution / Interpretation)
   with an explicit silent-vs-loud distinction.
2. **Silent-failure detection** + propagation-depth analysis (long-horizon
   failure spread).
3. **Counterfactual causal replay** for failure attribution (re-run with a
   correct step; if outcome flips, that step caused the failure).
4. **Token-economics layer**: token-per-success, cost-accuracy frontier,
   invalid-token ratio.
5. **Method contribution**: a failure-aware adaptive verifier that triggers
   retries on likely silent failures, reducing SFR at low token overhead.

## Zero-cost reproducibility

The framework ships with a **deterministic MockLLM** that simulates weak /
medium / strong models and can inject realistic failure modes (planning errors,
wrong-tool selection, silent misaggregation). The entire pipeline runs
end-to-end on a single CPU with **no API key**. For real experiments, swap in
the OpenAI-compatible backend with a one-line config change.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
```

Sample output reports success rate ± std, silent-failure rate, token-per-success,
bootstrap 95% CIs, and paired t-tests (baseline vs verifier).

## Run tests

```bash
python agentfail_tests/test_e2e.py
```

## Project structure

```
agentfail/
  llm/            # LLM backends (mock + OpenAI-compatible) with token accounting
  agent/          # ReAct agent + code-execution sandbox with full trace logging
  benchmark/      # Task set with ground-truth paths + failure traps
  diagnosis/      # 4-stage taxonomy, classifier, propagation, causal replay
  metrics/        # failure metrics, token economics, statistical aggregation
  method/         # failure-aware adaptive verifier (method contribution)
  eval/           # end-to-end runner + experiment config
agentfail_tests/  # end-to-end smoke tests
run_demo.py       # one-command demo
```

## Switching to real models

Edit `agentfail/eval/config.py` to use `REAL_MODEL_GRID` and set the
`OPENAI_API_KEY` (or `DEEPSEEK_API_KEY`, etc.) environment variable. All
downstream diagnosis and metrics code is backend-agnostic.
