# AgentFail Blind Annotation Guide

Annotate independently. Do not communicate trace-level decisions until all
forms are locked. Model identity, system predictions, and LLM pre-annotations
are intentionally hidden.

Use the first stage at which the failure is detectable:

1. `runtime`: code raises an exception or execution error. Silent = false.
2. `analytical_plan`: Thought states an incorrect approach before code.
3. `code_generation`: Thought is correct, but code implements a wrong operation.
4. `output_mismatch`: execution output supports one result, but the agent reports
   another result or hallucinates/misreads the output.
5. `answer_error`: code, output, and report are internally consistent, but the
   approach is wrong relative to the ground truth.

For every trace, fill all fields. Cite a Thought, Code, stdout, or error line in
`evidence`. Use `unclassifiable=true` only when the trace is insufficient.
