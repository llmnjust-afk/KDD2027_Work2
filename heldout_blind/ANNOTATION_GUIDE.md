# AgentFail Independent Held-Out Annotation Guide

Annotate independently. Do not use an LLM and do not inspect classifier output,
model identity, other annotators' forms, or the private directory.

Choose the earliest stage supported by evidence:

1. `runtime`: an actual execution exception or traceback.
2. `analytical_plan`: before relevant code executes, Thought selects the wrong
   analysis target or operation.
3. `code_generation`: Thought is correct, but code uses a wrong column,
   operation, implementation, or omits the promised computation/output.
4. `output_mismatch`: the agent misreads, omits, or reports a result unsupported
   by execution output. Misinterpreting empty stdout from an unprinted expression
   belongs here.
5. `answer_error`: plan, code, output, and report are internally consistent but
   wrong relative to the ground truth.

Stage and observability are independent. Set `is_silent=false` when the trace
explicitly exposes failure (exception, abstention, or clear error message), even
if its stage is Output-Mismatch or Code-Generation. Fill all fields and cite
specific Thought/Code/stdout evidence.
