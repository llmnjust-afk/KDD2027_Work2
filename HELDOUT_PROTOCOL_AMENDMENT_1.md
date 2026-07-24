# Held-Out Protocol Amendment 1: Additional Model Coverage

This amendment is recorded before sample locking, human annotation, or any
execution of the frozen classifier on held-out traces.

The preregistered three-model collection produced 67 failed traces among 298
persisted runs (GPT-4o: 17/100 failures; GPT-4o-mini: 50/98; DeepSeek-chat:
0/100). This is below the minimum of 100 failures required for the human
held-out set. Two interrupted GPT-4o-mini tasks will first be resumed under the
unchanged configuration.

To meet the minimum without lowering the threshold, selecting by predicted
stage, or over-representing repeated runs from one model, we add the pinned
`qwen3-max-2026-01-23` model on the same frozen 100 tasks. Sampling remains
based only on correctness and the preregistered SHA-256 order. The classifier,
taxonomy, tasks, annotation guide, seed, target, and minimum sample size remain
unchanged. Qwen collection results will be included in the model cap and family
coverage rules already specified by `HELDOUT_PROTOCOL.md`.
