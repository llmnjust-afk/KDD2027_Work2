# AgentFail Independent Held-Out Protocol

This protocol is frozen before collecting or annotating held-out traces.

## Frozen artifacts

- GitHub classifier release: `d406f5a0d623d2a7b1d1d0c55eb348b9d52b9217`
- `agentfail/diagnosis/classifier.py` SHA-256: `2e401764b4ed26e132941969d9e5ce84d005f3faed6945c3853fc423a22409ba`
- `agentfail/diagnosis/taxonomy.py` SHA-256: `3013513f62f0d402578538bb83049aabc3eacec5bd0e9c2eae94c55e218b22f5`
- Held-out task module SHA-256: `53d4f50cfd025359dde84b2379fe976bf7fb53ae18d5a5c88e25824789ab9b9f`

The classifier must not change until held-out gold labels and predictions are locked.

## Task suite

The suite contains 100 deterministic tasks: ten new compositional families with ten instances each. These templates do not occur in the original single-operation suite. All task IDs begin with `heldout_`.

## Collection

Run GPT-4o, GPT-4o-mini, and DeepSeek-chat once on every task at temperature zero and at most eight ReAct steps. Preserve complete traces. API failures are excluded from human sampling and reported separately.

## Sample lock

Before running the frozen classifier, select up to 120 failed traces by a deterministic SHA-256 ordering with seed `agentfail-heldout-2027`. Sampling may enforce a maximum of 45 traces per model and minimum family coverage, but must not use predicted stage, classifier confidence, or human labels. Save selected IDs and their SHA-256 checksum.

## Annotation

Three annotators independently label the locked traces while blinded to model identity, classifier output, and other annotators. Each form contains five anonymous duplicate QC traces. The revised guide treats stage and observability as independent. Raw annotations are locked before agreement calculation and consensus adjudication.

## Evaluation

Run the frozen classifier exactly once after final held-out gold is locked. Report stage accuracy, four-/five-class macro-F1 as applicable, per-class scores, observability accuracy, originating-step accuracy, and bootstrap confidence intervals. No classifier rule may be changed in response to held-out errors; any later modification requires a new test set.
