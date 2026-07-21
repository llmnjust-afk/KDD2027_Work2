# Failure Annotation Guide (v3 — Five-Stage Taxonomy)

## Task

For each failed trace, **independently** assign three labels:

1. **is_silent**: `True` (code ran without error but answer is wrong) | `False` (code crashed)
2. **stage**: One of five stages (see below)
3. **subcategory**: Specific failure type (see below)

## Five-Stage Taxonomy

Stages are defined by **where the error is first detectable**, making them mutually exclusive.

### Silent failures (is_silent = True)

- **output_mismatch**: Code executed successfully but the agent extracted, computed, or reported a wrong answer.
  - `misread_output`: Misinterpreted the execution output
  - `hallucinated_answer`: Answer not supported by code output
  - `no_answer_printed`: Code ran but no ANSWER was extracted
  - `type_confusion`: Mixed up data types (e.g., string vs numeric)

- **answer_error**: Code runs and produces a result, but the result itself is wrong relative to ground truth.
  - `wrong_aggregation`: Used wrong function (e.g., `.count()` instead of `.sum()`)
  - `wrong_result`: Correct operation but wrong result due to data issues
  - `incomplete_analysis`: Analysis is incomplete or partial

### Loud failures (is_silent = False)

- **runtime**: Code raised an exception.
  - `runtime_error`: General runtime exception
  - `key_error`: KeyError (wrong column name)
  - `type_error`: TypeError (incompatible operations)
  - `security_block`: Sandbox blocked the operation

### Planning failures (is_silent varies)

- **analytical_plan**: Agent chose the wrong analytical approach.
  - `wrong_operation_plan`: Selected wrong statistical operation
  - `temporal_leakage`: Used future data in time-series analysis

### Code generation failures (is_silent varies)

- **code_generation**: Agent wrote code that embodies a wrong operation.
  - `wrong_column`: Selected the wrong column or field
  - `wrong_aggregation_code`: Code uses wrong aggregation function

## Disambiguation Rule

The classifier assigns the **first** stage at which the error becomes detectable:

- If the code uses the **wrong operation** (e.g., `.count()` instead of `.sum()`), it is **code_generation** — the error is detectable at code-writing time.
- If the code uses the **correct operation** but the result is wrong due to data issues, it is **answer_error**.
- If the code and result are **correct** but the agent **misreports** the final answer, it is **output_mismatch**.
- If the code **crashes**, it is **runtime** — regardless of whether the approach was also wrong.

## Rules

1. **Annotate INDEPENDENTLY** — do not discuss with other annotators.
2. **Do NOT look at the system labels** — they are hidden and should not influence you.
3. If `final_answer` is `None` or empty, it usually means **runtime** (loud) failure.
4. If `final_answer` has a value but doesn't match `gt_answer`, it is a silent failure — decide between **output_mismatch**, **answer_error**, and **code_generation**.
5. If the code crashes (stderr contains traceback/exception), it is **runtime** (loud).
6. Use the `notes` field to explain ambiguous cases.

## Output Format

For each trace, fill in:

```json
{
  "stage": "output_mismatch|answer_error|runtime|analytical_plan|code_generation",
  "is_silent": "True|False",
  "subcategory": "misread_output|wrong_aggregation|runtime_error|...",
  "notes": "brief explanation"
}
```
