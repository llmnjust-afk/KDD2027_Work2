# Failure Annotation Guide (v3 — Five-Stage Taxonomy)

## Task

For each failed trace, **independently** assign three labels:

1. **is_silent**: `True` (code ran without error but answer is wrong) | `False` (code crashed)
2. **stage**: `interpretation` | `execution` | `planning`
3. **subcategory**: specific failure type (see below)

## Taxonomy

### Silent failures (is_silent = True)

- **interpretation**: Code executed successfully but the agent produced a wrong answer.
  - `wrong_aggregation`: Used wrong function (e.g., `.count()` instead of `.sum()`)
  - `wrong_column`: Selected the wrong column or field
  `misread_output`: Misinterpreted the execution output
  - `hallucinated_answer`: Answer not supported by code output
  - `no_answer_printed`: Code ran but no ANSWER was extracted
  - `type_confusion`: Mixed up data types (e.g., string vs numeric)

### Loud failures (is_silent = False)

- **execution**: Code raised an exception.
  - `runtime_error`: General runtime exception
  - `key_error`: KeyError (wrong column name)
  - `type_error`: TypeError (incompatible operations)
  - `security_block`: Sandbox blocked the operation

### Planning failures

- **planning**: Agent chose the wrong analytical approach.
  - `wrong_operation`: Selected wrong statistical operation
  - `temporal_leakage`: Used future data in time-series analysis

## Rules

1. **Annotate INDEPENDENTLY** — do not discuss with other annotators.
2. **Do NOT look at the system labels** — they are hidden and should not influence you.
3. If `final_answer` is `None` or empty, it usually means **execution** (loud) failure.
4. If `final_answer` has a value but it doesn't match `gt_answer`, it usually means **interpretation** (silent) failure.
5. If the code crashes (stderr contains traceback/exception), it is **execution** (loud).
6. If the code runs (no exception) but the answer is wrong, it is **interpretation** (silent).
7. If you are unsure between two categories, choose the one that best reflects **where the error first occurs**.
8. Use the `notes` field to explain ambiguous cases.

## Example

### Example 1: Silent failure

```
Task: What is the average sepal_length for each species?
Code: df.groupby('species')['sepal_length'].count()  # WRONG: count not mean
Output: setosa 50  # code ran, but wrong answer
Answer: setosa 50
GT: setosa 5.006

Annotation:
  is_silent: True
  stage: interpretation
  subcategory: wrong_aggregation
  notes: "Used .count() instead of .mean()"
```

### Example 2: Loud failure

```
Task: What is the survival rate on Titanic?
Code: df['survived'].mean()  # column name is 'Survived' not 'survived'
Output: KeyError: 'survived'
Answer: None
GT: 0.384

Annotation:
  is_silent: False
  stage: execution
  subcategory: key_error
  notes: "Column name case mismatch"
```

## Output Format

For each trace, fill in:

```json
{
  "stage": "interpretation|execution|planning",
  "is_silent": "True|False",
  "subcategory": "wrong_aggregation|key_error|...",
  "notes": "brief explanation"
}
```
