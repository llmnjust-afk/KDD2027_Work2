# Failure Annotation Guide

## Task
For each failed trace, independently assign:
1. **Stage**: planning | tool_use | execution | interpretation
2. **Category**: specific failure type (see taxonomy)
3. **is_silent**: True (code ran, answer wrong) | False (code crashed)

## Taxonomy
- planning: wrong_operation, wrong_decomposition, temporal_leakage, data_leakage
- tool_use: wrong_tool, wrong_params, over_privileged
- execution (loud): runtime_error, type_error, key_error, security_block
- interpretation (silent): wrong_aggregation, wrong_index, misread_output, hallucinated_answer

## Rules
- Annotate INDEPENDENTLY
- final_answer=None usually means execution (loud)
- final_answer=value but wrong usually means interpretation (silent)
- System's classification is for reference only; do not be biased by it
