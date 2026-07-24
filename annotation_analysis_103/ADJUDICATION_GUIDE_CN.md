# AgentFail 103条新增标注仲裁指南

## 当前状态

- 三位标注者已锁定原始独立标注；不得覆盖或修改原文件。
- 103条正式trace中，28条stage全体一致，74条为2--1分歧，1条为三方分歧。
- 87条因stage、silent或originating step存在分歧而进入仲裁队列。
- 5个匿名重复QC样本在三位标注者内部均达到5/5一致。

## 需要填写的文件

仅填写 `adjudication_queue_103.json` 中每条末尾的 `adjudication`：

```json
{
  "gold_stage": "output_mismatch",
  "gold_silent": true,
  "gold_originating_step": 1,
  "adjudication_reason": "Step 1 misinterprets empty stdout from an unprinted boolean as evidence that data.csv is absent."
}
```

## 关键边界规则

按照“错误最早可由哪类证据检测”仲裁，而不是按照错误最终造成的结果仲裁：

1. `runtime`：执行产生异常、traceback或明确执行失败。
2. `analytical_plan`：在任何相关代码执行前，Thought独立提出错误的目标操作或分析方案，例如任务要求sum但初始计划明确提出mean。
3. `code_generation`：Thought中的操作正确，但随后代码实现了不同的操作或列。
4. `output_mismatch`：错误源于读取、遗漏或错误解释先前执行输出，包括把“表达式未print导致stdout为空”解释为文件不存在。
5. `answer_error`：计划、代码、输出和最终报告内部一致，但相对ground truth仍使用了错误的问题定义、过滤条件或统计方法。

因此，对于当前最常见的模式：

> Agent执行 `file_exists = os.path.isfile('data.csv'); file_exists`，因为没有调用print而stdout为空，随后Thought错误断言文件不存在。

建议归入 `output_mismatch`，因为错误来自对先前执行输出的错误解释，而不是执行前独立提出错误分析计划。

## 仲裁流程

1. 三位标注者共同查看原始trace及三份独立理由。
2. 先讨论5个代表性分歧，确认上述边界规则是否一致接受。
3. 逐条填写87条仲裁项；不得仅按多数票自动填入。
4. 对信息确实不足的trace，可设置 `gold_stage` 为 `unclassifiable`，但必须说明缺失证据。
5. 所有仲裁项完成后锁定文件，再运行最终gold生成与统计脚本。

## 不应修改

- 三份 `annotator_*_completed.json` 原始标注；
- 每条trace的question、ground truth、steps及三位标注者原始答案；
- `trace_id`。
