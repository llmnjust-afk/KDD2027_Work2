#!/usr/bin/env python3
"""Collect classifier-blind traces for the frozen AgentFail held-out suite."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from agentfail.agent.react_agent import ReActAgent
from agentfail.agent.sandbox import CodeSandbox
from agentfail.benchmark.heldout_task_generators import build_heldout_taskset
from agentfail.llm.openai_backend import OpenAIBackend


MODELS = {
    "gpt-4o": ("gpt-4o", 2.5, 10.0),
    "gpt-4o-mini": ("gpt-4o-mini", 0.15, 0.60),
    "deepseek-chat": ("deepseek-chat", 0.14, 0.28),
}


def atomic_save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def run_one(model_key, task, max_steps, api_key, base_url, timeout, api_retries):
    api_model, price_in, price_out = MODELS[model_key]
    llm = OpenAIBackend(
        model=api_model,
        api_key=api_key,
        base_url=base_url,
        price_in=price_in,
        price_out=price_out,
        timeout=timeout,
        max_retries=api_retries,
    )
    workdir = tempfile.mkdtemp(prefix=f"agentfail-heldout-{task.task_id}-")
    task.prepare_data(workdir)
    trace = ReActAgent(llm=llm, sandbox=CodeSandbox(workdir), max_steps=max_steps).run(
        task.task_id, task.question
    )
    return {
        "task_id": task.task_id,
        "family": task.metadata["family"],
        "model": model_key,
        "correct": task.check_answer(trace.final_answer),
        "final_answer": str(trace.final_answer)[:300] if trace.final_answer is not None else None,
        "tokens": trace.total_tokens.total_tokens,
        "gt_answer": task.answer,
        "gt_path": task.gt_path,
        "question": task.question,
        "trace": trace.to_dict(),
        "status": "OK",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--output", default="heldout_collection")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument("--run-retries", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Require 0 <= shard-index < shard-count")

    all_tasks = list(build_heldout_taskset())
    tasks = [
        task for index, task in enumerate(all_tasks)
        if index % args.shard_count == args.shard_index
    ]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{args.shard_index}" if args.shard_count > 1 else ""
    output_path = output_dir / f"{args.model}{suffix}.json"
    runs = []
    if args.resume and output_path.exists():
        runs = json.loads(output_path.read_text()).get("runs", [])
    completed = {run["task_id"] for run in runs}
    print(
        f"model={args.model} tasks={len(tasks)} resume={len(completed)} "
        f"shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )

    for index, task in enumerate(tasks, start=1):
        if task.task_id in completed:
            continue
        result = None
        last_error = None
        for attempt in range(args.run_retries + 1):
            try:
                result = run_one(
                    args.model,
                    task,
                    args.max_steps,
                    api_key,
                    base_url,
                    args.timeout,
                    args.api_retries,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < args.run_retries:
                    time.sleep(min(2 ** attempt, 8))
        if result is None:
            result = {
                "task_id": task.task_id,
                "family": task.metadata["family"],
                "model": args.model,
                "correct": False,
                "final_answer": None,
                "tokens": 0,
                "gt_answer": task.answer,
                "gt_path": task.gt_path,
                "question": task.question,
                "trace": None,
                "status": "ERR",
                "error": str(last_error),
            }
        runs.append(result)
        atomic_save(output_path, {"runs": runs})
        print(
            f"[{index}/{len(tasks)}] {task.task_id} status={result['status']} "
            f"correct={result['correct']} tokens={result['tokens']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
