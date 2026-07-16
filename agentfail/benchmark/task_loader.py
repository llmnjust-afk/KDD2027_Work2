"""Serialise/deserialise a TaskSet to JSON for sharing and review."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import List

from .tasks import Task, TaskSet, Trap


def save_taskset(ts: TaskSet, path: str) -> None:
    data = {
        "tasks": [
            {
                "task_id": t.task_id,
                "domain": t.domain,
                "question": t.question,
                "answer": t.answer,
                "gt_path": t.gt_path,
                "traps": [asdict(tr) for tr in t.traps],
                "metadata": t.metadata,
            }
            for t in ts.tasks
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_taskset(path: str) -> TaskSet:
    with open(path) as f:
        data = json.load(f)
    tasks: List[Task] = []
    for t in data["tasks"]:
        traps = [Trap(**tr) for tr in t.get("traps", [])]
        tasks.append(Task(
            task_id=t["task_id"],
            domain=t["domain"],
            question=t["question"],
            answer=t["answer"],
            gt_path=t["gt_path"],
            traps=traps,
            metadata=t.get("metadata", {}),
        ))
    return TaskSet(tasks=tasks)
