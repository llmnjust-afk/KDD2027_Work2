#!/usr/bin/env python3
"""Standalone integrity validation for the AgentFail held-out task suite."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median
from typing import Any, Dict, List, Tuple

from agentfail.benchmark.heldout_task_generators import build_heldout_taskset


def _read(workdir: str, filename: str) -> List[Dict[str, str]]:
    with open(os.path.join(workdir, filename), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _argmax(values: Dict[Any, float]) -> Any:
    return min(values, key=lambda key: (-values[key], key))


def _solve(family: str, workdir: str) -> Tuple[str, float]:
    if family == "weighted_group_mean":
        rows = _read(workdir, "data.csv")
        numerator, denominator = defaultdict(float), defaultdict(float)
        for row in rows:
            numerator[row["group"]] += float(row["score"]) * float(row["weight"])
            denominator[row["group"]] += float(row["weight"])
        values = {key: numerator[key] / denominator[key] for key in numerator}

    elif family == "conditional_rate_by_cohort":
        rows = [row for row in _read(workdir, "data.csv") if row["eligible"] == "1"]
        successes, totals = defaultdict(int), defaultdict(int)
        for row in rows:
            successes[row["cohort"]] += int(row["converted"])
            totals[row["cohort"]] += 1
        values = {key: successes[key] / totals[key] for key in totals}

    elif family == "two_key_grouped_top_combination":
        grouped = defaultdict(float)
        for row in _read(workdir, "data.csv"):
            grouped[(row["region"], row["channel"])] += float(row["revenue"])
        pair = _argmax(grouped)
        return f"{pair[0]}/{pair[1]}", grouped[pair]

    elif family == "latest_record_dedup_then_aggregate":
        latest: Dict[str, Dict[str, str]] = {}
        for row in _read(workdir, "data.csv"):
            previous = latest.get(row["entity_id"])
            if previous is None or row["updated_at"] > previous["updated_at"]:
                latest[row["entity_id"]] = row
        values = defaultdict(float)
        for row in latest.values():
            values[row["category"]] += float(row["amount"])

    elif family == "rolling_window_max_date":
        rows = sorted(_read(workdir, "data.csv"), key=lambda row: row["date"])
        values = {rows[i]["date"]: sum(float(row["value"]) for row in rows[i - 4:i + 1])
                  for i in range(4, len(rows))}

    elif family == "largest_percent_change":
        values = {row["product"]: 100.0 * (float(row["final_value"]) - float(row["initial_value"]))
                  / float(row["initial_value"]) for row in _read(workdir, "data.csv")}

    elif family == "two_file_join_then_group_aggregate":
        regions = {row["customer_id"]: row["region"] for row in _read(workdir, "customers.csv")}
        values = defaultdict(float)
        for row in _read(workdir, "orders.csv"):
            values[regions[row["customer_id"]]] += float(row["quantity"]) * float(row["unit_price"])

    elif family == "null_aware_conditional_median":
        grouped = defaultdict(list)
        for row in _read(workdir, "data.csv"):
            if row["status"] == "active" and row["score"] != "":
                grouped[row["team"]].append(float(row["score"]))
        values = {key: median(scores) for key, scores in grouped.items()}

    elif family == "log_sessionization":
        grouped = defaultdict(list)
        for row in _read(workdir, "events.csv"):
            grouped[row["user"]].append(datetime.fromisoformat(row["timestamp"]))
        values = {}
        for user, timestamps in grouped.items():
            timestamps.sort()
            values[user] = 1 + sum((current - previous).total_seconds() > 30 * 60
                                   for previous, current in zip(timestamps, timestamps[1:]))

    elif family == "parsed_text_severity_source_aggregation":
        points = {"LOW": 1, "MEDIUM": 2, "HIGH": 4, "CRITICAL": 8}
        values = defaultdict(float)
        for row in _read(workdir, "logs.csv"):
            fields = dict(part.strip().split("=", 1) for part in row["message"].split(";"))
            values[fields["source"]] += points[fields["severity"]]

    else:
        raise AssertionError(f"no validator for family {family}")

    winner = _argmax(values)
    return str(winner), values[winner]


def _same_answer(computed: Tuple[str, float], expected: Any) -> bool:
    return (isinstance(expected, tuple) and len(expected) == 2
            and computed[0] == str(expected[0])
            and math.isclose(computed[1], float(expected[1]), rel_tol=1e-12, abs_tol=1e-12))


def main() -> None:
    taskset = build_heldout_taskset()
    tasks = list(taskset)
    ids = [task.task_id for task in tasks]
    assert len(tasks) == 100, f"expected 100 tasks, got {len(tasks)}"
    assert len(ids) == len(set(ids)), "task IDs are not unique"
    assert all(task_id.startswith("heldout_") for task_id in ids), "invalid task ID prefix"

    family_counts = Counter()
    with tempfile.TemporaryDirectory(prefix="agentfail-heldout-") as root:
        for task in tasks:
            family = task.metadata["family"]
            family_counts[family] += 1
            workdir = os.path.join(root, task.task_id)
            task.prepare_data(workdir)
            expected_files = task.metadata["files"]
            assert sorted(os.listdir(workdir)) == expected_files, f"file mismatch for {task.task_id}"

            computed = _solve(family, workdir)
            assert _same_answer(computed, task.answer), (
                f"ground truth mismatch for {task.task_id}: computed={computed}, stored={task.answer}"
            )
            canonical = f"ANSWER: {computed[0]} {computed[1]:.10g}"
            assert task.check_answer(canonical), f"checker rejected canonical answer for {task.task_id}"
            assert task.check_answer(task.answer), f"checker rejected stored answer for {task.task_id}"

    assert len(family_counts) == 10, f"expected 10 families, got {len(family_counts)}"
    assert set(family_counts.values()) == {10}, f"family counts are not all 10: {family_counts}"
    print(f"Validated {len(tasks)} deterministic held-out tasks with {len(ids)} unique IDs.")
    for family in sorted(family_counts):
        print(f"  {family}: {family_counts[family]}")
    print("All data preparation, independent ground truths, and answer checkers passed.")


if __name__ == "__main__":
    main()
