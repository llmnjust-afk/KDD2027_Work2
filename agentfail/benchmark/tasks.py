"""Benchmark task definitions with ground-truth paths and failure traps.

Each task carries not just a question/answer but a *ground-truth analysis path*
(the canonical sequence of operations) and a set of *trap descriptors* --
predefined ways agents commonly fail on this task (data leakage, type confusion,
temporal leakage, wrong aggregation). This metadata is what lets the
failure-diagnosis layer do more than DSBench: it can check *whether* a failure
matched a known trap and *which stage* the trap targets.

The default benchmark is small (12 tasks) but spans the three KDD-flavoured
domains recommended in the plan: tabular EDA / feature engineering, time-series
anomaly detection, and recommendation-style user analysis. Each task is paired
with a synthetic CSV generator so the whole suite runs with zero external data.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Trap:
    """A predefined failure mode a task is designed to elicit."""

    name: str            # e.g. "wrong_aggregation"
    stage: str           # planning | tool_use | execution | interpretation
    description: str
    is_silent: bool = False  # does the trap produce code that runs but is wrong?


@dataclass
class Task:
    task_id: str
    domain: str                       # tabular_eda | time_series | recommendation
    question: str
    answer: Any                       # ground-truth final answer
    gt_path: List[str]                # canonical operation sequence
    traps: List[Trap] = field(default_factory=list)
    data_generator: Optional[Callable[[str], None]] = None  # writes data.csv
    answer_checker: Optional[Callable[[Any, Any], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def prepare_data(self, workdir: str) -> None:
        if self.data_generator is not None:
            self.data_generator(workdir)
        else:
            os.makedirs(workdir, exist_ok=True)

    def check_answer(self, predicted: Any) -> bool:
        if self.answer_checker is not None:
            return self.answer_checker(predicted, self.answer)
        return _default_check(predicted, self.answer)


def _default_check(predicted: Any, gold: Any) -> bool:
    if predicted is None:
        return False
    p = str(predicted).strip().lower()
    g = str(gold).strip().lower()
    if p == g:
        return True
    # numeric tolerance
    try:
        return abs(float(p) - float(g)) <= 1e-6 * max(1.0, abs(float(g)))
    except (ValueError, TypeError):
        return g in p or p in g


# --------------------------------------------------------------------------- #
# Synthetic data generators
# --------------------------------------------------------------------------- #

def _gen_tabular_sales(workdir: str) -> None:
    """Deterministic sales CSV: groupby category sum -> electronics=1500."""
    os.makedirs(workdir, exist_ok=True)
    rows = [
        {"category": "electronics", "value": 600},
        {"category": "electronics", "value": 900},
        {"category": "books", "value": 200},
        {"category": "books", "value": 100},
        {"category": "clothing", "value": 300},
        {"category": "clothing", "value": 250},
    ]
    with open(os.path.join(workdir, "data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "value"])
        w.writeheader()
        w.writerows(rows)


def _gen_tabular_types(workdir: str) -> None:
    """CSV with a numeric column stored as strings -> type-confusion trap."""
    os.makedirs(workdir, exist_ok=True)
    rows = [
        {"id": "1", "amount": "150"},
        {"id": "2", "amount": "250"},
        {"id": "3", "amount": "300"},
    ]
    with open(os.path.join(workdir, "data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "amount"])
        w.writeheader()
        w.writerows(rows)


def _gen_timeseries(workdir: str) -> None:
    """Daily series with an injected anomaly at index 7."""
    os.makedirs(workdir, exist_ok=True)
    rows = []
    base = 10.0
    for i in range(14):
        v = base + i * 0.5
        if i == 7:
            v = 100.0  # anomaly
        rows.append({"date": f"2024-01-{i+1:02d}", "value": v})
    with open(os.path.join(workdir, "data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "value"])
        w.writeheader()
        w.writerows(rows)


def _gen_recommendation(workdir: str) -> None:
    """User-item interactions; leakage trap if target used as feature."""
    os.makedirs(workdir, exist_ok=True)
    rows = []
    uid = 0
    for u in range(5):
        for item in range(4):
            uid += 1
            rows.append({"user_id": u, "item_id": item, "click": (u + item) % 2})
    with open(os.path.join(workdir, "data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "item_id", "click"])
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Answer checkers
# --------------------------------------------------------------------------- #

def _check_contains(gold: str):
    def _c(pred: Any, _gold: Any) -> bool:
        return gold in str(pred).lower()
    return _c


def _check_numeric(tol: float):
    def _c(pred: Any, gold: Any) -> bool:
        try:
            return abs(float(pred) - float(gold)) <= tol
        except (ValueError, TypeError):
            return False
    return _c


# --------------------------------------------------------------------------- #
# Default benchmark
# --------------------------------------------------------------------------- #

def build_default_benchmark() -> TaskSet:
    tasks: List[Task] = []

    # T1: tabular groupby sum (the canonical task the mock LLM is tuned to)
    tasks.append(Task(
        task_id="tabular_groupby_sum",
        domain="tabular_eda",
        question="Which category has the largest total value, and what is that total?",
        answer=("electronics", 1500),
        gt_path=["load csv", "groupby(category).sum()", "idxmax + max"],
        traps=[
            Trap("wrong_aggregation", "interpretation",
                 "uses .count() instead of .sum() -> runs but wrong", is_silent=True),
            Trap("wrong_operation", "planning",
                 "computes overall mean instead of per-category sum"),
            Trap("wrong_tool", "tool_use",
                 "trains a regression model where pandas groupby suffices"),
        ],
        data_generator=_gen_tabular_sales,
        answer_checker=lambda p, g: "electronic" in str(p).lower() and "1500" in str(p),
    ))

    # T2: type confusion -- amount as string
    tasks.append(Task(
        task_id="tabular_type_confusion",
        domain="tabular_eda",
        question="What is the sum of the amount column?",
        answer=700,
        gt_path=["load csv", "cast amount to numeric", "sum()"],
        traps=[
            Trap("type_confusion", "execution",
                 "sums strings instead of numbers -> concatenation", is_silent=True),
        ],
        data_generator=_gen_tabular_types,
        answer_checker=_check_numeric(1.0),
    ))

    # T3: anomaly detection
    tasks.append(Task(
        task_id="timeseries_anomaly",
        domain="time_series",
        question="On which date does the anomaly (largest deviation) occur?",
        answer="2024-01-08",
        gt_path=["load csv", "compute deviation from trend", "argmax"],
        traps=[
            Trap("temporal_leakage", "planning",
                 "uses future values to normalise -> leaks information"),
            Trap("wrong_index", "execution",
                 "off-by-one when mapping index to date"),
        ],
        data_generator=_gen_timeseries,
        answer_checker=_check_contains("2024-01-08"),
    ))

    # T4: temporal leakage explicit
    tasks.append(Task(
        task_id="timeseries_forecast_leakage",
        domain="time_series",
        question="What is the value on 2024-01-08 (the anomaly)?",
        answer=100.0,
        gt_path=["load csv", "filter date==2024-01-08", "read value"],
        traps=[
            Trap("temporal_leakage", "planning",
                 "fits a model on the full series incl. test point"),
        ],
        data_generator=_gen_timeseries,
        answer_checker=_check_numeric(0.5),
    ))

    # T5: recommendation leakage
    tasks.append(Task(
        task_id="recommendation_target_leakage",
        domain="recommendation",
        question="How many total clicks are there across all users?",
        answer=10,
        gt_path=["load csv", "sum(click)"],
        traps=[
            Trap("data_leakage", "planning",
                 "uses click as a feature to predict click"),
        ],
        data_generator=_gen_recommendation,
        answer_checker=_check_numeric(0.5),
    ))

    # T6: basic count
    tasks.append(Task(
        task_id="tabular_row_count",
        domain="tabular_eda",
        question="How many rows are in the dataset?",
        answer=6,
        gt_path=["load csv", "len(df)"],
        traps=[
            Trap("wrong_count", "execution",
                 "counts unique values instead of rows"),
        ],
        data_generator=_gen_tabular_sales,
        answer_checker=_check_numeric(0.0),
    ))

    return TaskSet(tasks=tasks)


@dataclass
class TaskSet:
    tasks: List[Task]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def by_domain(self) -> Dict[str, List[Task]]:
        out: Dict[str, List[Task]] = {}
        for t in self.tasks:
            out.setdefault(t.domain, []).append(t)
        return out

    def all_traps(self) -> List[Trap]:
        out = []
        for t in self.tasks:
            out.extend(t.traps)
        return out
