"""Programmatic task generators for the AgentFail benchmark.

Generates 80 tasks across 5 domains (16 each) with deterministic answers and
designed failure traps. Each domain covers a distinct data-science skill and
targets different failure stages:

  tabular_eda       -> planning + interpretation traps (wrong aggregation)
  time_series       -> temporal leakage + execution traps
  recommendation    -> data leakage + tool-use traps
  statistical       -> interpretation + planning traps
  text_log          -> execution + interpretation traps

Tasks are parameterised so each domain produces 16 variants with different
data sizes, answer values, and trap combinations. This gives enough statistical
power (80 tasks × 3 repeats = 240 samples per model) for reliable conclusions.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
from typing import Any, Callable, Dict, List, Tuple

from .tasks import Task, Trap, TaskSet


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _seeded_rng(task_id: str, salt: str = "") -> random.Random:
    h = int(hashlib.sha256(f"{task_id}|{salt}".encode()).hexdigest()[:8], 16)
    return random.Random(h)


def _write_csv(workdir: str, rows: List[dict], filename: str = "data.csv") -> None:
    os.makedirs(workdir, exist_ok=True)
    if not rows:
        return
    with open(os.path.join(workdir, filename), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _check_contains(substr: str):
    def _c(pred: Any, _gold: Any) -> bool:
        return substr.lower() in str(pred).lower()
    return _c


def _check_numeric(tol: float):
    def _c(pred: Any, gold: Any) -> bool:
        try:
            return abs(float(pred) - float(gold)) <= tol
        except (ValueError, TypeError):
            return False
    return _c


def _check_multi(*checks):
    def _c(pred: Any, gold: Any) -> bool:
        return all(ch(pred, gold) for ch in checks)
    return _c


# --------------------------------------------------------------------------- #
# Domain 1: Tabular EDA (16 tasks)
# --------------------------------------------------------------------------- #

def _gen_tabular_eda(task_id: str, variant: int) -> Task:
    rng = _seeded_rng(task_id)
    categories = rng.sample(["electronics", "books", "clothing", "food", "toys"], 3)
    n_rows = rng.randint(8, 20)
    rows = []
    for i in range(n_rows):
        cat = rng.choice(categories)
        val = rng.randint(10, 500)
        rows.append({"category": cat, "value": val})

    sub_variant = variant % 4

    if sub_variant == 0:
        # groupby sum + idxmax
        sums = {}
        for r in rows:
            sums[r["category"]] = sums.get(r["category"], 0) + r["value"]
        answer_cat = max(sums, key=sums.get)
        answer_val = sums[answer_cat]
        question = f"Which category has the largest total value, and what is that total? Print 'ANSWER: <category> <total>'"
        gt_path = ["load csv", "groupby(category).sum()", "idxmax + max"]
        traps = [
            Trap("wrong_aggregation", "interpretation", "uses .count() instead of .sum()", is_silent=True),
            Trap("wrong_operation", "planning", "computes overall mean instead of per-category sum"),
        ]
        checker = lambda p, g: answer_cat in str(p).lower() and str(answer_val) in str(p)

    elif sub_variant == 1:
        # type confusion: amount as string
        rows = [{"id": str(i+1), "amount": str(rng.randint(10, 500))} for i in range(n_rows)]
        total = sum(int(r["amount"]) for r in rows)
        question = f"What is the sum of the amount column? Print 'ANSWER: <value>'"
        gt_path = ["load csv", "cast amount to numeric", "sum()"]
        traps = [Trap("type_confusion", "execution", "sums strings instead of numbers", is_silent=True)]
        checker = _check_numeric(1.0)

    elif sub_variant == 2:
        # null handling
        for i in range(0, len(rows), 3):
            rows[i]["value"] = None
        clean_sums = {}
        for r in rows:
            if r["value"] is not None:
                clean_sums[r["category"]] = clean_sums.get(r["category"], 0) + r["value"]
        answer_cat = max(clean_sums, key=clean_sums.get)
        answer_val = clean_sums[answer_cat]
        question = f"Some values are null. Which category has the largest total value (excluding nulls), and what is that total? Print 'ANSWER: <category> <total>'"
        gt_path = ["load csv", "dropna or fillna", "groupby(category).sum()", "idxmax"]
        traps = [
            Trap("null_ignored", "execution", "sums without handling nulls -> NaN", is_silent=False),
            Trap("wrong_aggregation", "interpretation", "uses count instead of sum on non-null", is_silent=True),
        ]
        checker = lambda p, g: answer_cat in str(p).lower() and str(answer_val) in str(p)

    else:
        # filtering then aggregation
        threshold = rng.randint(50, 200)
        filtered = [r for r in rows if r["value"] > threshold]
        cat_counts = {}
        for r in filtered:
            cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
        answer_cat = max(cat_counts, key=cat_counts.get) if cat_counts else categories[0]
        answer_val = cat_counts.get(answer_cat, 0)
        question = f"How many rows have value > {threshold} for each category? Which category has the most such rows, and how many? Print 'ANSWER: <category> <count>'"
        gt_path = ["load csv", f"filter value > {threshold}", "groupby(category).count()", "idxmax"]
        traps = [
            Trap("wrong_filter", "planning", f"uses >= instead of > {threshold}", is_silent=True),
            Trap("wrong_aggregation", "interpretation", "counts all rows not just filtered"),
        ]
        checker = lambda p, g: answer_cat in str(p).lower() and str(answer_val) in str(p)

    def gen(workdir, _rows=rows):
        _write_csv(workdir, _rows)

    return Task(
        task_id=task_id, domain="tabular_eda", question=question,
        answer=(answer_cat, answer_val) if sub_variant != 1 else total,
        gt_path=gt_path, traps=traps, data_generator=gen, answer_checker=checker,
    )


# --------------------------------------------------------------------------- #
# Domain 2: Time Series (16 tasks)
# --------------------------------------------------------------------------- #

def _gen_timeseries(task_id: str, variant: int) -> Task:
    rng = _seeded_rng(task_id)
    n_days = rng.randint(10, 20)
    base = rng.uniform(10, 50)
    trend = rng.uniform(0.1, 2.0)
    anomaly_idx = rng.randint(2, n_days - 2)
    anomaly_val = base + n_days * trend + rng.uniform(50, 100)

    rows = []
    for i in range(n_days):
        v = base + i * trend + rng.gauss(0, 1)
        if i == anomaly_idx:
            v = anomaly_val
        rows.append({"date": f"2024-01-{i+1:02d}", "value": round(v, 2)})

    sub_variant = variant % 4

    if sub_variant == 0:
        # anomaly detection by date
        question = "On which date does the anomaly (largest deviation from trend) occur? Print 'ANSWER: <date>'"
        gt_path = ["load csv", "compute deviation from trend", "argmax"]
        traps = [
            Trap("temporal_leakage", "planning", "uses future values to normalise"),
            Trap("wrong_index", "execution", "off-by-one when mapping index to date"),
        ]
        checker = _check_contains(f"2024-01-{anomaly_idx+1:02d}")
        answer = f"2024-01-{anomaly_idx+1:02d}"

    elif sub_variant == 1:
        # value lookup
        question = f"What is the value on 2024-01-{anomaly_idx+1:02d}? Print 'ANSWER: <value>'"
        gt_path = ["load csv", f"filter date==2024-01-{anomaly_idx+1:02d}", "read value"]
        traps = [Trap("temporal_leakage", "planning", "fits a model on the full series incl. test point")]
        checker = _check_numeric(abs(anomaly_val) * 0.01)
        answer = round(anomaly_val, 2)

    elif sub_variant == 2:
        # trend calculation
        from statistics import linear_regression
        xs = list(range(n_days))
        ys = [r["value"] for r in rows]
        slope = (sum((x - sum(xs)/n_days) * (y - sum(ys)/n_days) for x, y in zip(xs, ys)) /
                 sum((x - sum(xs)/n_days) ** 2 for x in xs))
        question = "What is the approximate linear trend (slope) of the value over time? Print 'ANSWER: <slope>'"
        gt_path = ["load csv", "linear regression or diff", "extract slope"]
        traps = [
            Trap("temporal_leakage", "planning", "uses rolling mean with future window"),
            Trap("wrong_operation", "planning", "computes mean instead of slope"),
        ]
        checker = _check_numeric(abs(slope) * 0.15 + 0.05)
        answer = round(slope, 4)

    else:
        # max value date
        max_idx = max(range(n_days), key=lambda i: rows[i]["value"])
        question = "On which date is the maximum value recorded? Print 'ANSWER: <date>'"
        gt_path = ["load csv", "idxmax on value", "map to date"]
        traps = [
            Trap("wrong_index", "execution", "uses argmin instead of argmax"),
            Trap("wrong_aggregation", "interpretation", "reports the value instead of the date"),
        ]
        checker = _check_contains(f"2024-01-{max_idx+1:02d}")
        answer = f"2024-01-{max_idx+1:02d}"

    def gen(workdir, _rows=rows):
        _write_csv(workdir, _rows)

    return Task(
        task_id=task_id, domain="time_series", question=question,
        answer=answer, gt_path=gt_path, traps=traps,
        data_generator=gen, answer_checker=checker,
    )


# --------------------------------------------------------------------------- #
# Domain 3: Recommendation / User Analysis (16 tasks)
# --------------------------------------------------------------------------- #

def _gen_recommendation(task_id: str, variant: int) -> Task:
    rng = _seeded_rng(task_id)
    n_users = rng.randint(5, 10)
    n_items = rng.randint(3, 6)
    rows = []
    for u in range(n_users):
        for item in range(n_items):
            click = 1 if (u + item + rng.randint(0, 1)) % 2 == 0 else 0
            rows.append({"user_id": u, "item_id": item, "click": click})

    sub_variant = variant % 4

    if sub_variant == 0:
        total = sum(r["click"] for r in rows)
        question = "How many total clicks are there across all users? Print 'ANSWER: <count>'"
        gt_path = ["load csv", "sum(click)"]
        traps = [Trap("data_leakage", "planning", "uses click as a feature to predict click")]
        checker = _check_numeric(0.5)

    elif sub_variant == 1:
        # per-user click count, find top user
        user_clicks = {}
        for r in rows:
            user_clicks[r["user_id"]] = user_clicks.get(r["user_id"], 0) + r["click"]
        top_user = max(user_clicks, key=user_clicks.get)
        top_count = user_clicks[top_user]
        question = "Which user has the most clicks, and how many? Print 'ANSWER: <user_id> <count>'"
        gt_path = ["load csv", "groupby(user_id).sum(click)", "idxmax"]
        traps = [
            Trap("wrong_aggregation", "interpretation", "counts rows instead of summing clicks", is_silent=True),
            Trap("wrong_tool", "tool_use", "trains an ML model instead of simple groupby"),
        ]
        checker = lambda p, g: str(top_user) in str(p) and str(top_count) in str(p)

    elif sub_variant == 2:
        # conversion rate
        user_clicks = {}
        user_total = {}
        for r in rows:
            user_clicks[r["user_id"]] = user_clicks.get(r["user_id"], 0) + r["click"]
            user_total[r["user_id"]] = user_total.get(r["user_id"], 0) + 1
        rates = {u: user_clicks[u] / user_total[u] for u in user_clicks}
        avg_rate = sum(rates.values()) / len(rates)
        question = "What is the average click rate (clicks per interaction) across all users? Print 'ANSWER: <rate>'"
        gt_path = ["load csv", "groupby(user_id).mean(click)", "mean of per-user rates"]
        traps = [
            Trap("wrong_aggregation", "interpretation", "computes overall mean without grouping first", is_silent=True),
            Trap("data_leakage", "planning", "uses item features that won't be available at inference"),
        ]
        checker = _check_numeric(0.05)
        answer = round(avg_rate, 4)

    else:
        # item popularity
        item_clicks = {}
        for r in rows:
            item_clicks[r["item_id"]] = item_clicks.get(r["item_id"], 0) + r["click"]
        popular = max(item_clicks, key=item_clicks.get)
        pop_count = item_clicks[popular]
        question = "Which item is most popular (most clicks), and how many clicks? Print 'ANSWER: <item_id> <count>'"
        gt_path = ["load csv", "groupby(item_id).sum(click)", "idxmax"]
        traps = [
            Trap("wrong_aggregation", "interpretation", "counts unique users per item instead of clicks"),
            Trap("wrong_tool", "tool_use", "builds a recommender model instead of counting"),
        ]
        checker = lambda p, g: str(popular) in str(p) and str(pop_count) in str(p)

    def gen(workdir, _rows=rows):
        _write_csv(workdir, _rows)

    return Task(
        task_id=task_id, domain="recommendation", question=question,
        answer=total if sub_variant == 0 else (locals().get("top_user", 0), locals().get("top_count", 0)) if sub_variant == 1 else locals().get("avg_rate", 0) if sub_variant == 2 else (locals().get("popular", 0), locals().get("pop_count", 0)),
        gt_path=gt_path, traps=traps, data_generator=gen, answer_checker=checker,
    )


# --------------------------------------------------------------------------- #
# Domain 4: Statistical Analysis (16 tasks)
# --------------------------------------------------------------------------- #

def _gen_statistical(task_id: str, variant: int) -> Task:
    rng = _seeded_rng(task_id)
    n = rng.randint(10, 25)
    sub_variant = variant % 4

    if sub_variant == 0:
        # correlation
        xs = [rng.uniform(0, 100) for _ in range(n)]
        ys = [x * 0.8 + rng.gauss(0, 5) for x in xs]
        rows = [{"x": round(x, 2), "y": round(y, 2)} for x, y in zip(xs, ys)]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
        std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
        std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
        corr = cov / (std_x * std_y) if std_x * std_y > 0 else 0
        question = "What is the Pearson correlation between x and y? Print 'ANSWER: <correlation>'"
        gt_path = ["load csv", "compute pearson correlation", "report value"]
        traps = [
            Trap("wrong_operation", "planning", "computes covariance instead of correlation"),
            Trap("wrong_aggregation", "interpretation", "reports |r| without sign"),
        ]
        checker = _check_numeric(0.05)
        answer = round(corr, 4)

    elif sub_variant == 1:
        # mean and std
        vals = [rng.gauss(50, 10) for _ in range(n)]
        rows = [{"value": round(v, 2) for v in vals}]
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        question = "What is the mean and standard deviation of the value column? Print 'ANSWER: <mean> <std>'"
        gt_path = ["load csv", "mean()", "std()"]
        traps = [
            Trap("wrong_operation", "planning", "computes median instead of mean"),
            Trap("wrong_aggregation", "interpretation", "uses sample std (ddof=1) vs population std (ddof=0)"),
        ]
        checker = lambda p, g: _check_numeric(0.5)(str(p).split()[0] if " " in str(p) else p, mean) and \
                                _check_numeric(0.5)(str(p).split()[-1] if " " in str(p) else p, std)
        answer = (round(mean, 2), round(std, 2))

    elif sub_variant == 2:
        # frequency / mode
        cats = rng.choices(["A", "B", "C", "D"], k=n)
        rows = [{"category": c, "count": rng.randint(1, 10)} for c in cats]
        freq = {}
        for r in rows:
            freq[r["category"]] = freq.get(r["category"], 0) + 1
        mode_cat = max(freq, key=freq.get)
        mode_count = freq[mode_cat]
        question = "Which category appears most frequently, and how many times? Print 'ANSWER: <category> <count>'"
        gt_path = ["load csv", "value_counts(category)", "idxmax"]
        traps = [
            Trap("wrong_aggregation", "interpretation", "sums the count column instead of counting rows"),
            Trap("wrong_operation", "planning", "finds the category with max sum(count) instead of max frequency"),
        ]
        checker = lambda p, g: mode_cat in str(p) and str(mode_count) in str(p)
        answer = (mode_cat, mode_count)

    else:
        # quantile / median
        vals = sorted([rng.uniform(0, 100) for _ in range(n)])
        rows = [{"value": round(v, 2) for v in vals}]
        median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        q1 = vals[n // 4]
        q3 = vals[3 * n // 4]
        question = "What is the median and interquartile range (Q3-Q1) of the value column? Print 'ANSWER: <median> <iqr>'"
        gt_path = ["load csv", "median()", "quantile(0.25) and quantile(0.75)", "subtract"]
        traps = [
            Trap("wrong_operation", "planning", "computes mean instead of median"),
            Trap("wrong_aggregation", "interpretation", "reports Q1 and Q3 separately instead of IQR"),
        ]
        checker = lambda p, g: _check_numeric(1.0)(
            str(p).split()[0] if " " in str(p) else p, median
        )
        answer = (round(median, 2), round(q3 - q1, 2))

    def gen(workdir, _rows=rows):
        _write_csv(workdir, _rows)

    return Task(
        task_id=task_id, domain="statistical", question=question,
        answer=answer, gt_path=gt_path, traps=traps,
        data_generator=gen, answer_checker=checker,
    )


# --------------------------------------------------------------------------- #
# Domain 5: Text / Log Analysis (16 tasks)
# --------------------------------------------------------------------------- #

def _gen_text_log(task_id: str, variant: int) -> Task:
    rng = _seeded_rng(task_id)
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    n = rng.randint(15, 30)
    sub_variant = variant % 4

    if sub_variant == 0:
        # error count
        rows = [{"timestamp": f"2024-01-01 {i:02d}:00:00", "level": rng.choice(levels), "message": f"event_{i}"}
                for i in range(n)]
        error_count = sum(1 for r in rows if r["level"] == "ERROR")
        question = "How many ERROR log entries are there? Print 'ANSWER: <count>'"
        gt_path = ["load csv", "filter level==ERROR", "count"]
        traps = [
            Trap("wrong_filter", "planning", "counts WARN instead of ERROR"),
            Trap("wrong_aggregation", "interpretation", "counts all non-INFO entries"),
        ]
        checker = _check_numeric(0.0)

    elif sub_variant == 1:
        # most common level
        rows = [{"timestamp": f"2024-01-01 {i:02d}:00:00", "level": rng.choice(levels), "message": f"event_{i}"}
                for i in range(n)]
        level_freq = {}
        for r in rows:
            level_freq[r["level"]] = level_freq.get(r["level"], 0) + 1
        common = max(level_freq, key=level_freq.get)
        common_count = level_freq[common]
        question = "Which log level is most common, and how many entries? Print 'ANSWER: <level> <count>'"
        gt_path = ["load csv", "value_counts(level)", "idxmax"]
        traps = [
            Trap("wrong_aggregation", "interpretation", "counts unique messages per level"),
            Trap("wrong_operation", "planning", "finds the longest message instead"),
        ]
        checker = lambda p, g: common in str(p) and str(common_count) in str(p)

    elif sub_variant == 2:
        # message pattern matching
        messages = [f"user_{rng.randint(1,5)} action_{rng.randint(1,3)}" for _ in range(n)]
        rows = [{"timestamp": f"2024-01-01 {i:02d}:00:00", "level": "INFO", "message": msg}
                for i, msg in enumerate(messages)]
        user_counts = {}
        for msg in messages:
            user = msg.split()[0]
            user_counts[user] = user_counts.get(user, 0) + 1
        top_user = max(user_counts, key=user_counts.get)
        top_count = user_counts[top_user]
        question = f"Which user (from the message column, format 'user_X ...') has the most log entries, and how many? Print 'ANSWER: <user> <count>'"
        gt_path = ["load csv", "extract user from message", "count per user", "idxmax"]
        traps = [
            Trap("wrong_operation", "planning", "counts timestamp frequency instead of parsing message"),
            Trap("wrong_aggregation", "interpretation", "reports the action instead of the user"),
        ]
        checker = lambda p, g: top_user in str(p) and str(top_count) in str(p)

    else:
        # time-bucketed count
        rows = [{"timestamp": f"2024-01-01 {i % 24:02d}:00:00", "level": rng.choice(levels), "message": f"event_{i}"}
                for i in range(n)]
        hour_counts = {}
        for r in rows:
            hour = r["timestamp"].split()[1][:2]
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        peak_hour = max(hour_counts, key=hour_counts.get)
        peak_count = hour_counts[peak_hour]
        question = "Which hour (00-23) has the most log entries, and how many? Print 'ANSWER: <hour> <count>'"
        gt_path = ["load csv", "extract hour from timestamp", "count per hour", "idxmax"]
        traps = [
            Trap("wrong_operation", "planning", "counts by level instead of by hour"),
            Trap("wrong_aggregation", "interpretation", "reports the timestamp instead of just the hour"),
        ]
        checker = lambda p, g: peak_hour in str(p) and str(peak_count) in str(p)

    def gen(workdir, _rows=rows):
        _write_csv(workdir, _rows)

    answer = locals().get("error_count", 0) if sub_variant == 0 else \
             (locals().get("common", ""), locals().get("common_count", 0)) if sub_variant == 1 else \
             (locals().get("top_user", ""), locals().get("top_count", 0)) if sub_variant == 2 else \
             (locals().get("peak_hour", ""), locals().get("peak_count", 0))

    return Task(
        task_id=task_id, domain="text_log", question=question,
        answer=answer, gt_path=gt_path, traps=traps,
        data_generator=gen, answer_checker=checker,
    )


# --------------------------------------------------------------------------- #
# Master generator
# --------------------------------------------------------------------------- #

_GENERATORS = {
    "tabular_eda": _gen_tabular_eda,
    "time_series": _gen_timeseries,
    "recommendation": _gen_recommendation,
    "statistical": _gen_statistical,
    "text_log": _gen_text_log,
}


def build_full_benchmark(n_per_domain: int = 16) -> TaskSet:
    """Build the full 80-task benchmark (5 domains × n_per_domain tasks)."""
    tasks: List[Task] = []
    for domain, gen_fn in _GENERATORS.items():
        for i in range(n_per_domain):
            task_id = f"{domain}_{i:02d}"
            tasks.append(gen_fn(task_id, i))
    return TaskSet(tasks=tasks)
