"""Deterministic, template-disjoint held-out tasks for AgentFail.

The suite contains ten composed analysis families with ten instances each.  It
uses only generated CSV data and is intentionally separate from the original
single-operation templates in :mod:`task_generators`.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import re
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .tasks import Task, TaskSet, Trap


def _seed(task_id: str) -> int:
    return int(hashlib.sha256(task_id.encode("ascii")).hexdigest()[:16], 16)


def _rng(task_id: str) -> random.Random:
    return random.Random(_seed(task_id))


def _csv_generator(files: Mapping[str, Sequence[Mapping[str, Any]]]) -> Callable[[str], None]:
    frozen = {name: [dict(row) for row in rows] for name, rows in files.items()}

    def generate(workdir: str) -> None:
        os.makedirs(workdir, exist_ok=True)
        for filename, rows in frozen.items():
            if not rows:
                raise ValueError(f"held-out generator cannot write empty file: {filename}")
            with open(os.path.join(workdir, filename), "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    return generate


def _numbers(value: Any) -> List[float]:
    if isinstance(value, (tuple, list)):
        values: List[float] = []
        for item in value:
            values.extend(_numbers(item))
        return values
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return [float(token) for token in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", str(value))]


def _pair_checker(label: str, value: float, tolerance: float = 1e-6) -> Callable[[Any, Any], bool]:
    label_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(str(label))}(?![A-Za-z0-9_])", re.I)

    def check(predicted: Any, _gold: Any) -> bool:
        if isinstance(predicted, (tuple, list)) and len(predicted) >= 2:
            label_ok = str(predicted[0]).strip().lower() == str(label).lower()
        else:
            label_ok = bool(label_pattern.search(str(predicted)))
        return label_ok and any(math.isclose(number, value, rel_tol=tolerance, abs_tol=tolerance)
                                for number in _numbers(predicted))

    return check


def _date_checker(date: str) -> Callable[[Any, Any], bool]:
    def check(predicted: Any, _gold: Any) -> bool:
        return date in str(predicted)

    return check


def _task(
    task_id: str,
    family: str,
    question: str,
    answer: Any,
    gt_path: List[str],
    files: Mapping[str, Sequence[Mapping[str, Any]]],
    checker: Callable[[Any, Any], bool],
    traps: List[Trap],
) -> Task:
    return Task(
        task_id=task_id,
        domain="heldout_compositional",
        question=question,
        answer=answer,
        gt_path=gt_path,
        traps=traps,
        data_generator=_csv_generator(files),
        answer_checker=checker,
        metadata={"family": family, "seed": _seed(task_id), "files": sorted(files)},
    )


def _weighted_group_mean(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    groups = ["amber", "cobalt", "jade", "sienna"]
    rows = [
        {"group": group, "score": rng.randint(20, 95), "weight": rng.randint(1, 9)}
        for group in groups for _ in range(rng.randint(4, 7))
    ]
    means = {
        group: sum(r["score"] * r["weight"] for r in rows if r["group"] == group)
        / sum(r["weight"] for r in rows if r["group"] == group)
        for group in groups
    }
    winner = min(groups, key=lambda group: (-means[group], group))
    value = means[winner]
    return _task(
        task_id, "weighted_group_mean",
        "In data.csv, compute each group's weight-adjusted mean score as "
        "sum(score * weight) / sum(weight). Which group is highest? Break ties "
        "alphabetically. Print 'ANSWER: <group> <mean>' to at least 4 decimals.",
        (winner, value),
        ["load data.csv", "group by group", "sum score*weight and weight", "divide", "maximum with alphabetical tie-break"],
        {"data.csv": rows}, _pair_checker(winner, value, 1e-4),
        [Trap("wrong_aggregation", "interpretation", "uses an unweighted group mean", True)],
    )


def _conditional_rate_by_cohort(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    cohorts = ["alpha", "beta", "gamma", "delta"]
    rows = []
    for cohort in cohorts:
        for _ in range(12):
            rows.append({"cohort": cohort, "eligible": rng.choice([0, 1, 1]), "converted": rng.randint(0, 1)})
    rates = {}
    for cohort in cohorts:
        eligible = [r for r in rows if r["cohort"] == cohort and r["eligible"] == 1]
        rates[cohort] = sum(r["converted"] for r in eligible) / len(eligible)
    winner = min(cohorts, key=lambda cohort: (-rates[cohort], cohort))
    return _task(
        task_id, "conditional_rate_by_cohort",
        "Using data.csv, restrict the denominator and numerator to eligible == 1. "
        "Which cohort has the highest converted==1 rate among eligible rows? Break "
        "ties alphabetically. Print 'ANSWER: <cohort> <rate>' as a 0-to-1 rate.",
        (winner, rates[winner]),
        ["load data.csv", "filter eligible == 1", "group by cohort", "mean converted", "maximum with alphabetical tie-break"],
        {"data.csv": rows}, _pair_checker(winner, rates[winner], 1e-4),
        [Trap("wrong_denominator", "interpretation", "includes ineligible rows in the rate denominator", True)],
    )


def _two_key_grouped_top(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    regions = ["east", "north", "south"]
    channels = ["direct", "partner", "web"]
    rows = [
        {"region": region, "channel": channel, "revenue": rng.randint(40, 400)}
        for region in regions for channel in channels for _ in range(rng.randint(3, 6))
    ]
    totals = {(region, channel): sum(r["revenue"] for r in rows if r["region"] == region and r["channel"] == channel)
              for region in regions for channel in channels}
    winner = min(totals, key=lambda key: (-totals[key], key))
    label = f"{winner[0]}/{winner[1]}"
    return _task(
        task_id, "two_key_grouped_top_combination",
        "In data.csv, sum revenue for every (region, channel) pair. Which pair has "
        "the largest total? Break ties by region then channel alphabetically. Print "
        "'ANSWER: <region>/<channel> <total>'.",
        (label, totals[winner]),
        ["load data.csv", "group by region and channel", "sum revenue", "maximum with two-key tie-break"],
        {"data.csv": rows}, _pair_checker(label, totals[winner]),
        [Trap("collapsed_key", "execution", "groups by only one of the two keys", True)],
    )


def _latest_dedup_aggregate(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    categories = ["basic", "plus", "pro"]
    rows = []
    latest: Dict[str, Dict[str, Any]] = {}
    start = datetime(2025, 2, 1, 9)
    for entity_num in range(15):
        entity = f"E{entity_num:02d}"
        for version in range(rng.randint(2, 4)):
            row = {
                "entity_id": entity,
                "updated_at": (start + timedelta(days=entity_num, hours=version * 3)).isoformat(),
                "category": rng.choice(categories),
                "amount": rng.randint(20, 300),
            }
            rows.append(row)
            latest[entity] = row
    rng.shuffle(rows)
    totals = {category: sum(r["amount"] for r in latest.values() if r["category"] == category)
              for category in categories}
    winner = min(categories, key=lambda category: (-totals[category], category))
    return _task(
        task_id, "latest_record_dedup_then_aggregate",
        "For each entity_id in data.csv, retain only the row with the latest ISO "
        "updated_at, then sum amount by that retained row's category. Which category "
        "has the largest sum? Break ties alphabetically. Print 'ANSWER: <category> <sum>'.",
        (winner, totals[winner]),
        ["load data.csv", "parse updated_at", "sort and retain latest per entity_id", "group retained rows by category", "sum amount and select maximum"],
        {"data.csv": rows}, _pair_checker(winner, totals[winner]),
        [Trap("duplicate_leakage", "execution", "aggregates all historical versions", True)],
    )


def _rolling_window_max_date(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    start = datetime(2025, 3, 1)
    rows = [{"date": (start + timedelta(days=i)).date().isoformat(), "value": rng.randint(5, 80)} for i in range(24)]
    windows = [(rows[i]["date"], sum(rows[j]["value"] for j in range(i - 4, i + 1))) for i in range(4, len(rows))]
    answer_date, answer_value = min(windows, key=lambda item: (-item[1], item[0]))
    return _task(
        task_id, "rolling_window_max_date",
        "Sort data.csv by date and compute the trailing 5-calendar-row sum including "
        "the current date; only dates with five observations count. On which ending "
        "date is that sum largest? Break ties by earliest date. Print 'ANSWER: <YYYY-MM-DD> <sum>'.",
        (answer_date, answer_value),
        ["load and sort data.csv by date", "compute complete trailing five-row sums", "maximum with earliest-date tie-break"],
        {"data.csv": rows}, _pair_checker(answer_date, answer_value),
        [Trap("window_alignment", "execution", "uses a forward or partial rolling window", True)],
    )


def _largest_percent_change(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    products = ["atlas", "birch", "coral", "dune", "ember", "fjord"]
    rows = []
    changes = {}
    for product in products:
        initial = rng.randint(40, 180)
        final = rng.randint(20, 260)
        rows.append({"product": product, "initial_value": initial, "final_value": final})
        changes[product] = 100.0 * (final - initial) / initial
    winner = min(products, key=lambda product: (-changes[product], product))
    return _task(
        task_id, "largest_percent_change",
        "For each product in data.csv, calculate signed percent change as "
        "100 * (final_value - initial_value) / initial_value. Which product has the "
        "largest change (not largest absolute change)? Break ties alphabetically. "
        "Print 'ANSWER: <product> <percent>'.",
        (winner, changes[winner]),
        ["load data.csv", "compute signed percent change per product", "select largest with alphabetical tie-break"],
        {"data.csv": rows}, _pair_checker(winner, changes[winner], 1e-3),
        [Trap("wrong_baseline", "interpretation", "divides by final rather than initial value", True),
         Trap("absolute_change", "planning", "selects largest absolute percentage movement", True)],
    )


def _two_file_join_group(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    regions = ["central", "coastal", "mountain", "plains"]
    customers = [{"customer_id": f"C{i:02d}", "region": regions[i % len(regions)]} for i in range(16)]
    lookup = {row["customer_id"]: row["region"] for row in customers}
    orders = []
    for order_num in range(45):
        customer = rng.choice(customers)["customer_id"]
        orders.append({"order_id": f"O{order_num:03d}", "customer_id": customer,
                       "quantity": rng.randint(1, 6), "unit_price": rng.randint(5, 75)})
    totals = {region: sum(r["quantity"] * r["unit_price"] for r in orders if lookup[r["customer_id"]] == region)
              for region in regions}
    winner = min(regions, key=lambda region: (-totals[region], region))
    return _task(
        task_id, "two_file_join_then_group_aggregate",
        "Join orders.csv to customers.csv on customer_id. Compute each order's gross "
        "value as quantity * unit_price, then sum gross value by customer region. Which "
        "region is largest? Break ties alphabetically. Print 'ANSWER: <region> <total>'.",
        (winner, totals[winner]),
        ["load both CSV files", "join orders to customers on customer_id", "compute quantity*unit_price", "group by region and sum", "select maximum"],
        {"customers.csv": customers, "orders.csv": orders}, _pair_checker(winner, totals[winner]),
        [Trap("join_multiplication", "execution", "uses a non-key join that duplicates orders", True)],
    )


def _null_conditional_median(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    teams = ["falcon", "heron", "kestrel", "osprey"]
    rows = []
    for team in teams:
        for _ in range(11):
            score: Any = rng.randint(10, 99)
            if rng.random() < 0.22:
                score = ""
            rows.append({"team": team, "status": rng.choice(["active", "active", "inactive"]), "score": score})
    medians = {}
    for team in teams:
        values = [int(r["score"]) for r in rows if r["team"] == team and r["status"] == "active" and r["score"] != ""]
        medians[team] = float(median(values))
    winner = min(teams, key=lambda team: (-medians[team], team))
    return _task(
        task_id, "null_aware_conditional_median",
        "In data.csv, first keep status == 'active'. For each team, ignore blank score "
        "cells and compute the median of the remaining scores. Which team has the "
        "highest median? Break ties alphabetically. Print 'ANSWER: <team> <median>'.",
        (winner, medians[winner]),
        ["load data.csv", "filter active rows", "drop blank scores within groups", "median by team", "select maximum"],
        {"data.csv": rows}, _pair_checker(winner, medians[winner]),
        [Trap("null_imputation", "execution", "treats blank scores as zero", True),
         Trap("filter_order", "planning", "computes medians before filtering status", True)],
    )


def _log_sessionization(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    users = ["alice", "bruno", "carmen", "devon"]
    rows = []
    expected = {}
    day = datetime(2025, 4, 10, 8)
    event_num = 0
    for user in users:
        timestamp = day + timedelta(minutes=rng.randint(0, 15))
        sessions = 1
        for event_index in range(rng.randint(8, 13)):
            gap = rng.choice([5, 10, 15, 20, 45, 70])
            timestamp += timedelta(minutes=gap)
            if event_index > 0 and gap > 30:
                sessions += 1
            rows.append({"event_id": f"L{event_num:03d}", "user": user, "timestamp": timestamp.isoformat()})
            event_num += 1
        expected[user] = sessions
    rng.shuffle(rows)
    winner = min(users, key=lambda user: (-expected[user], user))
    return _task(
        task_id, "log_sessionization",
        "Sort events.csv by user and timestamp. A user's first event starts a session, "
        "and a new session starts only when the gap from that user's previous event is "
        "strictly greater than 30 minutes. Which user has the most sessions? Break ties "
        "alphabetically. Print 'ANSWER: <user> <sessions>'.",
        (winner, expected[winner]),
        ["load events.csv", "parse and sort timestamps within user", "compute within-user gaps", "mark first events and gaps > 30 minutes", "count sessions and select maximum"],
        {"events.csv": rows}, _pair_checker(winner, expected[winner]),
        [Trap("global_gap", "execution", "computes gaps across users rather than within user", True),
         Trap("unsorted_time", "execution", "sessionizes in CSV row order", True)],
    )


def _parsed_text_aggregation(task_id: str, _: int) -> Task:
    rng = _rng(task_id)
    severities = {"LOW": 1, "MEDIUM": 2, "HIGH": 4, "CRITICAL": 8}
    sources = ["api", "batch", "cache", "worker"]
    rows = []
    scores = {source: 0 for source in sources}
    for event_num in range(40):
        severity = rng.choice(list(severities))
        source = rng.choice(sources)
        message = f"severity={severity}; source={source}; event=evt_{event_num:03d}"
        rows.append({"timestamp": f"2025-05-{event_num % 20 + 1:02d}T{event_num % 24:02d}:00:00", "message": message})
        scores[source] += severities[severity]
    winner = min(sources, key=lambda source: (-scores[source], source))
    return _task(
        task_id, "parsed_text_severity_source_aggregation",
        "Parse message in logs.csv, whose exact format is 'severity=<LEVEL>; "
        "source=<SOURCE>; event=<ID>'. Map LOW=1, MEDIUM=2, HIGH=4, CRITICAL=8 "
        "and sum severity points by parsed source. Which source has the largest total? "
        "Break ties alphabetically. Print 'ANSWER: <source> <points>'.",
        (winner, scores[winner]),
        ["load logs.csv", "parse severity and source fields from message", "map severity to points", "sum points by source", "select maximum"],
        {"logs.csv": rows}, _pair_checker(winner, scores[winner]),
        [Trap("text_not_parsed", "planning", "groups the complete message instead of extracted source", True),
         Trap("severity_count", "interpretation", "counts records instead of summing severity points", True)],
    )


_FAMILY_GENERATORS: List[Tuple[str, Callable[[str, int], Task]]] = [
    ("weighted_group_mean", _weighted_group_mean),
    ("conditional_rate_by_cohort", _conditional_rate_by_cohort),
    ("two_key_grouped_top_combination", _two_key_grouped_top),
    ("latest_record_dedup_then_aggregate", _latest_dedup_aggregate),
    ("rolling_window_max_date", _rolling_window_max_date),
    ("largest_percent_change", _largest_percent_change),
    ("two_file_join_then_group_aggregate", _two_file_join_group),
    ("null_aware_conditional_median", _null_conditional_median),
    ("log_sessionization", _log_sessionization),
    ("parsed_text_severity_source_aggregation", _parsed_text_aggregation),
]


def build_heldout_benchmark(n_per_family: int = 10) -> TaskSet:
    """Build the deterministic held-out suite (100 tasks by default)."""
    if n_per_family < 0:
        raise ValueError("n_per_family must be non-negative")
    tasks = []
    for family, generator in _FAMILY_GENERATORS:
        for variant in range(n_per_family):
            task_id = f"heldout_{family}_{variant:02d}"
            tasks.append(generator(task_id, variant))
    return TaskSet(tasks=tasks)


def build_heldout_taskset() -> TaskSet:
    """Explicit 100-task alias for callers that use TaskSet terminology."""
    return build_heldout_benchmark(10)


__all__ = ["build_heldout_benchmark", "build_heldout_taskset"]
