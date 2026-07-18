"""UCI real-task benchmark: real CSV datasets with analytical questions.

This is the KEY fix for external validity. DSBench tasks are too hard (100%
execution failure), so we cannot observe silent failures on them. UCI datasets
are simple tabular CSVs that agents CAN load and compute on -- but may still
get wrong answers (silent failures). This is the regime where our diagnosis
framework adds value.

Each task uses a REAL well-known dataset (Iris, Titanic, Boston, etc.) with a
deterministic analytical question whose answer we compute at task-build time.
The agent must load the real CSV and compute the correct answer. If it loads
wrong columns, uses wrong aggregation, or misreads output -> silent failure.

Target: 40-50 tasks across 8 datasets, success rate 30-60%, SFR > 30%.
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
from typing import List

from .tasks import Task, TaskSet, Trap


# --------------------------------------------------------------------------- #
# Dataset loaders (real data from public sources)
# --------------------------------------------------------------------------- #

_DATASET_CACHE = {}

DATASETS = {
    "iris": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "description": "Iris flower dataset: sepal/petal measurements by species",
    },
    "titanic": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        "description": "Titanic passengers: survival, class, age, fare",
    },
    "tips": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
        "description": "Restaurant tips: total bill, tip, day, time, party size",
    },
    "planets": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/planets.csv",
        "description": "Exoplanets: orbital period, mass, distance, method",
    },
    "flights": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
        "description": "Monthly airline passengers 1949-1960",
    },
    "exercise": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/exercise.csv",
        "description": "Exercise data: pulse, diet, kind of exercise, time",
    },
    "mpg": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
        "description": "Auto MPG: mpg, cylinders, displacement, horsepower, weight",
    },
    "penguins": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
        "description": "Palmer penguins: species, island, bill/flipper measurements, body mass",
    },
}


def _download_csv(url: str) -> str:
    """Download a CSV from URL, with caching."""
    if url in _DATASET_CACHE:
        return _DATASET_CACHE[url]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        _DATASET_CACHE[url] = content
        return content
    except Exception as e:
        print(f"WARNING: failed to download {url}: {e}")
        return None


def _parse_csv(csv_text: str):
    """Parse CSV text into list of dicts."""
    if not csv_text:
        return []
    reader = csv.DictReader(io.StringIO(csv_text))
    return list(reader)


def _make_data_gen(csv_text: str):
    """Create a data_generator that writes the real CSV."""
    def _gen(workdir: str, _csv=csv_text):
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, "data.csv"), "w") as f:
            f.write(_csv)
    return _gen


def _check_numeric(tol_frac=0.05, abs_tol=2.0):
    def _c(pred, gold):
        if pred is None:
            return False
        try:
            g = float(gold)
            return abs(float(pred) - g) <= abs(g) * tol_frac + abs_tol
        except (ValueError, TypeError):
            return str(gold) in str(pred).replace(",", "")
    return _c


def _check_contains(substr):
    def _c(pred, gold):
        if pred is None:
            return False
        return substr.lower() in str(pred).lower()
    return _c


def _check_multi(*subs):
    def _c(pred, gold):
        if pred is None:
            return False
        s = str(pred).lower()
        return all(sub.lower() in s for sub in subs)
    return _c


# --------------------------------------------------------------------------- #
# Task builders for each dataset
# --------------------------------------------------------------------------- #

def _build_iris_tasks(csv_text: str) -> List[Task]:
    rows = _parse_csv(csv_text)
    if not rows:
        return []
    tasks = []

    # T1: average sepal length per species, which is largest?
    from collections import defaultdict
    sl_by_species = defaultdict(list)
    for r in rows:
        try:
            sl_by_species[r["species"]].append(float(r["sepal_length"]))
        except (ValueError, KeyError):
            pass
    avg_sl = {k: sum(v)/len(v) for k, v in sl_by_species.items() if v}
    if avg_sl:
        largest_species = max(avg_sl, key=avg_sl.get)
        largest_val = round(avg_sl[largest_species], 2)
        tasks.append(Task(
            task_id="uci_iris_avg_sepal",
            domain="uci_real",
            question="What is the average sepal_length for each species? Which species has the largest average sepal_length and what is that value? Print 'ANSWER: <species> <value>'",
            answer=f"{largest_species} {largest_val}",
            gt_path=["read_csv", "groupby(species).mean(sepal_length)", "idxmax"],
            traps=[
                Trap("wrong_aggregation", "interpretation", "uses count instead of mean", is_silent=True),
                Trap("wrong_column", "execution", "uses petal_length instead of sepal_length"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(largest_species, str(largest_val)),
        ))

    # T2: correlation between sepal_length and petal_length
    sls, pls = [], []
    for r in rows:
        try:
            sls.append(float(r["sepal_length"]))
            pls.append(float(r["petal_length"]))
        except (ValueError, KeyError):
            pass
    if len(sls) > 5:
        n = len(sls)
        ms, mp = sum(sls)/n, sum(pls)/n
        cov = sum((a-ms)*(b-mp) for a,b in zip(sls,pls))/n
        ss = (sum((a-ms)**2 for a in sls)/n)**0.5
        sp = (sum((b-mp)**2 for b in pls)/n)**0.5
        corr = round(cov/(ss*sp), 4) if ss*sp > 0 else 0
        tasks.append(Task(
            task_id="uci_iris_corr",
            domain="uci_real",
            question="What is the Pearson correlation between sepal_length and petal_length? Print 'ANSWER: <correlation>'",
            answer=corr,
            gt_path=["read_csv", "pearson correlation", "report value"],
            traps=[
                Trap("wrong_operation", "planning", "computes covariance instead of correlation"),
                Trap("wrong_aggregation", "interpretation", "reports |r| without sign"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0.05, 0.05),
        ))

    # T3: count of each species
    species_counts = defaultdict(int)
    for r in rows:
        species_counts[r.get("species", "")] += 1
    if species_counts:
        most_common = max(species_counts, key=species_counts.get)
        count = species_counts[most_common]
        tasks.append(Task(
            task_id="uci_iris_species_count",
            domain="uci_real",
            question="How many flowers of each species are there? Which species is most common and how many? Print 'ANSWER: <species> <count>'",
            answer=f"{most_common} {count}",
            gt_path=["read_csv", "value_counts(species)", "idxmax"],
            traps=[
                Trap("wrong_aggregation", "interpretation", "counts all rows not per species"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(most_common, str(count)),
        ))

    # T4: max petal width
    pws = []
    for r in rows:
        try:
            pws.append(float(r["petal_width"]))
        except (ValueError, KeyError):
            pass
    if pws:
        max_pw = round(max(pws), 2)
        tasks.append(Task(
            task_id="uci_iris_max_petal_width",
            domain="uci_real",
            question="What is the maximum petal_width across all flowers? Print 'ANSWER: <value>'",
            answer=max_pw,
            gt_path=["read_csv", "max(petal_width)"],
            traps=[
                Trap("wrong_operation", "planning", "computes mean instead of max"),
                Trap("wrong_column", "execution", "uses sepal_width instead of petal_width"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0.01, 0.1),
        ))

    # T5: median sepal width
    sws = sorted([float(r["sepal_width"]) for r in rows if r.get("sepal_width","").replace(".","").isdigit()])
    if sws:
        n = len(sws)
        median = sws[n//2] if n%2==1 else (sws[n//2-1]+sws[n//2])/2
        median = round(median, 2)
        tasks.append(Task(
            task_id="uci_iris_median_sepal_width",
            domain="uci_real",
            question="What is the median sepal_width? Print 'ANSWER: <value>'",
            answer=median,
            gt_path=["read_csv", "median(sepal_width)"],
            traps=[
                Trap("wrong_operation", "planning", "computes mean instead of median"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0.05, 0.1),
        ))

    return tasks


def _build_titanic_tasks(csv_text: str) -> List[Task]:
    rows = _parse_csv(csv_text)
    if not rows:
        return []
    tasks = []
    from collections import defaultdict

    # T1: survival rate
    survived = []
    for r in rows:
        try:
            survived.append(int(float(r.get("survived", 0))))
        except (ValueError, KeyError):
            pass
    if survived:
        sr = round(sum(survived)/len(survived), 4)
        tasks.append(Task(
            task_id="uci_titanic_survival_rate",
            domain="uci_real",
            question="What is the overall survival rate (proportion who survived)? Print 'ANSWER: <rate>'",
            answer=sr,
            gt_path=["read_csv", "mean(survived)"],
            traps=[
                Trap("wrong_aggregation", "interpretation", "counts survivors instead of rate"),
                Trap("wrong_operation", "planning", "computes median instead of mean"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0.02, 0.02),
        ))

    # T2: survival rate by class
    by_class = defaultdict(list)
    for r in rows:
        try:
            by_class[int(float(r.get("pclass", 0)))].append(int(float(r.get("survived", 0))))
        except (ValueError, KeyError):
            pass
    if by_class:
        best_class = max(by_class, key=lambda c: sum(by_class[c])/len(by_class[c]))
        best_rate = round(sum(by_class[best_class])/len(by_class[best_class]), 4)
        tasks.append(Task(
            task_id="uci_titanic_survival_by_class",
            domain="uci_real",
            question="What is the survival rate for each passenger class (pclass)? Which class had the highest survival rate and what was it? Print 'ANSWER: <class> <rate>'",
            answer=f"{best_class} {best_rate}",
            gt_path=["read_csv", "groupby(pclass).mean(survived)", "idxmax"],
            traps=[
                Trap("wrong_aggregation", "interpretation", "counts passengers instead of rate", is_silent=True),
                Trap("wrong_grouping", "planning", "groups by sex instead of pclass"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(str(best_class), str(best_rate)),
        ))

    # T3: average fare
    fares = []
    for r in rows:
        try:
            fares.append(float(r.get("fare", 0)))
        except (ValueError, KeyError):
            pass
    if fares:
        avg_fare = round(sum(fares)/len(fares), 2)
        tasks.append(Task(
            task_id="uci_titanic_avg_fare",
            domain="uci_real",
            question="What is the average fare paid by passengers? Print 'ANSWER: <value>'",
            answer=avg_fare,
            gt_path=["read_csv", "mean(fare)"],
            traps=[
                Trap("wrong_operation", "planning", "computes median instead of mean"),
                Trap("wrong_aggregation", "interpretation", "computes total instead of average"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0.05, 2.0),
        ))

    # T4: most common embarkation port
    ports = defaultdict(int)
    for r in rows:
        p = r.get("embark_town", r.get("embarked", ""))
        if p:
            ports[p] += 1
    if ports:
        common_port = max(ports, key=ports.get)
        port_count = ports[common_port]
        tasks.append(Task(
            task_id="uci_titanic_common_port",
            domain="uci_real",
            question="Which embarkation town is most common and how many passengers boarded there? Print 'ANSWER: <town> <count>'",
            answer=f"{common_port} {port_count}",
            gt_path=["read_csv", "value_counts(embark_town)", "idxmax"],
            traps=[
                Trap("wrong_column", "execution", "uses wrong column name"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(common_port, str(port_count)),
        ))

    # T5: number of passengers
    tasks.append(Task(
        task_id="uci_titanic_passenger_count",
        domain="uci_real",
        question="How many passengers are in the dataset? Print 'ANSWER: <count>'",
        answer=len(rows),
        gt_path=["read_csv", "len(df)"],
        traps=[
            Trap("wrong_aggregation", "interpretation", "counts unique values instead of rows"),
        ],
        data_generator=_make_data_gen(csv_text),
        answer_checker=_check_numeric(0, 0),
    ))

    return tasks


def _build_tips_tasks(csv_text: str) -> List[Task]:
    rows = _parse_csv(csv_text)
    if not rows:
        return []
    tasks = []
    from collections import defaultdict

    # T1: average tip by day
    by_day = defaultdict(list)
    for r in rows:
        try:
            by_day[r["day"]].append(float(r["tip"]))
        except (ValueError, KeyError):
            pass
    if by_day:
        best_day = max(by_day, key=lambda d: sum(by_day[d])/len(by_day[d]))
        best_tip = round(sum(by_day[best_day])/len(by_day[best_day]), 2)
        tasks.append(Task(
            task_id="uci_tips_avg_tip_by_day",
            domain="uci_real",
            question="What is the average tip for each day? Which day has the highest average tip and what is it? Print 'ANSWER: <day> <tip>'",
            answer=f"{best_day} {best_tip}",
            gt_path=["read_csv", "groupby(day).mean(tip)", "idxmax"],
            traps=[
                Trap("wrong_aggregation", "interpretation", "uses sum instead of mean", is_silent=True),
                Trap("wrong_grouping", "planning", "groups by time instead of day"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(best_day, str(best_tip)),
        ))

    # T2: total bill
    bills = []
    for r in rows:
        try:
            bills.append(float(r["total_bill"]))
        except (ValueError, KeyError):
            pass
    if bills:
        total = round(sum(bills), 2)
        tasks.append(Task(
            task_id="uci_tips_total_bill",
            domain="uci_real",
            question="What is the total of all bills? Print 'ANSWER: <value>'",
            answer=total,
            gt_path=["read_csv", "sum(total_bill)"],
            traps=[
                Trap("wrong_operation", "planning", "computes mean instead of sum"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0.02, 5.0),
        ))

    # T3: tip percentage
    tips, bills_list = [], []
    for r in rows:
        try:
            t = float(r["tip"])
            b = float(r["total_bill"])
            if b > 0:
                tips.append(t)
                bills_list.append(b)
        except (ValueError, KeyError):
            pass
    if tips:
        avg_pct = round(sum(t/b for t,b in zip(tips, bills_list))/len(tips)*100, 2)
        tasks.append(Task(
            task_id="uci_tips_tip_percentage",
            domain="uci_real",
            question="What is the average tip percentage (tip/total_bill * 100)? Print 'ANSWER: <percentage>'",
            answer=avg_pct,
            gt_path=["read_csv", "compute tip/total_bill*100", "mean"],
            traps=[
                Trap("wrong_operation", "planning", "computes mean(tip)/mean(total_bill) instead of mean(tip/total_bill)"),
                Trap("wrong_aggregation", "interpretation", "forgets to multiply by 100"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0.05, 1.0),
        ))

    # T4: most common party size
    sizes = defaultdict(int)
    for r in rows:
        try:
            sizes[int(float(r["size"]))] += 1
        except (ValueError, KeyError):
            pass
    if sizes:
        common_size = max(sizes, key=sizes.get)
        size_count = sizes[common_size]
        tasks.append(Task(
            task_id="uci_tips_common_party_size",
            domain="uci_real",
            question="Which party size is most common and how many parties had that size? Print 'ANSWER: <size> <count>'",
            answer=f"{common_size} {size_count}",
            gt_path=["read_csv", "value_counts(size)", "idxmax"],
            traps=[
                Trap("wrong_column", "execution", "uses wrong column"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(str(common_size), str(size_count)),
        ))

    return tasks


def _build_mpg_tasks(csv_text: str) -> List[Task]:
    rows = _parse_csv(csv_text)
    if not rows:
        return []
    tasks = []
    from collections import defaultdict

    # T1: average mpg by origin
    by_origin = defaultdict(list)
    for r in rows:
        try:
            by_origin[r["origin"]].append(float(r["mpg"]))
        except (ValueError, KeyError):
            pass
    if by_origin:
        best_origin = max(by_origin, key=lambda o: sum(by_origin[o])/len(by_origin[o]))
        best_mpg = round(sum(by_origin[best_origin])/len(by_origin[best_origin]), 2)
        tasks.append(Task(
            task_id="uci_mpg_avg_by_origin",
            domain="uci_real",
            question="What is the average mpg for each origin? Which origin has the highest average mpg and what is it? Print 'ANSWER: <origin> <mpg>'",
            answer=f"{best_origin} {best_mpg}",
            gt_path=["read_csv", "groupby(origin).mean(mpg)", "idxmax"],
            traps=[
                Trap("wrong_aggregation", "interpretation", "uses sum instead of mean", is_silent=True),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(best_origin, str(best_mpg)),
        ))

    # T2: max horsepower
    hps = []
    for r in rows:
        try:
            hp = r.get("horsepower", "")
            if hp and hp.replace(".", "").replace("-", "").isdigit():
                hps.append(float(hp))
        except (ValueError, KeyError):
            pass
    if hps:
        max_hp = int(max(hps))
        tasks.append(Task(
            task_id="uci_mpg_max_hp",
            domain="uci_real",
            question="What is the maximum horsepower? Print 'ANSWER: <value>'",
            answer=max_hp,
            gt_path=["read_csv", "max(horsepower)"],
            traps=[
                Trap("wrong_operation", "planning", "computes mean instead of max"),
                Trap("type_confusion", "execution", "horsepower may be string, needs conversion"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0, 1),
        ))

    # T3: correlation mpg vs weight
    mpgs, weights = [], []
    for r in rows:
        try:
            mpgs.append(float(r["mpg"]))
            weights.append(float(r["weight"]))
        except (ValueError, KeyError):
            pass
    if len(mpgs) > 5:
        n = len(mpgs)
        mm, mw = sum(mpgs)/n, sum(weights)/n
        cov = sum((a-mm)*(b-mw) for a,b in zip(mpgs,weights))/n
        sm = (sum((a-mm)**2 for a in mpgs)/n)**0.5
        sw = (sum((b-mw)**2 for b in weights)/n)**0.5
        corr = round(cov/(sm*sw), 4) if sm*sw > 0 else 0
        tasks.append(Task(
            task_id="uci_mpg_corr_mpg_weight",
            domain="uci_real",
            question="What is the Pearson correlation between mpg and weight? Print 'ANSWER: <correlation>'",
            answer=corr,
            gt_path=["read_csv", "pearson correlation", "report"],
            traps=[
                Trap("wrong_operation", "planning", "computes covariance"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0.05, 0.05),
        ))

    # T4: number of cylinders most common
    cyls = defaultdict(int)
    for r in rows:
        try:
            cyls[int(float(r["cylinders"]))] += 1
        except (ValueError, KeyError):
            pass
    if cyls:
        common_cyl = max(cyls, key=cyls.get)
        cyl_count = cyls[common_cyl]
        tasks.append(Task(
            task_id="uci_mpg_common_cylinders",
            domain="uci_real",
            question="Which number of cylinders is most common and how many cars have it? Print 'ANSWER: <cylinders> <count>'",
            answer=f"{common_cyl} {cyl_count}",
            gt_path=["read_csv", "value_counts(cylinders)", "idxmax"],
            traps=[
                Trap("wrong_column", "execution", "uses wrong column"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(str(common_cyl), str(cyl_count)),
        ))

    return tasks


def _build_penguins_tasks(csv_text: str) -> List[Task]:
    rows = _parse_csv(csv_text)
    if not rows:
        return []
    tasks = []
    from collections import defaultdict

    # T1: average body mass by species
    by_species = defaultdict(list)
    for r in rows:
        try:
            by_species[r["species"]].append(float(r["body_mass_g"]))
        except (ValueError, KeyError):
            pass
    if by_species:
        heaviest = max(by_species, key=lambda s: sum(by_species[s])/len(by_species[s]))
        mass = round(sum(by_species[heaviest])/len(by_species[heaviest]), 0)
        tasks.append(Task(
            task_id="uci_penguins_avg_mass",
            domain="uci_real",
            question="What is the average body_mass_g for each species? Which species is heaviest on average and what is its average mass? Print 'ANSWER: <species> <mass>'",
            answer=f"{heaviest} {int(mass)}",
            gt_path=["read_csv", "groupby(species).mean(body_mass_g)", "idxmax"],
            traps=[
                Trap("wrong_aggregation", "interpretation", "uses count instead of mean", is_silent=True),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(heaviest, str(int(mass))),
        ))

    # T2: count by island
    by_island = defaultdict(int)
    for r in rows:
        island = r.get("island", "")
        if island:
            by_island[island] += 1
    if by_island:
        common_island = max(by_island, key=by_island.get)
        count = by_island[common_island]
        tasks.append(Task(
            task_id="uci_penguins_island_count",
            domain="uci_real",
            question="How many penguins are from each island? Which island has the most penguins and how many? Print 'ANSWER: <island> <count>'",
            answer=f"{common_island} {count}",
            gt_path=["read_csv", "value_counts(island)", "idxmax"],
            traps=[
                Trap("wrong_column", "execution", "uses species column instead of island"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(common_island, str(count)),
        ))

    # T3: max flipper length
    fls = []
    for r in rows:
        try:
            fls.append(float(r["flipper_length_mm"]))
        except (ValueError, KeyError):
            pass
    if fls:
        max_fl = int(max(fls))
        tasks.append(Task(
            task_id="uci_penguins_max_flipper",
            domain="uci_real",
            question="What is the maximum flipper_length_mm? Print 'ANSWER: <value>'",
            answer=max_fl,
            gt_path=["read_csv", "max(flipper_length_mm)"],
            traps=[
                Trap("wrong_operation", "planning", "computes mean instead of max"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0, 1),
        ))

    return tasks


def _build_flights_tasks(csv_text: str) -> List[Task]:
    rows = _parse_csv(csv_text)
    if not rows:
        return []
    tasks = []
    from collections import defaultdict

    # T1: total passengers per year, which year had most?
    by_year = defaultdict(int)
    for r in rows:
        try:
            by_year[int(r["year"])] += int(r["passengers"])
        except (ValueError, KeyError):
            pass
    if by_year:
        peak_year = max(by_year, key=by_year.get)
        peak_pax = by_year[peak_year]
        tasks.append(Task(
            task_id="uci_flights_peak_year",
            domain="uci_real",
            question="What is the total passengers per year? Which year had the most passengers and how many? Print 'ANSWER: <year> <passengers>'",
            answer=f"{peak_year} {peak_pax}",
            gt_path=["read_csv", "groupby(year).sum(passengers)", "idxmax"],
            traps=[
                Trap("wrong_aggregation", "interpretation", "uses mean instead of sum", is_silent=True),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_multi(str(peak_year), str(peak_pax)),
        ))

    # T2: average passengers in July
    jul_pax = []
    for r in rows:
        if r.get("month", "").lower().startswith("jul"):
            try:
                jul_pax.append(int(r["passengers"]))
            except (ValueError, KeyError):
                pass
    if jul_pax:
        avg_jul = round(sum(jul_pax)/len(jul_pax), 0)
        tasks.append(Task(
            task_id="uci_flights_avg_july",
            domain="uci_real",
            question="What is the average number of passengers in July across all years? Print 'ANSWER: <value>'",
            answer=int(avg_jul),
            gt_path=["read_csv", "filter month==July", "mean(passengers)"],
            traps=[
                Trap("wrong_filter", "planning", "filters wrong month"),
                Trap("wrong_aggregation", "interpretation", "uses sum instead of mean"),
            ],
            data_generator=_make_data_gen(csv_text),
            answer_checker=_check_numeric(0.05, 5.0),
        ))

    return tasks


# --------------------------------------------------------------------------- #
# Master builder
# --------------------------------------------------------------------------- #

_BUILDERS = {
    "iris": _build_iris_tasks,
    "titanic": _build_titanic_tasks,
    "tips": _build_tips_tasks,
    "mpg": _build_mpg_tasks,
    "penguins": _build_penguins_tasks,
    "flights": _build_flights_tasks,
}


def build_uci_benchmark() -> TaskSet:
    """Build real-data tasks from UCI/seaborn public datasets.

    Downloads real CSVs at build time and computes deterministic answers.
    These are simple enough for agents to load (CSV, not xlsx), but have
    enough complexity to trigger silent failures (wrong aggregation, wrong
    column, wrong filter).
    """
    all_tasks: List[Task] = []

    for name, builder in _BUILDERS.items():
        url = DATASETS[name]["url"]
        csv_text = _download_csv(url)
        if csv_text is None:
            print(f"  SKIP {name}: download failed")
            continue
        tasks = builder(csv_text)
        print(f"  {name}: {len(tasks)} tasks")
        all_tasks.extend(tasks)

    print(f"Built {len(all_tasks)} UCI real-data tasks")
    return TaskSet(tasks=all_tasks)
