#!/usr/bin/env python3
"""Offline recomputation of oracle repair rate using expanded GT library.

Does NOT call any API. Re-executes the ground-truth code on the task's data
to check if oracle repair would fix each failed run. This is possible because
causal replay only needs (task, data) not the original LLM output.

Usage:
    python recompute_oracle_repair.py
"""
import sys, os, json, tempfile
sys.path.insert(0, '.')

from agentfail.diagnosis.causality import CausalReplay, _gt_signature, GT_CODE_LIBRARY
from agentfail.benchmark.task_generators import build_full_benchmark
from agentfail.benchmark.uci_tasks import build_uci_benchmark
from agentfail.benchmark.dsbench_real_adapter import build_dsbench_real_subset
from agentfail.agent.sandbox import CodeSandbox
from agentfail.diagnosis.classifier import FailureClassifier
from agentfail.agent.react_agent import AgentTrace

BASE = '/data/lab/KDD2027_Work2'

def build_all_tasks():
    """Build all tasks with their data generators."""
    tasks = {}
    # synthetic
    synth = build_full_benchmark(16)
    for t in synth.tasks:
        tasks[t.task_id] = t
    # UCI
    uci = build_uci_benchmark()
    for t in uci.tasks:
        tasks[t.task_id] = t
    # DSBench real
    try:
        dsbench = build_dsbench_real_subset('/data/lab/DSBench', 200,
            preferred_task_ids=['00000043','00000033','00000038','00000010','00000035',
                               '00000030','00000016','00000005','00000006','00000007'])
        for t in dsbench.tasks:
            tasks[t.task_id] = t
    except Exception as e:
        print(f'DSBench tasks skipped: {e}')
    return tasks


def recompute_repair_for_experiment(exp_dir, all_tasks, label):
    """Recompute oracle repair for all failed runs in an experiment."""
    cr = CausalReplay()
    results = {}

    for fname in sorted(os.listdir(os.path.join(BASE, exp_dir))):
        if not fname.endswith('_detail.json'):
            continue
        model = fname.replace('_detail.json', '')
        with open(os.path.join(BASE, exp_dir, fname)) as f:
            data = json.load(f)
        runs = data.get('runs', [])
        failed = [r for r in runs if not r.get('correct')]

        # Group by task_id (only need to test repair once per task, not per rep)
        tested = 0; repaired = 0; no_gt = 0
        task_tested = set()

        for run in failed:
            task_id = run.get('task_id', '')
            if task_id in task_tested:
                continue  # already tested this task
            task = all_tasks.get(task_id)
            if task is None:
                continue

            sig = _gt_signature(task)
            if sig is None:
                no_gt += 1
                task_tested.add(task_id)
                continue

            # Execute GT code on task data
            try:
                sb = CodeSandbox(tempfile.mkdtemp())
                task.prepare_data(sb.workdir)
                gt_code = GT_CODE_LIBRARY[sig]
                res = sb.execute(gt_code)
                success = res.success and task.check_answer(res.answer)
                tested += 1
                if success:
                    repaired += 1
            except Exception as e:
                tested += 1

            task_tested.add(task_id)

        rate = repaired / tested if tested else 0
        results[model] = {
            'n_failed_runs': len(failed),
            'n_unique_tasks_tested': tested,
            'n_repaired': repaired,
            'n_no_gt_code': no_gt,
            'oracle_repair_rate': round(rate, 4),
        }
        print(f'  {label:<8} {model:<28} tested={tested:<4} repaired={repaired:<4} no_gt={no_gt:<4} rate={rate:.4f}')

    return results


def main():
    print('Building all tasks...')
    all_tasks = build_all_tasks()
    print(f'Total unique tasks: {len(all_tasks)}')

    # Check GT coverage
    covered = sum(1 for t in all_tasks.values() if _gt_signature(t))
    print(f'GT code coverage: {covered}/{len(all_tasks)}')

    print('\n' + '='*90)
    print('ORACLE REPAIR RATE (expanded GT library, only on failed tasks)')
    print('='*90)

    all_results = {}
    for exp_dir, label in [('results_full', 'main'), ('results_uci', 'uci'), ('results_final', 'final')]:
        if not os.path.exists(os.path.join(BASE, exp_dir)):
            continue
        print(f'\n--- {label} ({exp_dir}) ---')
        all_results[label] = recompute_repair_for_experiment(exp_dir, all_tasks, label)

    # Grand summary
    print('\n' + '='*90)
    print('GRAND SUMMARY')
    print('='*90)
    total_tested = 0; total_repaired = 0
    for label, models in all_results.items():
        for model, r in models.items():
            total_tested += r['n_unique_tasks_tested']
            total_repaired += r['n_repaired']
    grand_rate = total_repaired / total_tested if total_tested else 0
    print(f'Total unique failed tasks tested: {total_tested}')
    print(f'Total repaired: {total_repaired}')
    print(f'Grand oracle repair rate: {grand_rate:.4f}')

    # Save
    output = {
        'oracle_repair_rate': grand_rate,
        'total_tested': total_tested,
        'total_repaired': total_repaired,
        'per_experiment': all_results,
        'note': 'Recomputed with expanded GT library (27 signatures). Only on failed tasks. No API calls.',
    }
    with open(os.path.join(BASE, 'oracle_repair_recomputed.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved oracle_repair_recomputed.json')


if __name__ == '__main__':
    main()
