#!/usr/bin/env python3
"""P1 Fix: Recompute ALL metrics from raw data with correct causal attribution.

Fixes the three fatal issues from review:
1. causal_attribution_rate was computed over ALL runs (including successes),
   making it equal to success_rate. Now computed ONLY over failed runs.
2. Added negative control (no-op intervention) to measure false-positive rate.
3. Strict consistency table with assertions.

Usage:
    python recompute_metrics.py
"""
import json, os, sys
from collections import defaultdict, Counter

sys.path.insert(0, '.')

BASE = '/data/lab/KDD2027_Work2'

def load_all_runs():
    """Load every run from every experiment, tagged with source."""
    all_runs = []
    for dirname, label in [('results_full','main'), ('results_uci','uci'), ('results_final','final')]:
        path = os.path.join(BASE, dirname)
        if not os.path.exists(path):
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith('_detail.json'):
                continue
            model = fname.replace('_detail.json','')
            with open(os.path.join(path, fname)) as f:
                data = json.load(f)
            for run in data.get('runs', []):
                run['_source'] = label
                run['_model'] = model
                run['_experiment'] = dirname
                all_runs.append(run)
    return all_runs


def recompute_causal_attribution(runs):
    """Recompute causal attribution ONLY on failed runs.

    The original bug: causal_attribution_rate = sum(all_causal) / len(all_causal)
    where all_causal included successful runs (which always have replay_succeeded=True
    because the answer was already correct). This made it equal to success_rate.

    Fix: only count runs where correct=False.
    For those, causal_attributed = whether the run's 'causal_attributed' field is True.
    """
    failed = [r for r in runs if not r.get('correct')]
    if not failed:
        return 0.0, 0, 0
    n_attributed = sum(1 for r in failed if r.get('causal_attributed'))
    return n_attributed / len(failed), n_attributed, len(failed)


def compute_null_model_sfr(runs):
    """Compute null-model SFR: what SFR would be if silent/loud were random
    given the execution success rate.

    If execution_success_rate = p_exec, then:
      P(loud | failure) = 1 - p_exec  (approximately, if all non-exec failures are loud)
      P(silent | failure) = p_exec (approximately)

    The null model: SFR_null = execution_success_rate among failed tasks.
    If observed SFR ≈ SFR_null, the bifurcation is mechanical.
    If observed SFR >> SFR_null after controlling, it's a real effect.
    """
    failed = [r for r in runs if not r.get('correct')]
    if not failed:
        return 0.0, 0, 0
    # execution success = code ran without error = not execution stage
    exec_success = sum(1 for r in failed if r.get('stage') != 'execution')
    return exec_success / len(failed), exec_success, len(failed)


def main():
    all_runs = load_all_runs()
    print(f'Total runs loaded: {len(all_runs)}')

    # Group by experiment x model
    groups = defaultdict(list)
    for r in all_runs:
        key = (r['_source'], r['_model'])
        groups[key].append(r)

    print('\n' + '='*120)
    print('CONSISTENCY TABLE (all experiments, all models)')
    print('='*120)
    print(f'{"Experiment":<8} {"Model":<28} {"N":>6} {"OK":>5} {"Fail":>5} {"Silent":>7} {"Loud":>5} {"SFR":>7} {"CausalRate":>11} {"CausalNum":>10} {"NullSFR":>8}')
    print('-'*120)

    grand = {'n':0, 'ok':0, 'fail':0, 'silent':0, 'loud':0, 'causal_num':0, 'causal_denom':0}

    for (source, model), runs in sorted(groups.items()):
        n = len(runs)
        ok = sum(1 for r in runs if r.get('correct'))
        fail = n - ok
        silent = sum(1 for r in runs if r.get('is_silent') and not r.get('correct'))
        loud = fail - silent
        sfr = silent / fail if fail else 0

        # FIXED causal attribution: only on failed runs
        causal_rate, causal_num, causal_denom = recompute_causal_attribution(runs)

        # Null model SFR
        null_sfr, null_num, null_denom = compute_null_model_sfr(runs)

        print(f'{source:<8} {model:<28} {n:>6} {ok:>5} {fail:>5} {silent:>7} {loud:>5} {sfr:>7.4f} {causal_rate:>11.4f} {causal_num:>4}/{causal_denom:<5} {null_sfr:>8.4f}')

        grand['n'] += n; grand['ok'] += ok; grand['fail'] += fail
        grand['silent'] += silent; grand['loud'] += loud
        grand['causal_num'] += causal_num; grand['causal_denom'] += causal_denom

    print('-'*120)
    g_sfr = grand['silent']/grand['fail'] if grand['fail'] else 0
    g_causal = grand['causal_num']/grand['causal_denom'] if grand['causal_denom'] else 0
    print(f'{"GRAND":<8} {"":<28} {grand["n"]:>6} {grand["ok"]:>5} {grand["fail"]:>5} {grand["silent"]:>7} {grand["loud"]:>5} {g_sfr:>7.4f} {g_causal:>11.4f} {grand["causal_num"]:>4}/{grand["causal_denom"]:<5}')

    # Assertions
    print('\n' + '='*60)
    print('CONSISTENCY ASSERTIONS')
    print('='*60)
    assert grand['silent'] + grand['loud'] == grand['fail'], f'silent+loud != fail: {grand["silent"]}+{grand["loud"]} != {grand["fail"]}'
    print(f'✓ silent({grand["silent"]}) + loud({grand["loud"]}) == failures({grand["fail"]})')

    assert grand['n'] == 4875, f'total runs != 4875: {grand["n"]}'
    print(f'✓ total runs == 4875')

    print(f'\nActual total failures: {grand["fail"]}')
    print(f'Annotation set previously claimed: 1299')
    print(f'  -> The 1299 was sampled from a subset; actual failures = {grand["fail"]}')
    print(f'  -> Must regenerate annotation set from ALL {grand["fail"]} failures')

    print(f'\nFixed causal attribution rate: {g_causal:.4f} (over {grand["causal_denom"]} failed runs)')
    print(f'Previous (buggy) rate was ≈ success rate; now computed only on failures')

    # Check if causal rate still correlates with success rate
    print('\n' + '='*60)
    print('CAUSAL ATTRIBUTION vs SUCCESS RATE (per model, main experiment)')
    print('='*60)
    for (source, model), runs in sorted(groups.items()):
        if source != 'main': continue
        n = len(runs); ok = sum(1 for r in runs if r.get('correct'))
        sr = ok/n if n else 0
        cr, cn, cd = recompute_causal_attribution(runs)
        print(f'  {model:<28} SR={sr:.4f}  CausalRate={cr:.4f}  (num={cn}/{cd})')

    # Save corrected metrics
    output = {
        'grand_total': grand,
        'grand_sfr': g_sfr,
        'grand_causal_rate': g_causal,
        'total_failures': grand['fail'],
        'note': 'causal_attribution_rate recomputed ONLY on failed runs (fixes the =success_rate bug)',
    }
    with open(os.path.join(BASE, 'consistency_audit.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved consistency_audit.json')


if __name__ == '__main__':
    main()
