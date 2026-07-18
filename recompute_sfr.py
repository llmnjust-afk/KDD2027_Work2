#!/usr/bin/env python3
"""Recompute SFR with FIXED classifier (final_answer=None → runtime, not silent).

The old classifier defaulted final_answer=None traces to INTERPRETATION+silent=True,
inflating SFR. This script reclassifies all failed runs using the fixed logic:
  - final_answer=None or empty → RUNTIME (loud)
  - final_answer=wrong value → INTERPRETATION (silent)

No API calls needed — just reinterprets existing run records.
"""
import json, os, sys
from collections import Counter

BASE = '/data/lab/KDD2027_Work2'

def reclassify(run):
    """Reclassify a single run with the fixed logic."""
    if run.get('correct'):
        return 'correct', False
    final_answer = run.get('final_answer')
    old_stage = run.get('stage', '')
    old_silent = run.get('is_silent', False)

    # Fixed logic: if no answer produced, it's runtime (loud)
    if final_answer is None or (isinstance(final_answer, str) and not final_answer.strip()):
        return 'runtime', False  # loud
    # If answer produced but wrong, check if code actually ran
    if old_stage == 'execution':
        return 'runtime', False  # loud (code crashed)
    # Answer produced and code ran → genuinely silent
    return 'interpretation', True  # silent


def main():
    print('='*100)
    print('CORRECTED SFR (fixed classifier: final_answer=None → runtime, not silent)')
    print('='*100)
    print(f'{"Experiment":<10} {"Model":<28} {"N":>5} {"Fail":>5} {"OldSil":>7} {"NewSil":>7} {"OldSFR":>7} {"NewSFR":>7} {"Delta":>7}')
    print('-'*100)

    grand = {'n':0, 'fail':0, 'old_silent':0, 'new_silent':0}

    for dirname, label in [('results_full','main'), ('results_uci','uci'), ('results_final','final')]:
        path = os.path.join(BASE, dirname)
        if not os.path.exists(path): continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith('_detail.json'): continue
            model = fname.replace('_detail.json','')
            with open(os.path.join(path, fname)) as f:
                data = json.load(f)
            runs = data.get('runs', [])
            n = len(runs)
            ok = sum(1 for r in runs if r.get('correct'))
            fail = n - ok
            old_silent = sum(1 for r in runs if r.get('is_silent') and not r.get('correct'))

            # Reclassify
            new_silent = 0
            new_stages = Counter()
            for r in runs:
                if r.get('correct'): continue
                stage, is_silent = reclassify(r)
                new_stages[stage] += 1
                if is_silent:
                    new_silent += 1

            old_sfr = old_silent / fail if fail else 0
            new_sfr = new_silent / fail if fail else 0
            delta = new_sfr - old_sfr

            print(f'{label:<10} {model:<28} {n:>5} {fail:>5} {old_silent:>7} {new_silent:>7} {old_sfr:>7.4f} {new_sfr:>7.4f} {delta:>+7.4f}')

            grand['n'] += n; grand['fail'] += fail
            grand['old_silent'] += old_silent; grand['new_silent'] += new_silent

    print('-'*100)
    g_old_sfr = grand['old_silent']/grand['fail'] if grand['fail'] else 0
    g_new_sfr = grand['new_silent']/grand['fail'] if grand['fail'] else 0
    print(f'{"GRAND":<10} {"":28} {grand["n"]:>5} {grand["fail"]:>5} {grand["old_silent"]:>7} {grand["new_silent"]:>7} {g_old_sfr:>7.4f} {g_new_sfr:>7.4f} {g_new_sfr-g_old_sfr:>+7.4f}')

    print(f'\n{"="*60}')
    print(f'IMPACT OF CLASSIFIER FIX')
    print(f'{"="*60}')
    print(f'Old SFR (buggy): {g_old_sfr:.4f} ({grand["old_silent"]}/{grand["fail"]})')
    print(f'New SFR (fixed): {g_new_sfr:.4f} ({grand["new_silent"]}/{grand["fail"]})')
    print(f'Change: {g_new_sfr - g_old_sfr:+.4f}')
    print(f'\nThe old classifier inflated SFR by treating final_answer=None')
    print(f'(code crashes) as silent failures. The fixed classifier correctly')
    print(f'classifies these as runtime (loud) failures.')

    # Save corrected SFR
    output = {
        'old_sfr': g_old_sfr, 'new_sfr': g_new_sfr,
        'old_silent': grand['old_silent'], 'new_silent': grand['new_silent'],
        'total_failures': grand['fail'], 'total_runs': grand['n'],
    }
    with open(os.path.join(BASE, 'corrected_sfr.json'), 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
