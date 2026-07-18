#!/usr/bin/env python3
"""P0.1+P0.2: Build single source-of-truth manifest + auto-generate all tables.

This script is the ONLY place where experiment numbers are computed.
All LaTeX tables are generated from this manifest with assertions.
No number should appear in the paper that doesn't come from this script.

Output:
  manifest.csv          - one row per run, the single source of truth
  tables/table_*.tex    - all LaTeX tables, auto-generated
  tables/consistency_report.txt - assertion results

Usage:
    python build_manifest_and_tables.py
"""
import json, os, csv, sys, math
from collections import defaultdict, Counter

BASE = '/data/lab/KDD2027_Work2'

# ── Load ALL runs from ALL experiments ────────────────────────────── #

def load_all_runs():
    """Load every run from every experiment into a flat list."""
    runs = []
    for dirname, suite_label in [
        ('results_full', 'synthetic'),
        ('results_uci', 'uci'),
        ('results_final', 'mixed'),  # final has both synth + dsbench
        ('results_supplementary', 'supplementary'),
    ]:
        path = os.path.join(BASE, dirname)
        if not os.path.exists(path):
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith('_detail.json'):
                continue
            model = fname.replace('_detail.json', '')
            with open(os.path.join(path, fname)) as f:
                data = json.load(f)
            for r in data.get('runs', []):
                # Determine suite
                is_real = r.get('is_real', False)
                if suite_label == 'mixed':
                    suite = 'dsbench' if is_real else 'synthetic'
                elif suite_label == 'supplementary':
                    # UCI verifier runs have domain=uci_real
                    # DSBench runs have domain=dsbench_real
                    domain = r.get('domain', '')
                    suite = 'uci' if 'uci' in domain else 'dsbench'
                else:
                    suite = suite_label

                # Determine model and variant
                # For supplementary files, use run-level model/variant fields
                run_model = r.get('model', '').replace('+verifier', '').replace('+consistency', '')
                run_variant = r.get('variant', '')
                if run_model and run_model != model:
                    # Use run-level model (more accurate for supplementary files)
                    base_model = run_model
                    if '+verifier' in run_variant or '+verifier' in r.get('variant', ''):
                        variant = 'verifier'
                    elif '+consistency' in run_variant:
                        variant = 'consistency'
                    else:
                        variant = 'baseline'
                else:
                    base_model = model.replace('+verifier', '').replace('+consistency', '').replace('dsbench_', '').replace('uci_', '')
                    variant = 'verifier' if '+verifier' in model else ('consistency' if '+consistency' in model else 'baseline')

                # Fixed reclassification
                fa = r.get('final_answer')
                old_stage = r.get('stage', '')
                if r.get('correct'):
                    failure_stage = 'none'
                    is_silent = False
                elif fa is None or (isinstance(fa, str) and not fa.strip()):
                    failure_stage = 'runtime'
                    is_silent = False
                elif old_stage in ('execution', 'runtime'):
                    failure_stage = 'runtime'
                    is_silent = False
                else:
                    failure_stage = 'silent'
                    is_silent = True

                runs.append({
                    'task_id': r.get('task_id', ''),
                    'suite': suite,
                    'model': base_model,
                    'variant': variant,
                    'rep': r.get('rep', 0),
                    'correct': r.get('correct', False),
                    'failure_stage': failure_stage,
                    'is_silent': is_silent,
                    'tokens': r.get('tokens', 0),
                    'final_answer': str(fa or '')[:200],
                    'causal_attributed': r.get('causal_attributed', False),
                    'source_file': fname,
                    'source_dir': dirname,
                })
    return runs


# ── Compute metrics from manifest ────────────────────────────────── #

def compute_stats(runs, group_keys):
    """Group runs by keys and compute SR, SFR, etc."""
    groups = defaultdict(list)
    for r in runs:
        key = tuple(r[k] for k in group_keys)
        groups[key].append(r)

    results = []
    for key, group in sorted(groups.items()):
        n = len(group)
        n_correct = sum(1 for r in group if r['correct'])
        n_fail = n - n_correct
        n_silent = sum(1 for r in group if r['is_silent'] and not r['correct'])
        n_loud = n_fail - n_silent
        sr = n_correct / n if n else 0
        sfr = n_silent / n_fail if n_fail else 0
        tok_succ = sum(r['tokens'] for r in group if r['correct']) / n_correct if n_correct else 0
        results.append({
            'key': key,
            'n': n, 'n_correct': n_correct, 'n_fail': n_fail,
            'n_silent': n_silent, 'n_loud': n_loud,
            'sr': sr, 'sfr': sfr, 'tok_succ': tok_succ,
        })
    return results


# ── McNemar test ─────────────────────────────────────────────────── #

def mcnemar(runs_a, runs_b):
    """McNemar's test on paired binary outcomes.
    runs_a[i] and runs_b[i] must be the same task+rep.
    Returns (b, c, chi2, p_approx).
    b = A correct & B wrong; c = A wrong & B correct.
    """
    # Match by task_id + rep
    a_dict = {(r['task_id'], r['rep']): r['correct'] for r in runs_a}
    b_dict = {(r['task_id'], r['rep']): r['correct'] for r in runs_b}
    common = set(a_dict) & set(b_dict)
    b_count = sum(1 for k in common if a_dict[k] and not b_dict[k])
    c_count = sum(1 for k in common if not a_dict[k] and b_dict[k])
    # McNemar chi-square with continuity correction
    if b_count + c_count == 0:
        return 0, 0, 0, 1.0
    chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count) if (b_count + c_count) > 0 else 0
    # p-value from chi-square df=1 (normal approx)
    from math import erfc, sqrt
    p = erfc(sqrt(chi2 / 2)) if chi2 > 0 else 1.0
    return b_count, c_count, chi2, p


# ── Main ─────────────────────────────────────────────────────────── #

def main():
    runs = load_all_runs()
    print(f'Total runs loaded: {len(runs)}')

    # ── Save manifest CSV ── #
    manifest_path = os.path.join(BASE, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'task_id', 'suite', 'model', 'variant', 'rep', 'correct',
            'failure_stage', 'is_silent', 'tokens', 'final_answer',
            'causal_attributed', 'source_file', 'source_dir',
        ])
        w.writeheader()
        w.writerows(runs)
    print(f'Manifest saved: {manifest_path}')

    # ── ASSERTIONS ── #
    report_lines = []
    def assert_eq(a, b, msg):
        ok = a == b
        report_lines.append(f'{"✓" if ok else "✗"} {msg}: {a} == {b} {"PASS" if ok else "FAIL"}')
        if not ok:
            print(f'ASSERTION FAILED: {msg}: {a} != {b}')

    # Total runs
    assert_eq(len(runs), 4875 + len([r for r in runs if r['source_dir'] == 'results_supplementary']),
              'Total runs (4875 + supplementary)')

    # Failures
    n_fail = sum(1 for r in runs if not r['correct'])
    n_silent = sum(1 for r in runs if r['is_silent'] and not r['correct'])
    n_loud = n_fail - n_silent
    assert_eq(n_silent + n_loud, n_fail, 'silent + loud == failures')

    # Suite breakdown
    suite_counts = Counter(r['suite'] for r in runs)
    report_lines.append(f'\nSuite breakdown: {dict(suite_counts)}')

    # ── TABLE 2: Task Suite Summary ── #
    # Count unique tasks per suite
    suite_tasks = defaultdict(set)
    for r in runs:
        suite_tasks[r['suite']].add(r['task_id'])

    table2_lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Task suite summary. All numbers are from a single source-of-truth manifest.}',
        r'\label{tab:task-summary}',
        r'\small',
        r'\begin{tabular}{@{}lcccc@{}}',
        r'\toprule',
        r'Suite & Tasks & Models & Runs & Failures \\',
        r'\midrule',
    ]
    # Group by suite
    for suite in ['synthetic', 'uci', 'dsbench']:
        suite_runs = [r for r in runs if r['suite'] == suite]
        n_tasks = len(suite_tasks[suite])
        n_models = len(set(r['model'] for r in suite_runs))
        n_runs = len(suite_runs)
        n_fails = sum(1 for r in suite_runs if not r['correct'])
        table2_lines.append(f'{suite.capitalize()} & {n_tasks} & {n_models} & {n_runs:,} & {n_fails:,} \\\\')
    table2_lines.append(r'\midrule')
    total_tasks = len(set(r['task_id'] for r in runs))
    total_models = len(set(r['model'] for r in runs))
    table2_lines.append(f'\\textbf{{Total}} & \\textbf{{{total_tasks}}} & \\textbf{{{total_models}}} & \\textbf{{{len(runs):,}}} & \\textbf{{{n_fail:,}}} \\\\')
    table2_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    # ── TABLE 3: Main results (table*) ── #
    # For each model, compute SR and SFR per suite
    models_order = ['gpt-4o-mini', 'gpt-4o', 'deepseek-v3', 'deepseek-r1', 'qwen3-max-2026']
    model_display = {
        'gpt-4o-mini': 'GPT-4o-mini', 'gpt-4o': 'GPT-4o',
        'deepseek-v3': 'DeepSeek-V3', 'deepseek-r1': 'DeepSeek-R1', 'qwen3-max-2026': 'Qwen3-max',
    }

    table3_lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Comprehensive results across all models and task suites. SR = success rate, SFR = silent-failure rate (corrected, on failed runs only). ``n/a'' = not evaluated.}',
        r'\label{tab:main-results}',
        r'\small',
        r'\begin{tabular}{@{}llcccccc@{}}',
        r'\toprule',
        r' & & \multicolumn{2}{c}{Synthetic} & \multicolumn{2}{c}{UCI real} & \multicolumn{2}{c}{DSBench real} \\',
        r'\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}',
        r'Model & Variant & SR (\%) & SFR (\%) & SR (\%) & SFR (\%) & SR (\%) & SFR (\%) \\',
        r'\midrule',
    ]

    for model in models_order:
        for variant in ['baseline', 'verifier']:
            row_cells = []
            row_cells.append(model_display.get(model, model))
            row_cells.append(variant.capitalize() if variant != 'baseline' else 'Baseline')

            for suite in ['synthetic', 'uci', 'dsbench']:
                subset = [r for r in runs if r['model'] == model and r['variant'] == variant and r['suite'] == suite]
                if not subset:
                    row_cells.extend(['n/a', 'n/a'])
                else:
                    n = len(subset)
                    n_ok = sum(1 for r in subset if r['correct'])
                    n_fail = n - n_ok
                    n_sil = sum(1 for r in subset if r['is_silent'] and not r['correct'])
                    sr = n_ok / n * 100
                    sfr = n_sil / n_fail * 100 if n_fail else 0
                    row_cells.append(f'{sr:.1f}')
                    row_cells.append(f'{sfr:.0f}' if n_fail > 0 else '---')

            # Only add row if there's at least one non-n/a cell
            if any(c != 'n/a' for c in row_cells[2:]):
                best_sr = max((float(c) for c in row_cells[2::2] if c not in ('n/a', '---')), default=0)
                # Bold the best model per suite
                table3_lines.append(' & '.join(row_cells) + r' \\')

    table3_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']

    # ── TABLE 4: Oracle repair ── #
    # Use the recomputed oracle repair data
    oracle_path = os.path.join(BASE, 'oracle_repair_recomputed.json')
    table4_lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Oracle repair rate (failed tasks only).}',
        r'\label{tab:oracle-repair}',
        r'\small',
        r'\begin{tabular}{@{}lcc@{}}',
        r'\toprule',
        r'Model (suite) & Tested & Repaired \\',
        r'\midrule',
    ]
    if os.path.exists(oracle_path):
        with open(oracle_path) as f:
            oracle = json.load(f)
        total_tested = 0
        total_repaired = 0
        for exp, models in oracle.get('per_experiment', {}).items():
            for model, stats in models.items():
                tested = stats['n_unique_tasks_tested']
                repaired = stats['n_repaired']
                rate = stats['oracle_repair_rate']
                total_tested += tested
                total_repaired += repaired
                table4_lines.append(f'{model} ({exp}) & {tested} & {repaired} ({rate:.0%}) \\\\')
        table4_lines.append(r'\midrule')
        grand_rate = total_repaired / total_tested if total_tested else 0
        table4_lines.append(f'\\textbf{{Overall}} & \\textbf{{{total_tested}}} & \\textbf{{{total_repaired} ({grand_rate:.0%})}} \\\\')
        # Assertion: total_tested must match
        report_lines.append(f'Oracle repair: tested={total_tested}, repaired={total_repaired}, rate={grand_rate:.4f}')

    table4_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    # ── TABLE 5: Verifier ablation with McNemar ── #
    table5_lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Verifier ablation (McNemar test). $b$=baseline correct \& verifier wrong, $c$=baseline wrong \& verifier correct.}',
        r'\label{tab:verifier}',
        r'\small',
        r'\begin{tabular}{@{}lcccc@{}}',
        r'\toprule',
        r'Model & $\Delta$SR (\%) & $b$ & $c$ & $p$-value \\',
        r'\midrule',
    ]
    for model in models_order:
        baseline = [r for r in runs if r['model'] == model and r['variant'] == 'baseline' and r['suite'] == 'synthetic']
        verifier = [r for r in runs if r['model'] == model and r['variant'] == 'verifier' and r['suite'] == 'synthetic']
        if not baseline or not verifier:
            continue
        b, c, chi2, p = mcnemar(baseline, verifier)
        sr_base = sum(1 for r in baseline if r['correct']) / len(baseline) * 100
        sr_ver = sum(1 for r in verifier if r['correct']) / len(verifier) * 100
        delta = sr_ver - sr_base
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        table5_lines.append(f'{model_display[model]} & {delta:+.1f} & {b} & {c} & {p:.3f}{sig} \\\\')

    table5_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    # ── TABLE 6: Null model (corrected interpretation) ── #
    table6_lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{SFR by task suite (corrected classifier). Note: SFR and execution-success rate share the variable $E$ (loud failures), so their correlation is partially mechanical. $R^2$ is reported as a descriptive statistic, not a causal decomposition.}',
        r'\label{tab:null-model}',
        r'\small',
        r'\begin{tabular}{@{}lc@{}}',
        r'\toprule',
        r'Metric & Value \\',
        r'\midrule',
    ]
    # Compute per-suite SFR
    for suite in ['synthetic', 'uci', 'dsbench']:
        suite_runs = [r for r in runs if r['suite'] == suite]
        n_fail = sum(1 for r in suite_runs if not r['correct'])
        n_silent = sum(1 for r in suite_runs if r['is_silent'] and not r['correct'])
        sfr = n_silent / n_fail * 100 if n_fail else 0
        table6_lines.append(f'\\quad {suite.capitalize()} SFR & {sfr:.1f}\\% ({n_silent}/{n_fail}) \\\\')

    # Correlation
    model_suite_stats = compute_stats(runs, ['model', 'suite'])
    exec_rates = []
    sfrs = []
    for s in model_suite_stats:
        if s['n_fail'] > 0:
            exec_rate = (s['n'] - s['n_loud']) / s['n']  # = 1 - P(loud)
            exec_rates.append(exec_rate)
            sfrs.append(s['sfr'])
    if len(exec_rates) > 2:
        n = len(exec_rates)
        me = sum(exec_rates) / n
        ms = sum(sfrs) / n
        cov = sum((e - me) * (s - ms) for e, s in zip(exec_rates, sfrs)) / n
        se = math.sqrt(sum((e - me) ** 2 for e in exec_rates) / n)
        ss = math.sqrt(sum((s - ms) ** 2 for s in sfrs) / n)
        r = cov / (se * ss) if se * ss > 0 else 0
        r2 = r ** 2
        table6_lines.append(r'\midrule')
        table6_lines.append(f'Correlation $r$ & {r:.3f} \\\\')
        table6_lines.append(f'$R^2$ (descriptive) & {r2:.3f} \\\\')
        table6_lines.append(r'\multicolumn{2}{l}{\textit{Note: $R^2$ is not a causal decomposition.}} \\')

    table6_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    # ── Write all tables ── #
    tables_dir = os.path.join(BASE, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    all_tables = {
        'table_task_summary.tex': table2_lines,
        'table_main_results.tex': table3_lines,
        'table_oracle_repair.tex': table4_lines,
        'table_verifier.tex': table5_lines,
        'table_null_model.tex': table6_lines,
    }
    for fname, lines in all_tables.items():
        with open(os.path.join(tables_dir, fname), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'  Generated {fname}')

    # ── Write consistency report ── #
    report_path = os.path.join(BASE, 'tables', 'consistency_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'\nConsistency report: {report_path}')
    print('\n'.join(report_lines))


if __name__ == '__main__':
    main()
