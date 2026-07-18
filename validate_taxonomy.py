#!/usr/bin/env python3
"""P2: LLM-as-judge taxonomy validation + Cohen's kappa.

Samples 100 failed traces from ALL experiments (2836 total failures), uses
GPT-4o-mini as an independent annotator (blind to the rule-based classifier's
label), and computes Cohen's kappa between rule-based and LLM-judged labels.

The LLM judge sees: task question, final answer, system stage label (hidden),
and must independently assign a stage from the v2 taxonomy.

Usage:
    python validate_taxonomy.py --n 100
"""
import json, os, sys, random, re
from collections import Counter, defaultdict

sys.path.insert(0, '.')

BASE = '/data/lab/KDD2027_Work2'


def collect_all_failures():
    """Collect all failed runs from all experiments."""
    all_failures = []
    for dirname, label in [('results_full','main'), ('results_uci','uci'), ('results_final','final')]:
        path = os.path.join(BASE, dirname)
        if not os.path.exists(path): continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith('_detail.json'): continue
            model = fname.replace('_detail.json','')
            with open(os.path.join(path, fname)) as f:
                data = json.load(f)
            for run in data.get('runs', []):
                if not run.get('correct'):
                    all_failures.append({
                        'source': label, 'model': model,
                        'task_id': run.get('task_id',''),
                        'stage': run.get('stage',''),
                        'category': run.get('category',''),
                        'is_silent': run.get('is_silent', False),
                        'final_answer': run.get('final_answer',''),
                        'tokens': run.get('tokens',0),
                    })
    return all_failures


def sample_stratified(failures, n=100, seed=42):
    """Stratified sample across models and stages."""
    rng = random.Random(seed)
    by_model = defaultdict(list)
    for f in failures:
        by_model[f['model']].append(f)
    sampled = []
    total = len(failures)
    for model, runs in by_model.items():
        n_model = max(1, round(n * len(runs) / total))
        n_model = min(n_model, len(runs))
        sampled.extend(rng.sample(runs, n_model))
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)
    elif len(sampled) < n:
        remaining = [f for f in failures if f not in sampled]
        sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))
    return sampled


def llm_judge_prompt(failure):
    """Build a prompt for the LLM judge. The judge does NOT see the system's label."""
    return f"""You are annotating a failed LLM agent run on a data-science task.

Task ID: {failure['task_id']}
Model: {failure['model']}
Final answer produced: {failure.get('final_answer', 'None')}

Based on this information, assign the failure to exactly ONE of these 5 stages:

1. analytical_plan - The agent's stated approach was wrong (e.g., planned to compute mean when task needed sum)
2. code_generation - The approach was correct but the generated code didn't implement it (e.g., used .count() instead of .sum())
3. runtime - The code crashed with an exception (loud failure)
4. output_mismatch - The code ran but the agent extracted/reported the wrong answer from the output
5. answer_error - The code ran and the output was internally consistent, but the answer was wrong because the approach was flawed

Rules:
- If the final answer is None (no answer produced), it's likely "runtime" (code crashed)
- If the final answer is a value but wrong, it could be any of the silent stages (1,2,4,5)
- Assign exactly ONE stage
- Respond with ONLY the stage name, nothing else

Stage:"""


def compute_kappa(labels1, labels2):
    """Cohen's kappa between two label lists."""
    n = len(labels1)
    if n == 0 or len(labels1) != len(labels2):
        return 0, 0, 0, "error"
    categories = sorted(set(labels1 + labels2))
    agree = sum(1 for a, b in zip(labels1, labels2) if a == b)
    po = agree / n
    c1, c2 = Counter(labels1), Counter(labels2)
    pe = sum((c1[c]/n) * (c2[c]/n) for c in categories)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    interp = ("almost perfect" if kappa > 0.81 else "substantial" if kappa > 0.61
              else "moderate" if kappa > 0.41 else "fair" if kappa > 0.21 else "poor")
    return po, pe, kappa, interp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--output', default='taxonomy_validation.json')
    args = ap.parse_args()

    failures = collect_all_failures()
    print(f'Total failures collected: {len(failures)}')

    sampled = sample_stratified(failures, args.n)
    print(f'Sampled {len(sampled)} failures (stratified by model)')

    # Model distribution
    model_dist = Counter(f['model'] for f in sampled)
    stage_dist = Counter(f['stage'] for f in sampled)
    print(f'Model dist: {dict(model_dist)}')
    print(f'Stage dist (rule-based): {dict(stage_dist)}')

    # LLM-as-judge annotation
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get('OPENAI_API_KEY'),
        base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.chatanywhere.tech/v1'),
    )

    # Map old stage names to new v2 stages
    old_to_new = {
        'planning': 'analytical_plan',
        'tool_use': 'code_generation',
        'execution': 'runtime',
        'interpretation': 'output_mismatch',
    }

    rule_labels = []
    llm_labels = []
    results = []

    for i, f in enumerate(sampled):
        prompt = llm_judge_prompt(f)
        try:
            resp = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role':'user','content':prompt}],
                max_tokens=20, temperature=0,
            )
            llm_stage = (resp.choices[0].message.content or '').strip().lower()
            # normalize
            llm_stage = llm_stage.replace(' ', '_').replace('-', '_')
            if llm_stage not in ['analytical_plan','code_generation','runtime','output_mismatch','answer_error']:
                llm_stage = 'answer_error'  # default
        except Exception as e:
            llm_stage = 'error'

        rule_stage = old_to_new.get(f['stage'], 'answer_error')
        rule_labels.append(rule_stage)
        llm_labels.append(llm_stage)

        results.append({
            'task_id': f['task_id'],
            'model': f['model'],
            'rule_stage': rule_stage,
            'llm_stage': llm_stage,
            'is_silent': f['is_silent'],
            'final_answer': str(f.get('final_answer','') or '')[:100],
        })

        if (i+1) % 20 == 0:
            print(f'  Annotated {i+1}/{len(sampled)}...')

    # Compute kappa
    po, pe, kappa, interp = compute_kappa(rule_labels, llm_labels)
    print(f'\n{"="*60}')
    print(f'TAXONOMY VALIDATION RESULTS')
    print(f'{"="*60}')
    print(f'N annotated: {len(rule_labels)}')
    print(f'Observed agreement: {po:.4f}')
    print(f'Expected agreement: {pe:.4f}')
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f'Interpretation: {interp}')

    # Confusion matrix
    print(f'\nConfusion matrix (rows=rule, cols=LLM):')
    cats = sorted(set(rule_labels + llm_labels))
    print(f'  {"":<20} {"  ".join(c[:8] for c in cats)}')
    for r in cats:
        row = [sum(1 for a,b in zip(rule_labels,llm_labels) if a==r and b==c) for c in cats]
        print(f'  {r:<20} {"  ".join(f"{v:>8}" for v in row)}')

    # Per-category agreement
    print(f'\nPer-category agreement:')
    for cat in cats:
        n_cat = sum(1 for l in rule_labels if l == cat)
        if n_cat > 0:
            agree_cat = sum(1 for a,b in zip(rule_labels,llm_labels) if a==cat and b==cat)
            print(f'  {cat:<20}: {agree_cat}/{n_cat} = {agree_cat/n_cat:.2%}')

    # Save
    output = {
        'n': len(rule_labels),
        'observed_agreement': po,
        'expected_agreement': pe,
        'cohens_kappa': kappa,
        'interpretation': interp,
        'rule_labels': rule_labels,
        'llm_labels': llm_labels,
        'per_trace': results,
    }
    with open(os.path.join(BASE, args.output), 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
