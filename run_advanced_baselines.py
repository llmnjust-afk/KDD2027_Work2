#!/usr/bin/env python3
"""P0.5: Train 3 advanced baselines - FIXED to remove label leakage.

Key fix: Do NOT use is_silent, has_error, category-derived features 
(which directly encode the label). Use only trace-structural features.
"""
import csv, json, os, sys, re, numpy as np
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

runs = list(csv.DictReader(open('manifest_v3.csv')))
split = json.load(open('benchmark_split.json'))
train_tasks = set(split['train'])
dev_tasks = set(split['dev'])
test_tasks = set(split['test'])

valid = [r for r in runs if r['task_correct'] == 'False' 
         and r['is_infra_failure'] == 'False'
         and r['failure_stage'] in ('output_mismatch', 'runtime')
         and r['label_source'] == 'rule']
print(f'Valid failures: {len(valid)}')

# Load detail
detail_runs = {}
for src_dir in ['results_full', 'results_uci', 'results_final', 'results_supplementary']:
    if not os.path.exists(src_dir): continue
    for fname in os.listdir(src_dir):
        if not fname.endswith('_detail.json'): continue
        with open(os.path.join(src_dir, fname)) as f:
            data = json.load(f)
        for run in data.get('runs', []):
            v = run.get('variant', '')
            variant = 'verifier' if '+verifier' in v else 'baseline'
            key = (run.get('task_id',''), run.get('model',''), variant, run.get('rep',0))
            detail_runs[key] = run

def extract_features(r, detail):
    """Extract ONLY features that don't leak the label.
    NO: is_silent, has_error, category, stage (these ARE the label)
    YES: tokens, n_steps, suite, model, propagation_depth (structural only)
    """
    feats = {}
    
    # Structural features (no label leakage)
    feats['tokens'] = int(r.get('tokens', 0) or 0)
    feats['propagation_depth'] = int(r.get('propagation_depth', 0) or 0)
    feats['suite_synthetic'] = 1 if r.get('suite') == 'synthetic' else 0
    feats['suite_uci'] = 1 if r.get('suite') == 'uci' else 0
    feats['suite_dsbench'] = 1 if r.get('suite') == 'dsbench' else 0
    
    # Model one-hot
    model = r.get('model', '')
    feats['model_gpt4o'] = 1 if model == 'gpt-4o' else 0
    feats['model_gpt4o_mini'] = 1 if model == 'gpt-4o-mini' else 0
    feats['model_deepseek_v3'] = 1 if model == 'deepseek-v3' else 0
    feats['model_deepseek_r1'] = 1 if model == 'deepseek-r1' else 0
    feats['model_qwen3'] = 1 if model == 'qwen3-max-2026' else 0
    
    # From detail - structural only
    if detail:
        feats['elapsed_s'] = float(detail.get('elapsed_s', 0) or 0)
        feats['has_final_answer'] = 1 if detail.get('final_answer') and detail.get('final_answer') != 'None' else 0
        feats['final_answer_len'] = len(str(detail.get('final_answer', ''))) if detail.get('final_answer') else 0
        # Propagation depth from detail
        feats['detail_prop_depth'] = int(detail.get('propagation_depth', 0) or 0)
        # Recovered
        feats['recovered'] = 1 if detail.get('recovered') else 0
    else:
        feats['elapsed_s'] = 0
        feats['has_final_answer'] = 0
        feats['final_answer_len'] = 0
        feats['detail_prop_depth'] = 0
        feats['recovered'] = 0
    
    return feats

def extract_trace_text(r, detail):
    """Extract trace text for TF-IDF - using task_id and final_answer only (no stage/category)."""
    parts = [r.get('task_id', ''), r.get('suite', ''), r.get('model', '')]
    if detail:
        parts.append(str(detail.get('final_answer', '')))
    return ' '.join(parts)

# Build features
features_list = []
trace_texts = []
labels = []
splits = []

for r in valid:
    key = (r['task_id'], r['model'], r['variant'], int(r.get('rep', 0)))
    detail = detail_runs.get(key, {})
    feats = extract_features(r, detail)
    text = extract_trace_text(r, detail)
    features_list.append(feats)
    trace_texts.append(text)
    labels.append(r['failure_stage'])
    
    if r['task_id'] in train_tasks:
        splits.append('train')
    elif r['task_id'] in dev_tasks:
        splits.append('dev')
    else:
        splits.append('test')

splits = np.array(splits)
labels = np.array(labels)
train_mask = splits == 'train'
test_mask = splits == 'test'

print(f'Split: train={sum(train_mask)}, test={sum(test_mask)}')
print(f'Train: {Counter(labels[train_mask])}')
print(f'Test: {Counter(labels[test_mask])}')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===== Baseline 1: TF-IDF + Logistic Regression =====
print('\n=== Baseline 1: TF-IDF + Logistic Regression ===')
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X_train = vectorizer.fit_transform([trace_texts[i] for i in range(len(trace_texts)) if train_mask[i]])
X_test = vectorizer.transform([trace_texts[i] for i in range(len(trace_texts)) if test_mask[i]])
y_train, y_test = labels[train_mask], labels[test_mask]

clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf1.fit(X_train, y_train)
pred1 = clf1.predict(X_test)
acc1 = accuracy_score(y_test, pred1)
print(f'Accuracy: {acc1*100:.1f}%')
print(classification_report(y_test, pred1, zero_division=0))

# ===== Baseline 2: Supervised classifier (Random Forest on structural features) =====
print('\n=== Baseline 2: Trace Features + Random Forest ===')
feature_names = list(features_list[0].keys())
X_train_f = np.array([[features_list[i][f] for f in feature_names] for i in range(len(features_list)) if train_mask[i]])
X_test_f = np.array([[features_list[i][f] for f in feature_names] for i in range(len(features_list)) if test_mask[i]])

clf2 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
clf2.fit(X_train_f, y_train)
pred2 = clf2.predict(X_test_f)
acc2 = accuracy_score(y_test, pred2)
print(f'Accuracy: {acc2*100:.1f}%')
print(classification_report(y_test, pred2, zero_division=0))
importances = clf2.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print('Top 5 features:')
for i in sorted_idx[:5]:
    print(f'  {feature_names[i]}: {importances[i]:.4f}')

# ===== Baseline 3: Code-aware classifier =====
# Use structural features + token-based code features (no label leakage)
print('\n=== Baseline 3: Code-aware classifier (structural + token features + LR) ===')
# Add log-transformed tokens and interaction features
code_features = []
for i in range(len(features_list)):
    f = features_list[i].copy()
    f['log_tokens'] = np.log1p(f['tokens'])
    f['tokens_per_step'] = f['tokens'] / max(f['detail_prop_depth'] + 1, 1)
    f['has_large_answer'] = 1 if f['final_answer_len'] > 50 else 0
    code_features.append(f)

code_feature_names = list(code_features[0].keys())
X_train_c = np.array([[code_features[i][f] for f in code_feature_names] for i in range(len(code_features)) if train_mask[i]])
X_test_c = np.array([[code_features[i][f] for f in code_feature_names] for i in range(len(code_features)) if test_mask[i]])

clf3 = LogisticRegression(max_iter=1000, random_state=42)
clf3.fit(X_train_c, y_train)
pred3 = clf3.predict(X_test_c)
acc3 = accuracy_score(y_test, pred3)
print(f'Accuracy: {acc3*100:.1f}%')
print(classification_report(y_test, pred3, zero_division=0))

# ===== Summary =====
print('\n=== COMPLETE BASELINE SUMMARY ===')
print(f'{"Baseline":<40} {"Accuracy":>10}')
print(f'{"Majority":<40} {"61.6%":>10}')
print(f'{"Silent heuristic (uses is_silent)":<40} {"80.3%":>10}')
print(f'{"Token threshold":<40} {"55.2%":>10}')
print(f'{"GPT-4o-mini judge (47 traces)":<40} {"70.7%":>10}')
print(f'{"GPT-4o judge (47 traces)":<40} {"17.1%":>10}')
print(f'{"Random-step (Task D)":<40} {"82.1%":>10}')
print(f'{"Last-step (Task D)":<40} {"100.0%":>10}')
print(f'{"TF-IDF + LR (NEW)":<40} {f"{acc1*100:.1f}%":>10}')
print(f'{"Trace Features + RF (NEW)":<40} {f"{acc2*100:.1f}%":>10}')
print(f'{"Code-aware + LR (NEW)":<40} {f"{acc3*100:.1f}%":>10}')

results = {
    'tfidf_lr': {'accuracy': float(acc1) * 100, 'n_train': int(sum(train_mask)), 'n_test': int(sum(test_mask))},
    'trace_features_rf': {'accuracy': float(acc2) * 100, 'n_train': int(sum(train_mask)), 'n_test': int(sum(test_mask))},
    'code_aware_lr': {'accuracy': float(acc3) * 100, 'n_train': int(sum(train_mask)), 'n_test': int(sum(test_mask))},
}
json.dump(results, open('advanced_baselines_results.json', 'w'), indent=2)
print(f'\nSaved advanced_baselines_results.json')
print(f'\nNote: If accuracy is ~61.6% (majority), features have no signal.')
print(f'If accuracy is high, features still have signal (but not direct label leakage).')
