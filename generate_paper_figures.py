#!/usr/bin/env python3
"""Generate all publication figures for the paper using matplotlib.

Produces:
  1. bifurcation.pdf  - SFR vs SR scatter showing bifurcation
  2. model_comparison.pdf - 5-model success rate + SFR bar chart
  3. stage_distribution.pdf - failure stage stacked bar
  4. verifier_ablation.pdf - verifier effect per model
  5. token_economics.pdf - cost-accuracy frontier
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict, Counter

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'figure.dpi': 150,
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.unicode_minus': False,
})

BASE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(BASE, 'figures')
os.makedirs(FIGS, exist_ok=True)

# ---- load data ----
def load_results(dirname):
    out = {}
    d = os.path.join(BASE, dirname)
    if not os.path.exists(d): return out
    for f in sorted(os.listdir(d)):
        if f.endswith('_detail.json'):
            with open(os.path.join(d,f)) as fh:
                out[f.replace('_detail.json','')] = json.load(fh)
    return out

main = load_results('results_full')
uci = load_results('results_uci')
final = load_results('results_final')

# ---- Fig 1: Bifurcation ----
fig, ax = plt.subplots(figsize=(7,5))
markers = {'synthetic':'s','uci_real':'^','dsbench_real':'D'}
colors = {'synthetic':'#2196F3','uci_real':'#4CAF50','dsbench_real':'#FF5722'}
# synthetic (main experiment, baselines only)
for m in ['gpt-4o-mini','gpt-4o','deepseek-chat','qwen3-max']:
    if m in main:
        d = main[m]
        fm = d.get('failure_metrics',{})
        runs = d.get('runs',[])
        sr = sum(1 for r in runs if r['correct'])/len(runs)
        sfr = fm.get('silent_failure_rate',0)
        ax.scatter(sr, sfr, c=colors['synthetic'], marker='s', s=120, zorder=3, edgecolors='black', linewidths=0.5)
        ax.annotate(m.replace('deepseek-chat','deepseek-v3').replace('gpt-4o-mini','4o-mini').replace('gpt-4o','4o').replace('qwen3-max','qwen3'), (sr,sfr), fontsize=8, ha='left', xytext=(5,5), textcoords='offset points')
# uci
for m in uci:
    d = uci[m]
    runs = d.get('runs',[])
    if not runs: continue
    sr = sum(1 for r in runs if r['correct'])/len(runs)
    fail = [r for r in runs if not r['correct']]
    sfr = sum(1 for r in fail if r.get('is_silent'))/max(len(fail),1)
    ax.scatter(sr, sfr, c=colors['uci_real'], marker='^', s=120, zorder=3, edgecolors='black', linewidths=0.5)
    ax.annotate(m.replace('deepseek-chat','deepseek-v3').replace('gpt-4o-mini','4o-mini').replace('gpt-4o','4o').replace('qwen3-max','qwen3'), (sr,sfr), fontsize=8, ha='left', xytext=(5,5), textcoords='offset points')
# dsbench (final experiment)
for m in final:
    d = final[m]
    runs = d.get('runs',[])
    real_runs = [r for r in runs if r.get('is_real')]
    if not real_runs: continue
    sr = sum(1 for r in real_runs if r['correct'])/len(real_runs)
    fail = [r for r in real_runs if not r['correct']]
    sfr = sum(1 for r in fail if r.get('is_silent'))/max(len(fail),1)
    ax.scatter(sr, sfr, c=colors['dsbench_real'], marker='D', s=120, zorder=3, edgecolors='black', linewidths=0.5)
    ax.annotate(m.replace('gpt-4o-mini','4o-mini').replace('gpt-4o','4o'), (sr,sfr), fontsize=8, ha='left', xytext=(5,5), textcoords='offset points')

handles = [mpatches.Patch(color=colors[k], label=v) for k,v in [('synthetic','Synthetic (trap tasks)'),('uci_real','UCI real data'),('dsbench_real','DSBench real data')]]
ax.legend(handles=handles, loc='upper right', fontsize=9)
ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Silent Failure Rate (SFR)', fontsize=12)
ax.set_title('Failure Bifurcation: SFR vs Task Difficulty', fontsize=13)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.08)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'bifurcation.pdf')); plt.close()
print('Saved bifurcation.pdf')

# ---- Fig 2: Model comparison (main experiment) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
models_order = ['gpt-4o-mini','gpt-4o','deepseek-chat','qwen3-max']
labels = ['GPT-4o\nmini','GPT-4o','DeepSeek\nV3','Qwen3\nmax']
baselines = [main.get(m,{}).get('aggregate',{}).get('success_rate','0') for m in models_order]
baselines = [float(b.split('+')[0].strip()) for b in baselines]
sfrs = [main.get(m,{}).get('aggregate',{}).get('silent_failure_rate','0') for m in models_order]
sfrs = [float(s.split('+')[0].strip()) for s in sfrs]
x = np.arange(len(labels))
ax1.bar(x, baselines, 0.5, color='#2196F3', alpha=0.8, label='Success Rate')
ax1.set_ylabel('Success Rate'); ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_title('Task Success Rate (95 tasks, 3 runs)')
ax1.set_ylim(0, 1)
for i,v in enumerate(baselines): ax1.text(i, v+0.02, f'{v:.1%}', ha='center', fontsize=9)

ax2.bar(x, sfrs, 0.5, color='#FF5722', alpha=0.8, label='SFR')
ax2.set_ylabel('Silent Failure Rate'); ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_title('Silent Failure Rate (of failures)')
ax2.set_ylim(0, 1.1)
for i,v in enumerate(sfrs): ax2.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'model_comparison.pdf')); plt.close()
print('Saved model_comparison.pdf')

# ---- Fig 3: Stage distribution ----
fig, ax = plt.subplots(figsize=(8, 4.5))
stages = ['planning','tool_use','execution','interpretation']
stage_colors = ['#FFC107','#9C27B0','#F44336','#4CAF50']
data = []
for m in models_order:
    d = main.get(m,{}).get('failure_metrics',{}).get('stage_distribution',{})
    data.append([d.get(s,0) for s in stages])
data = np.array(data)
bottom = np.zeros(len(models_order))
for i, s in enumerate(stages):
    ax.bar(labels, data[:,i], 0.5, bottom=bottom, color=stage_colors[i], label=s)
    bottom += data[:,i]
ax.set_ylabel('Proportion of Failures')
ax.set_title('Failure Stage Distribution by Model')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'stage_distribution.pdf')); plt.close()
print('Saved stage_distribution.pdf')

# ---- Fig 4: Verifier ablation ----
fig, ax = plt.subplots(figsize=(8, 4.5))
base_sr = []; ver_sr = []
for m in models_order:
    b = main.get(m,{}).get('aggregate',{}).get('success_rate','0')
    v = main.get(m+'+verifier',{}).get('aggregate',{}).get('success_rate','0')
    base_sr.append(float(b.split('+')[0].strip()))
    ver_sr.append(float(v.split('+')[0].strip()))
x = np.arange(len(labels))
w = 0.35
ax.bar(x-w/2, base_sr, w, color='#2196F3', label='Baseline')
ax.bar(x+w/2, ver_sr, w, color='#FF9800', label='+ Verifier')
ax.set_ylabel('Success Rate'); ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title('Verifier Ablation: Baseline vs +Verifier')
ax.legend()
for i in range(len(labels)):
    delta = ver_sr[i] - base_sr[i]
    ax.text(i, max(base_sr[i],ver_sr[i])+0.03, f'{delta:+.1%}', ha='center', fontsize=9, fontweight='bold' if abs(delta)>0.1 else 'normal')
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'verifier_ablation.pdf')); plt.close()
print('Saved verifier_ablation.pdf')

# ---- Fig 5: Token economics ----
fig, ax = plt.subplots(figsize=(7, 5))
for m in models_order:
    d = main.get(m,{})
    agg = d.get('aggregate',{})
    tps = agg.get('token_per_success', None)
    sr = agg.get('success_rate', '0')
    if tps and tps != 'inf':
        sr_f = float(sr.split('+')[0].strip())
        tps_f = float(tps)
        ax.scatter(tps_f, sr_f, s=150, zorder=3, edgecolors='black', linewidths=0.5)
        ax.annotate(m.replace('gpt-4o-mini','4o-mini').replace('gpt-4o','4o').replace('deepseek-chat','deepseek-v3').replace('qwen3-max','qwen3'), (tps_f, sr_f), fontsize=8, xytext=(5,5), textcoords='offset points')
ax.set_xlabel('Tokens per Success'); ax.set_ylabel('Success Rate')
ax.set_title('Token Economics: Cost vs Accuracy')
ax.set_xscale('log')
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'token_economics.pdf')); plt.close()
print('Saved token_economics.pdf')

print('\nAll figures generated in figures/')
