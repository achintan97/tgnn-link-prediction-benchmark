"""Generate stratified MRR figures from stratified CSV data."""
import sys, os; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.plotstyle import apply_style, MODEL_COLORS

apply_style()
os.makedirs('experiments/figures', exist_ok=True)

# Load all stratified results
dfs = []
for f in ['experiments/results/stratified_edgebank.csv',
          'experiments/results/stratified_tgnn.csv']:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f))
if not dfs:
    print("No stratified data found"); exit(1)
data = pd.concat(dfs, ignore_index=True)

# --- Recurring vs Novel ---
rec = data[(data['stratum_type'] == 'recurring')]
models_present = rec['model'].unique()
fig, ax = plt.subplots(figsize=(3.5, 2.5))
strata = ['Recurring', 'Novel']
x = np.arange(len(strata))
w = 0.8 / len(models_present)
for i, m in enumerate(models_present):
    vals = [rec[(rec['model']==m) & (rec['stratum']==s)]['mrr'].values for s in strata]
    vals = [v[0] if len(v) else 0 for v in vals]
    ax.bar(x + i*w - 0.4 + w/2, vals, w, label=m,
           color=MODEL_COLORS.get(m, '#999'), edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(strata)
ax.set_ylabel('TGB MRR')
ax.set_title('MRR by Edge Novelty')
ax.legend(frameon=False, fontsize=7)
fig.savefig('experiments/figures/stratified_recurring_vs_novel.pdf')
fig.savefig('experiments/figures/stratified_recurring_vs_novel.png')
print('Saved stratified_recurring_vs_novel.pdf')

# --- By degree quartile ---
deg = data[data['stratum_type'] == 'degree']
fig, ax = plt.subplots(figsize=(3.5, 2.5))
strata = ['Q1', 'Q2', 'Q3', 'Q4']
x = np.arange(len(strata))
w = 0.8 / len(models_present)
for i, m in enumerate(models_present):
    vals = [deg[(deg['model']==m) & (deg['stratum']==s)]['mrr'].values for s in strata]
    vals = [v[0] if len(v) else 0 for v in vals]
    ax.bar(x + i*w - 0.4 + w/2, vals, w, label=m,
           color=MODEL_COLORS.get(m, '#999'), edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)'])
ax.set_xlabel('Source Node Degree Quartile')
ax.set_ylabel('TGB MRR')
ax.set_title('MRR by Source Degree')
ax.legend(frameon=False, fontsize=7)
fig.savefig('experiments/figures/stratified_by_degree.pdf')
fig.savefig('experiments/figures/stratified_by_degree.png')
print('Saved stratified_by_degree.pdf')
