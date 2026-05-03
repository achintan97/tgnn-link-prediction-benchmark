"""Compute/accuracy Pareto figure."""
import sys, os; sys.path.insert(0, '.')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.plotstyle import apply_style, MODEL_COLORS

apply_style()

# Data from RUNS.md — training time per epoch (seconds)
# TGN: ~3min/100ep = 1.8s/ep, JODIE: ~1.5min/100ep = 0.9s/ep, TGAT: ~3min/100ep = 1.8s/ep
models = {
    'TGN':   {'time_per_epoch': 1.8, 'mrr': [0.437, 0.291, 0.458], 'seeds': [1,2,3]},
    'JODIE': {'time_per_epoch': 0.9, 'mrr': [0.335, 0.306, 0.340], 'seeds': [1,2,3]},
    'TGAT':  {'time_per_epoch': 1.8, 'mrr': [0.164, 0.169],        'seeds': [1,2]},
}

fig, ax = plt.subplots(figsize=(3.5, 2.5))

for name, d in models.items():
    for i, mrr in enumerate(d['mrr']):
        ax.scatter(d['time_per_epoch'], mrr, c=MODEL_COLORS[name], s=40, zorder=3,
                   label=name if i == 0 else None, edgecolors='black', linewidth=0.5)

# EdgeBank reference line
ax.axhline(y=0.580, color=MODEL_COLORS['EdgeBank'], linestyle='--', linewidth=1, alpha=0.7)
ax.text(2.0, 0.585, 'EdgeBank (0 training)', fontsize=7, color=MODEL_COLORS['EdgeBank'])

# SOTA reference
ax.axhline(y=0.827, color=MODEL_COLORS['SOTA (TPNet)'], linestyle=':', linewidth=1, alpha=0.5)
ax.text(2.0, 0.832, 'SOTA (TPNet)', fontsize=7, color=MODEL_COLORS['SOTA (TPNet)'])

ax.set_xlabel('Training time per epoch (s)')
ax.set_ylabel('TGB MRR')
ax.set_ylim(0, 0.95)
ax.legend(loc='center left', frameon=False)
ax.set_title('Compute–Accuracy Tradeoff')

os.makedirs('experiments/figures', exist_ok=True)
fig.savefig('experiments/figures/pareto.pdf')
fig.savefig('experiments/figures/pareto.png')
print('Saved pareto.pdf')
