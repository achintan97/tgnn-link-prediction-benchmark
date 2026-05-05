"""Recurrence rate vs MRR scatter plot."""
import sys, os; sys.path.insert(0, '.')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.plotstyle import apply_style, MODEL_COLORS

apply_style()
os.makedirs('experiments/figures', exist_ok=True)
os.makedirs('report/figures', exist_ok=True)

# Data: (recurrence_rate, mrr, model, dataset)
points = [
    # wiki (60.6% recurrence)
    (60.6, 0.395, 'TGN', 'wiki'),
    (60.6, 0.327, 'JODIE', 'wiki'),
    (60.6, 0.580, 'EdgeBank', 'wiki'),
    # review (0.0% recurrence)
    (0.0, 0.315, 'TGN', 'review'),
    (0.0, 0.336, 'JODIE', 'review'),
    (0.0, 0.025, 'EdgeBank', 'review'),
]

fig, ax = plt.subplots(figsize=(3.5, 2.8))

markers = {'TGN': 'o', 'JODIE': 's', 'EdgeBank': 'D'}
for rec, mrr, model, ds in points:
    ax.scatter(rec, mrr, c=MODEL_COLORS[model], marker=markers[model],
               s=60, zorder=3, edgecolors='black', linewidth=0.5,
               label=model if ds == 'wiki' else None)

# Dashed line: EdgeBank ≈ recurrence rate
ax.plot([0, 65], [0, 0.65], '--', color=MODEL_COLORS['EdgeBank'], alpha=0.4, linewidth=1)
ax.text(35, 0.42, 'EdgeBank ≈\nrecurrence rate', fontsize=6, color=MODEL_COLORS['EdgeBank'], alpha=0.7)

# Dataset labels
ax.text(60.6, -0.05, 'tgbl-wiki', ha='center', fontsize=7, color='#555')
ax.text(0, -0.05, 'tgbl-review', ha='center', fontsize=7, color='#555')

ax.set_xlabel('Edge Recurrence Rate (%)')
ax.set_ylabel('TGB MRR')
ax.set_xlim(-5, 70)
ax.set_ylim(-0.05, 0.7)
ax.legend(loc='upper left', frameon=False, fontsize=7)
ax.set_title('Recurrence Rate vs Model Performance')

fig.savefig('experiments/figures/recurrence_vs_mrr.pdf')
fig.savefig('report/figures/recurrence_vs_mrr.pdf')
print('Saved recurrence_vs_mrr.pdf')
