"""Dual-protocol comparison: Legacy AP vs TGB MRR — the money figure."""
import sys, os; sys.path.insert(0, '.')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.plotstyle import apply_style, MODEL_COLORS

apply_style()

# Data from experiments
models =     ['TGAT',  'JODIE', 'TGN',  'EdgeBank']
legacy_ap =  [0.683,   0.988,   0.993,   None]
legacy_auc = [0.766,   0.990,   0.994,   None]
tgb_mrr =    [0.167,   0.327,   0.395,   0.580]
tgb_mrr_std= [0.003,   0.015,   0.074,   0.0]
sota_mrr = 0.827  # TPNet

fig, ax = plt.subplots(figsize=(3.5, 2.8))
x = np.arange(len(models))
w = 0.25

# Legacy AP bars
ap_vals = [v if v else 0 for v in legacy_ap]
ap_mask = [v is not None for v in legacy_ap]
bars1 = ax.bar(x[ap_mask] - w, [ap_vals[i] for i in range(len(models)) if ap_mask[i]],
               w, label='Legacy AP', color='#B8D4E3', edgecolor='#2176AE', linewidth=0.5)

# Legacy AUC bars
auc_vals = [v if v else 0 for v in legacy_auc]
bars2 = ax.bar(x[ap_mask], [auc_vals[i] for i in range(len(models)) if ap_mask[i]],
               w, label='Legacy AUC', color='#C8E6C9', edgecolor='#57B894', linewidth=0.5)

# TGB MRR bars
bars3 = ax.bar(x + w, tgb_mrr, w, yerr=tgb_mrr_std, capsize=2,
               label='TGB MRR', color=[MODEL_COLORS.get(m, '#666') for m in models],
               edgecolor='black', linewidth=0.5)

# SOTA reference line
ax.axhline(y=sota_mrr, color=MODEL_COLORS['SOTA (TPNet)'], linestyle='--', linewidth=1, alpha=0.7)
ax.text(len(models)-0.5, sota_mrr + 0.02, 'SOTA (TPNet)', fontsize=7,
        color=MODEL_COLORS['SOTA (TPNet)'], ha='right')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.legend(loc='upper left', frameon=False, ncol=1)
ax.set_title('Legacy vs TGB Evaluation Protocol')

os.makedirs('experiments/figures', exist_ok=True)
fig.savefig('experiments/figures/eval_protocol_gap.pdf')
fig.savefig('experiments/figures/eval_protocol_gap.png')
print('Saved eval_protocol_gap.pdf')
