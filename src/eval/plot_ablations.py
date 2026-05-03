"""Ablation figures: memory ablation + sampler strategy."""
import sys, os; sys.path.insert(0, '.')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.plotstyle import apply_style, MODEL_COLORS

apply_style()
os.makedirs('experiments/figures', exist_ok=True)

# --- Memory ablation (use AP since TGB MRR is unreliable for no-memory) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

models = ['TGN', 'TGN (no mem)', 'TGAT']
ap_vals = [0.993, 0.681, 0.683]
mrr_vals = [0.395, None, 0.164]  # no-memory MRR unreliable
colors = ['#2176AE', '#2176AE', '#F0803C']
hatches = ['', '//', '']

bars = ax1.bar(range(len(models)), ap_vals, color=colors, edgecolor='black', linewidth=0.5)
bars[1].set_hatch('//')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, fontsize=8)
ax1.set_ylabel('Legacy AP')
ax1.set_ylim(0, 1.05)
ax1.set_title('Memory Ablation (AP)')
for i, v in enumerate(ap_vals):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=7)

# --- Sampler strategy ablation ---
configs = ['Recent\n(default)', 'Uniform', 'EdgeBank']
mrr_vals2 = [0.395, 0.332, 0.580]
mrr_std = [0.074, 0, 0]
colors2 = ['#2176AE', '#2176AE', '#8B8B8B']
hatches2 = ['', '//', '']

bars2 = ax2.bar(range(len(configs)), mrr_vals2, yerr=mrr_std, capsize=3,
                color=colors2, edgecolor='black', linewidth=0.5)
bars2[1].set_hatch('//')
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels(configs, fontsize=8)
ax2.set_ylabel('TGB MRR')
ax2.set_ylim(0, 0.75)
ax2.set_title('Sampler Strategy (TGN)')
for i, v in enumerate(mrr_vals2):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=7)

plt.tight_layout()
fig.savefig('experiments/figures/ablations.pdf')
fig.savefig('experiments/figures/ablations.png')
print('Saved ablations.pdf')
