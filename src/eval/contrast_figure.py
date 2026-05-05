"""Cross-dataset contrast figure: wiki vs review."""
import sys, os; sys.path.insert(0, '.')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.plotstyle import apply_style, MODEL_COLORS

apply_style()
os.makedirs('experiments/figures', exist_ok=True)
os.makedirs('report/figures', exist_ok=True)

models = ['TGN', 'JODIE', 'EdgeBank']
wiki_mrr =   [0.395, 0.327, 0.580]
wiki_std =   [0.074, 0.015, 0.0]
review_mrr = [0.315, 0.336, 0.025]  # seeds 1-2 mean (excluding collapsed seed 3)
review_std = [0.070, 0.005, 0.0]

fig, ax = plt.subplots(figsize=(3.5, 2.8))
x = np.arange(len(models))
w = 0.32

bars1 = ax.bar(x - w/2, wiki_mrr, w, yerr=wiki_std, capsize=3,
               label='tgbl-wiki (60% recurrence)', color='#B8D4E3', edgecolor='#2176AE', linewidth=0.8)
bars2 = ax.bar(x + w/2, review_mrr, w, yerr=review_std, capsize=3,
               label='tgbl-review (0% recurrence)', color='#FFD4B8', edgecolor='#F0803C', linewidth=0.8)

# Annotate the crossover
ax.annotate('TGNNs win\n(low recurrence)', xy=(0.5, 0.34), fontsize=6.5, ha='center',
            color='#F0803C', style='italic')
ax.annotate('EdgeBank wins\n(high recurrence)', xy=(2, 0.52), fontsize=6.5, ha='center',
            color='#2176AE', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('TGB MRR')
ax.set_ylim(0, 0.75)
ax.legend(loc='upper center', frameon=False, fontsize=7, ncol=1)
ax.set_title('Cross-Dataset Contrast')

fig.savefig('experiments/figures/contrast_datasets.pdf')
fig.savefig('report/figures/contrast_datasets.pdf')
print('Saved contrast_datasets.pdf')
