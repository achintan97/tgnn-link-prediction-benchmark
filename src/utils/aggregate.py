"""Aggregate experiment results into main_table.csv and main_table.md"""
import json, glob, os
import numpy as np

results = {}
for f in sorted(glob.glob('experiments/results/*.json')):
    with open(f) as fh:
        r = json.load(fh)
    name = r.get('exp_name', os.path.basename(f).split('_seed')[0])
    if name not in results:
        results[name] = []
    results[name].append(r)

# Build table
rows = []
for name, runs in sorted(results.items()):
    aps = [r['test_ap'] for r in runs if r.get('test_ap')]
    aucs = [r['test_auc'] for r in runs if r.get('test_auc')]
    mrrs = [r['tgb_test_mrr'] for r in runs if r.get('tgb_test_mrr')]
    rows.append({
        'model': name,
        'legacy_ap': f'{np.mean(aps):.4f} ± {np.std(aps):.4f}' if aps else '-',
        'legacy_auc': f'{np.mean(aucs):.4f} ± {np.std(aucs):.4f}' if aucs else '-',
        'tgb_mrr': f'{np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}' if mrrs else '-',
        'n_seeds': len(runs),
    })

# Add EdgeBank (hardcoded from run)
rows.insert(0, {'model': 'EdgeBank(unlimited)', 'legacy_ap': '-', 'legacy_auc': '-', 'tgb_mrr': '0.5801', 'n_seeds': 1})

# CSV
os.makedirs('experiments/results', exist_ok=True)
with open('experiments/results/main_table.csv', 'w') as f:
    f.write('model,legacy_ap,legacy_auc,tgb_mrr,n_seeds\n')
    for r in rows:
        f.write(f'{r["model"]},{r["legacy_ap"]},{r["legacy_auc"]},{r["tgb_mrr"]},{r["n_seeds"]}\n')

# Markdown
with open('experiments/results/main_table.md', 'w') as f:
    f.write('| Model | Legacy AP | Legacy AUC | TGB MRR | Seeds |\n')
    f.write('|-------|-----------|------------|---------|-------|\n')
    for r in rows:
        f.write(f'| {r["model"]} | {r["legacy_ap"]} | {r["legacy_auc"]} | {r["tgb_mrr"]} | {r["n_seeds"]} |\n')

print('Generated experiments/results/main_table.csv')
print('Generated experiments/results/main_table.md')
for r in rows:
    print(f'  {r["model"]:<25} AP={r["legacy_ap"]:<20} MRR={r["tgb_mrr"]}')
