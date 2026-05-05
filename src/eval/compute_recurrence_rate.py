"""Compute edge recurrence rates for each dataset."""
import sys, os; sys.path.insert(0, '.')
import numpy as np, pandas as pd

def compute_recurrence(edges_csv):
    df = pd.read_csv(edges_csv)
    te = df[df['ext_roll'].gt(0)].index[0]
    ve = df[df['ext_roll'].gt(1)].index[0]
    train_df = df[:te]
    test_df = df[ve:]
    
    train_pairs = set(zip(train_df['src'], train_df['dst']))
    train_srcs = set(train_df['src'])
    
    n = len(test_df)
    recurring = sum(1 for s,d in zip(test_df['src'], test_df['dst']) if (s,d) in train_pairs)
    novel_known_src = sum(1 for s,d in zip(test_df['src'], test_df['dst']) if (s,d) not in train_pairs and s in train_srcs)
    cold_start = sum(1 for s in test_df['src'] if s not in train_srcs)
    
    return {
        'n_test': n,
        'recurring': recurring / n,
        'novel_known_src': novel_known_src / n,
        'cold_start': cold_start / n,
    }

rows = []
for dataset in ['tgbl-wiki-v2', 'tgbl-review-v2']:
    path = f'DATA/{dataset}/edges.csv'
    if os.path.exists(path):
        r = compute_recurrence(path)
        r['dataset'] = dataset
        rows.append(r)
        print(f"{dataset}: recurring={r['recurring']:.1%}, novel_known_src={r['novel_known_src']:.1%}, cold_start={r['cold_start']:.1%} (n={r['n_test']})")
    else:
        print(f"{dataset}: edges.csv not found at {path}")

if rows:
    pd.DataFrame(rows).to_csv('experiments/results/recurrence_rates.csv', index=False)
    print('\nSaved recurrence_rates.csv')
