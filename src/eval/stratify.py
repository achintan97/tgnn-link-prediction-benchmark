"""Stratified MRR analysis: recurring vs novel edges, degree quartiles.

For EdgeBank we compute MRR directly. For TGNNs we output the strata
assignments so GPU eval can use them.
"""
import sys, os, pickle; sys.path.insert(0, '.')
import numpy as np, pandas as pd

# Load data
df = pd.read_csv('DATA/tgbl-wiki-v2/edges.csv')

train_end = df[df['ext_roll'].gt(0)].index[0]
val_end = df[df['ext_roll'].gt(1)].index[0]
train_df = df[:train_end]
test_df = df[val_end:].copy()

# Compute strata
train_pairs = set(zip(train_df['src'], train_df['dst']))
test_df['is_recurring'] = [((s,d) in train_pairs) for s,d in zip(test_df['src'], test_df['dst'])]

# Source degree quartiles
src_degree = train_df.groupby('src').size()
test_df['src_degree'] = test_df['src'].map(src_degree).fillna(0).astype(int)
test_df['degree_quartile'] = pd.qcut(test_df['src_degree'].clip(lower=0), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')

# Time since last src activity
last_src_time = train_df.groupby('src')['time'].max()
test_df['time_since_src'] = test_df['time'] - test_df['src'].map(last_src_time)
test_df.loc[test_df['time_since_src'].isna(), 'time_since_src'] = float('inf')

print(f"Test edges: {len(test_df)}")
print(f"Recurring: {test_df['is_recurring'].sum()} ({test_df['is_recurring'].mean():.1%})")
print(f"Novel: {(~test_df['is_recurring']).sum()} ({(~test_df['is_recurring']).mean():.1%})")
print(f"\nDegree quartile counts:\n{test_df['degree_quartile'].value_counts().sort_index()}")

# EdgeBank MRR stratified
with open('DATA/tgbl-wiki-v2/tgbl-wiki_test_ns_v2.pkl', 'rb') as f:
    test_ns = pickle.load(f)

# Build EdgeBank memory (unlimited)
edge_mem = {}
for _, row in train_df.iterrows():
    edge_mem[(int(row['src']), int(row['dst']))] = float(row['time'])
for _, row in df[train_end:val_end].iterrows():
    edge_mem[(int(row['src']), int(row['dst']))] = float(row['time'])

rows_out = []
for stratum_col, stratum_name in [('is_recurring', 'recurring'), ('degree_quartile', 'degree')]:
    for group_val, group_df in test_df.groupby(stratum_col):
        mrrs = []
        for _, row in group_df.iterrows():
            key = (int(row['src']), int(row['dst']), int(row['time']))
            if key not in test_ns:
                continue
            negs = test_ns[key]
            pos_score = edge_mem.get((int(row['src']), int(row['dst'])), -1e9)
            neg_scores = [edge_mem.get((int(row['src']), int(n)), -1e9) for n in negs]
            rank = sum(1 for s in neg_scores if s > pos_score) + 1
            mrrs.append(1.0 / rank)
        label = str(group_val) if stratum_name == 'degree' else ('Recurring' if group_val else 'Novel')
        rows_out.append({'model': 'EdgeBank', 'stratum_type': stratum_name,
                         'stratum': label, 'n': len(mrrs), 'mrr': np.mean(mrrs) if mrrs else 0})
        print(f"  EdgeBank | {stratum_name}={label} | n={len(mrrs)} | MRR={np.mean(mrrs):.4f}")

# Save strata assignments for GPU eval
test_df[['src','dst','time','is_recurring','degree_quartile','src_degree','time_since_src']].to_csv(
    'experiments/results/test_strata.csv', index=False)

# Save EdgeBank stratified
pd.DataFrame(rows_out).to_csv('experiments/results/stratified_edgebank.csv', index=False)
print("\nSaved test_strata.csv and stratified_edgebank.csv")
