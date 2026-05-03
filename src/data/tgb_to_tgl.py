"""Convert tgbl-wiki-v2 CSV to TGL format, preserving original TGB node IDs."""
import os, sys, itertools
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'DATA', 'tgbl-wiki-v2')

def convert():
    csv_path = os.path.join(DATA_DIR, 'tgbl-wiki_edgelist_v2.csv')
    raw = pd.read_csv(csv_path, header=None, skiprows=1)
    print(f'Loaded: {raw.shape}')

    # TGB bipartite IDs: users 0-8226, items 0-999 in CSV
    # Offset items by num_users to match TGB neg sample pickle (items 8227-9226)
    src = raw[0].values.astype(np.int32)
    num_users = src.max() + 1  # 8227
    dst = raw[1].values.astype(np.int32) + num_users  # items become 8227-9226
    ts = raw[2].values.astype(np.float64)
    num_nodes = max(src.max(), dst.max()) + 1
    n = len(raw)

    # Edge features: cols 4-175 = 172 dims
    edge_feats = raw.iloc[:, 4:].values.astype(np.float32)

    # TGB official split: 70/15/15 chronological
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    ext_roll = np.zeros(n, dtype=np.int32)
    ext_roll[train_end:val_end] = 1
    ext_roll[val_end:] = 2

    # edges.csv
    edges_df = pd.DataFrame({
        'Unnamed: 0': np.arange(n), 'src': src, 'dst': dst,
        'time': ts, 'ext_roll': ext_roll, 'int_roll': ext_roll,
        'label': np.zeros(n, dtype=np.int32),
    })
    edges_df.to_csv(os.path.join(DATA_DIR, 'edges.csv'), index=False)
    torch.save(torch.from_numpy(edge_feats), os.path.join(DATA_DIR, 'edge_features.pt'))

    # CSR graph (ext_full.npz)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    idx_lists = [[] for _ in range(num_nodes)]
    ts_lists = [[] for _ in range(num_nodes)]
    eid_lists = [[] for _ in range(num_nodes)]
    for i in tqdm(range(n), desc='Building CSR'):
        s = src[i]
        idx_lists[s].append(dst[i])
        ts_lists[s].append(ts[i])
        eid_lists[s].append(i)
    for i in range(num_nodes):
        indptr[i + 1] = indptr[i] + len(idx_lists[i])
    indices = np.array(list(itertools.chain(*idx_lists)))
    ts_arr = np.array(list(itertools.chain(*ts_lists)))
    eid_arr = np.array(list(itertools.chain(*eid_lists)))
    for i in range(num_nodes):
        b, e = indptr[i], indptr[i + 1]
        if e > b:
            sidx = np.argsort(ts_arr[b:e])
            indices[b:e] = indices[b:e][sidx]
            ts_arr[b:e] = ts_arr[b:e][sidx]
            eid_arr[b:e] = eid_arr[b:e][sidx]
    np.savez(os.path.join(DATA_DIR, 'ext_full.npz'),
             indptr=indptr, indices=indices, ts=ts_arr, eid=eid_arr)

    print(f'Nodes: {num_nodes} (bipartite: users 0-{src.max()}, items {dst.min()}-{dst.max()})')
    print(f'Edges: {n}, Features: {edge_feats.shape[1]}-dim')
    print(f'Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}')
    print(f'Timestamp range: {ts.min():.0f} - {ts.max():.0f}')

if __name__ == '__main__':
    convert()
