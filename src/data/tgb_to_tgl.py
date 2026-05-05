"""Convert TGB datasets to TGL format, preserving TGB node IDs for neg sample lookup."""
import os, sys, itertools, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def convert(dataset='tgbl-wiki-v2'):
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'DATA', dataset)
    os.makedirs(data_dir, exist_ok=True)

    # Load from TGB
    from tgb.linkproppred.dataset import LinkPropPredDataset
    d = LinkPropPredDataset(name=dataset.replace('-v2', ''), root=os.path.join(os.path.dirname(__file__), '..', '..', 'DATA'))

    raw_src = d.full_data['sources'].astype(np.int32)
    raw_dst = d.full_data['destinations'].astype(np.int32)
    ts = d.full_data['timestamps'].astype(np.float64)
    n = len(raw_src)

    # Edge features
    ef = d.full_data.get('edge_feat')
    if ef is not None:
        edge_feats = ef.astype(np.float32)
    else:
        edge_feats = np.zeros((n, 1), dtype=np.float32)

    # Node ID mapping: for bipartite graphs, offset destinations
    if dataset == 'tgbl-wiki-v2':
        # Wiki: users 0-8226, items 0-999 → offset items by num_users
        num_users = raw_src.max() + 1
        src = raw_src
        dst = raw_dst + num_users
    elif dataset == 'tgbl-review-v2':
        # Review: users and items share ID space in TGB
        # Users = sources, Items = destinations. Offset items by max_user+1
        all_srcs = set(raw_src.tolist())
        all_dsts = set(raw_dst.tolist())
        num_users = max(all_srcs) + 1
        src = raw_src
        dst = raw_dst + num_users
    else:
        src = raw_src
        dst = raw_dst

    num_nodes = max(src.max(), dst.max()) + 1

    # Split: use TGB masks
    train_mask = d.train_mask
    val_mask = d.val_mask
    test_mask = d.test_mask
    ext_roll = np.zeros(n, dtype=np.int32)
    ext_roll[val_mask] = 1
    ext_roll[test_mask] = 2

    # edges.csv
    edges_df = pd.DataFrame({
        'Unnamed: 0': np.arange(n), 'src': src, 'dst': dst,
        'time': ts, 'ext_roll': ext_roll, 'int_roll': ext_roll,
        'label': np.zeros(n, dtype=np.int32),
    })
    edges_df.to_csv(os.path.join(data_dir, 'edges.csv'), index=False)
    torch.save(torch.from_numpy(edge_feats), os.path.join(data_dir, 'edge_features.pt'))

    # Download neg samples
    ns_name = dataset.replace('-v2', '')
    for split in ['test', 'val']:
        ns = d.get_processed_data()[f'{split}_ns'] if hasattr(d, 'get_processed_data') else None
        # Try loading from TGB cache
        import glob
        for f in glob.glob(f'DATA/**/*{ns_name}*{split}*ns*.pkl', recursive=True):
            import shutil
            target = os.path.join(data_dir, os.path.basename(f))
            if not os.path.exists(target):
                shutil.copy2(f, target)
                print(f'Copied neg samples: {f} -> {target}')

    # CSR graph
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
    np.savez(os.path.join(data_dir, 'ext_full.npz'),
             indptr=indptr, indices=indices, ts=ts_arr, eid=eid_arr)

    print(f'\n=== {dataset} ===')
    print(f'Nodes: {num_nodes} (users 0-{src.max()}, items {dst.min()}-{dst.max()})')
    print(f'Edges: {n}, Features: {edge_feats.shape[1]}-dim')
    print(f'Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgbl-wiki-v2')
    args = parser.parse_args()
    convert(args.dataset)
