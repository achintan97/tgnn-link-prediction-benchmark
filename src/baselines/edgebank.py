"""EdgeBank baselines for temporal link prediction with TGB evaluation."""
import argparse, os, pickle, time
import numpy as np
import pandas as pd
from collections import defaultdict


def run_edgebank(data_dir, variant='unlimited'):
    df = pd.read_csv(os.path.join(data_dir, 'edges.csv'))
    train_end = df[df['ext_roll'].gt(0)].index[0]
    val_end = df[df['ext_roll'].gt(1)].index[0]

    # Load TGB neg samples — keys are (src, dst, timestamp) tuples
    with open(os.path.join(data_dir, 'tgbl-wiki_test_ns_v2.pkl'), 'rb') as f:
        test_ns = pickle.load(f)
    with open(os.path.join(data_dir, 'tgbl-wiki_val_ns_v2.pkl'), 'rb') as f:
        val_ns = pickle.load(f)

    # Build memory from training edges
    memory = defaultdict(set)  # src -> set of dst
    ts_mem = defaultdict(dict)  # src -> {dst: last_ts}

    for idx in range(train_end):
        s, d, t = int(df.iloc[idx]['src']), int(df.iloc[idx]['dst']), float(df.iloc[idx]['time'])
        memory[s].add(d)
        ts_mem[s][d] = t

    # Process val edges to update memory before test eval
    for idx in range(train_end, val_end):
        s, d, t = int(df.iloc[idx]['src']), int(df.iloc[idx]['dst']), float(df.iloc[idx]['time'])
        memory[s].add(d)
        ts_mem[s][d] = t

    # Evaluate on test set using TGB neg samples
    t0 = time.time()
    mrr_list = []
    for idx in range(val_end, len(df)):
        s = int(df.iloc[idx]['src'])
        d = int(df.iloc[idx]['dst'])
        t = float(df.iloc[idx]['time'])

        # TGB key format: (src, dst, timestamp)
        key = (s, d, int(t)) if (s, d, int(t)) in test_ns else None
        if key is None:
            # Try float timestamp
            for candidate_key in [(s, d, int(t)), (s, d, t)]:
                if candidate_key in test_ns:
                    key = candidate_key
                    break
        if key is None:
            continue

        neg_dsts = test_ns[key]

        # Score: 1.0 if (src, dst) in memory, else 0.0
        pos_score = 1.0 if d in memory.get(s, set()) else 0.0
        neg_scores = np.array([1.0 if int(nd) in memory.get(s, set()) else 0.0 for nd in neg_dsts])

        # Rank: how many negatives score >= positive
        rank = (neg_scores > pos_score).sum() + (neg_scores == pos_score).sum() // 2 + 1
        mrr_list.append(1.0 / rank)

        # Update memory with this edge
        memory[s].add(d)
        ts_mem[s][d] = t

    elapsed = time.time() - t0
    mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    print(f'EdgeBank ({variant}): MRR={mrr:.4f}, evaluated={len(mrr_list)}, time={elapsed:.1f}s')
    return {'mrr': mrr, 'n_eval': len(mrr_list)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='tgbl-wiki-v2')
    parser.add_argument('--variant', type=str, default='unlimited', choices=['unlimited', 'time_window'])
    args = parser.parse_args()
    data_dir = os.path.join('DATA', args.data)
    run_edgebank(data_dir, args.variant)
