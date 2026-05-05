"""Unified evaluator: legacy AP/AUC and TGB-official MRR."""
import pickle, os
import numpy as np
import torch
import dgl
from src.utils.utils import to_dgl_blocks, node_to_dgl_blocks, prepare_input


def evaluate_tgb_official(model, mailbox, sampler, sample_param, memory_param,
                          gnn_param, node_feats, edge_feats, df, val_edge_end,
                          neg_link_sampler, combine_first, data_dir='DATA/tgbl-wiki-v2',
                          per_edge_log_path=None):
    """Run TGB-official MRR evaluation: rank true dst among 999 hard negatives.

    Architecture-aware:
      - Memory models (TGN, JODIE): single 1001-node block per edge, memory updated.
      - No-memory models (TGAT, TGN-no-mem): batched pairwise scoring to avoid
        identical embeddings in shared blocks.

    Returns: dict with 'mrr' (float) and 'n_eval' (int).
    """
    # Infer neg sample filename from data_dir
    dataset_name = os.path.basename(data_dir).replace('-v2', '')
    ns_path = os.path.join(data_dir, f'{dataset_name}_test_ns_v2.pkl')
    if not os.path.exists(ns_path):
        print('No TGB neg samples found at', ns_path)
        return {'mrr': 0.0, 'n_eval': 0}

    with open(ns_path, 'rb') as f:
        test_ns = pickle.load(f)

    model.eval()
    test_df = df[val_edge_end:]
    mrr_list = []
    # Detect number of negatives from first entry
    first_key = next(iter(test_ns))
    n_neg = len(test_ns[first_key])
    has_memory = mailbox is not None
    NEG_BATCH = 100

    per_edge_rows = [] if per_edge_log_path else None

    with torch.no_grad():
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            src_i = int(row['src']); dst_i = int(row['dst']); ts_val = np.float32(row['time'])
            key = (src_i, dst_i, int(row['time']))
            has_tgb = key in test_ns
            if has_tgb:
                neg_nodes = test_ns[key].astype(np.int32)
            else:
                neg_nodes = neg_link_sampler.sample(n_neg).astype(np.int32)

            pos_score_val = None
            if has_memory:
                root_nodes = np.concatenate([[src_i], [dst_i], neg_nodes]).astype(np.int32)
                ts_batch = np.repeat(ts_val, n_neg + 2).astype(np.float32)
                if sampler is not None:
                    sampler.sample(root_nodes, ts_batch)
                    ret = sampler.get_ret()
                try:
                    if gnn_param['arch'] != 'identity':
                        mfgs = to_dgl_blocks(ret, sample_param['history'])
                    else:
                        mfgs = node_to_dgl_blocks(root_nodes, ts_batch)
                    mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
                    mailbox.prep_input_mails(mfgs[0])
                    pred_pos, pred_neg = model(mfgs, neg_samples=n_neg)
                except (RuntimeError, dgl._ffi.base.DGLError):
                    continue
                rank = (pred_neg.squeeze() > pred_pos.squeeze()).sum().item() + 1
                # Proper tie-breaking
                rank += (pred_neg.squeeze() == pred_pos.squeeze()).sum().item() // 2
                pos_score_val = pred_pos.item()

                # Update memory
                eid = test_df.iloc[idx:idx+1]['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param.get('deliver_to') == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts_batch, mem_edge_feats, block, neg_samples=n_neg)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=n_neg)
            else:
                # No memory: pairwise scoring
                rn = np.array([src_i, dst_i, dst_i], dtype=np.int32)
                ts3 = np.repeat(ts_val, 3).astype(np.float32)
                sampler.sample(rn, ts3); ret = sampler.get_ret()
                mfgs = to_dgl_blocks(ret, sample_param['history'])
                mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
                pp, _ = model(mfgs, neg_samples=1)
                pos_score = pp.item()
                pos_score_val = pos_score
                neg_scores = []
                for b in range(0, n_neg, NEG_BATCH):
                    chunk = neg_nodes[b:b+NEG_BATCH]; bs = len(chunk)
                    rn = np.concatenate([np.repeat(src_i, bs), chunk, chunk]).astype(np.int32)
                    ts_b = np.repeat(ts_val, bs * 3).astype(np.float32)
                    sampler.sample(rn, ts_b); ret = sampler.get_ret()
                    mfgs = to_dgl_blocks(ret, sample_param['history'])
                    mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
                    pp_b, _ = model(mfgs, neg_samples=1)
                    neg_scores.extend(pp_b.squeeze().cpu().tolist() if bs > 1 else [pp_b.item()])
                rank = sum(1 for s in neg_scores if s > pos_score) + 1
                # Proper tie-breaking: average rank among tied scores
                n_tied = sum(1 for s in neg_scores if s == pos_score)
                rank += n_tied // 2

            if has_tgb:
                mrr_list.append(1.0 / rank)

            if per_edge_rows is not None and has_tgb:
                per_edge_rows.append(f'{src_i},{dst_i},{row["time"]},{pos_score_val},{rank},{1.0/rank}')

    if per_edge_log_path and per_edge_rows:
        with open(per_edge_log_path, 'w') as f:
            f.write('src,dst,time,pos_score,rank,reciprocal_rank\n')
            f.write('\n'.join(per_edge_rows) + '\n')

    mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    return {'mrr': mrr, 'n_eval': len(mrr_list)}
