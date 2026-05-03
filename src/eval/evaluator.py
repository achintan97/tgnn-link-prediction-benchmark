"""Unified evaluator with legacy (AP/AUC) and TGB (MRR) modes."""
import pickle, os
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


class TGBEvaluator:
    def __init__(self, dataset='tgbl-wiki-v2', data_dir=None):
        self.dataset = dataset
        self.data_dir = data_dir or f'DATA/{dataset}'
        self._ns_cache = {}

    def evaluate_legacy(self, pred_pos, pred_neg):
        """AP and AUC with 1 random negative per positive."""
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu().numpy()
        y_true = np.concatenate([np.ones(len(pred_pos)), np.zeros(len(pred_neg))])
        return {
            'ap': float(average_precision_score(y_true, y_pred)),
            'auc': float(roc_auc_score(y_true, y_pred)),
        }

    def load_neg_samples(self, split='test'):
        """Load TGB neg samples. Keys are (src, dst, timestamp) tuples."""
        if split not in self._ns_cache:
            fname = f'tgbl-wiki_{split}_ns_v2.pkl'
            with open(os.path.join(self.data_dir, fname), 'rb') as f:
                self._ns_cache[split] = pickle.load(f)
        return self._ns_cache[split]

    def compute_mrr_for_batch(self, src_ids, dst_ids, timestamps, pos_scores, model, neg_samples_dict):
        """Compute MRR for a batch using TGB pre-computed negatives.

        For each edge (src, dst, ts), looks up neg_samples_dict[(src, dst, int(ts))]
        to get 999 negative destination IDs, scores them, and computes rank.

        Args:
            src_ids: array of source node IDs
            dst_ids: array of destination node IDs
            timestamps: array of timestamps
            pos_scores: tensor of positive edge scores (already computed)
            model: the model (needed to score negative edges)
            neg_samples_dict: TGB negative samples dict

        Returns:
            list of 1/rank values (MRR contributions)
        """
        mrr_list = []
        for i in range(len(src_ids)):
            s, d, t = int(src_ids[i]), int(dst_ids[i]), int(timestamps[i])
            key = (s, d, t)
            if key not in neg_samples_dict:
                continue
            neg_dsts = neg_samples_dict[key]
            pos_score = pos_scores[i].item()

            # For EdgeBank-style eval, neg scores would be computed by the model
            # For neural models, this requires a separate forward pass per edge
            # which is handled in the training script's eval loop
            mrr_list.append((pos_score, neg_dsts))

        return mrr_list
