# Quarantined Results

## tgat_tgat_seed3.json

**Status:** Excluded from all aggregates.

**Issue:** TGAT seed=3 was evaluated using the old single-block TGB eval path,
which places all 1001 nodes (src + dst + 999 negatives) in one DGL message-passing
block. For models without memory (TGAT), this produces identical embeddings for
all candidates, yielding MRR ≈ 1.0 — clearly incorrect.

The file was removed from `experiments/results/` and is preserved here for
transparency. See `experiments/RUNS.md` for full documentation.

**Original values:** `tgb_test_mrr: 1.0`, `test_ap: 0.683`

TGAT seeds 1 and 2 were re-evaluated using the corrected batched pairwise
scoring path and produce MRR ~0.16, consistent with the published TGB
leaderboard value of 0.141 ± 0.007.
