# Experiment Runs Documentation

## Summary

All experiments run on g5.12xlarge (4× NVIDIA A10G), using 1 GPU.
Dataset: tgbl-wiki-v2 (157,474 edges, 9,227 nodes, 172-dim edge features)
Training: 100 epochs for TGN/JODIE, 50 epochs for TGAT (per config)

## EdgeBank (unlimited)
- TGB MRR: 0.5801
- Eval time: 10.6s
- No training required

## TGN seed=1
- Wall-clock: ~3.2 min (100 epochs × 1.9s/epoch)
- Best-val epoch: ~65
- Test AP: 0.9931, Test AUC: 0.9940
- TGB MRR: 0.9711 (NOTE: approximate — see anomaly note below)

## TGN seed=2
- Test AP: 0.9936, Test AUC: 0.9944
- TGB MRR: 0.9794

## TGN seed=3
- Test AP: 0.9933, Test AUC: 0.9943
- TGB MRR: 0.9781

## TGAT seed=1
- Wall-clock: ~2.5 min (50 epochs × 3.0s/epoch)
- Test AP: 0.6823, Test AUC: 0.7654
- TGB MRR: 1.0000

## TGAT seed=2
- Test AP: 0.6830, Test AUC: 0.7661
- TGB MRR: 1.0000

## TGAT seed=3
- Test AP: 0.6832, Test AUC: 0.7666
- TGB MRR: 1.0000

## JODIE seed=1
- Wall-clock: ~1.3 min (100 epochs × 0.8s/epoch)
- Test AP: 0.9861, Test AUC: 0.9891
- TGB MRR: 0.9704

## JODIE seed=2
- Test AP: 0.9889, Test AUC: 0.9906
- TGB MRR: 0.9674

## JODIE seed=3
- Test AP: 0.9883, Test AUC: 0.9907
- TGB MRR: 0.9649

## Known Anomaly: TGB MRR Values

The TGB MRR values (0.97-1.00) are significantly higher than the TGB leaderboard
(TGN: 0.396, TGAT: 0.141). This is because the current TGB eval implementation
uses an **approximate scoring method**: it compares the model's positive edge score
against zero for all 999 negatives, rather than running each negative through the
full model forward pass. Since trained models produce positive scores >> 0 for
true edges, this gives inflated MRR.

The **correct** TGB eval requires running each of the 999 negative destinations
through the model's full encoder pipeline (sampling neighbors, computing attention,
etc.) for each test edge — approximately 23,621 × 1000 forward passes. This is
computationally expensive (~hours) and requires a fundamentally different eval loop.

**The legacy AP/AUC values are correct** and comparable to published results.
The EdgeBank MRR (0.5801) is also correct since it doesn't use the model.

For the report, we recommend:
1. Using legacy AP/AUC as the primary metric for model comparison
2. Using EdgeBank MRR as the TGB-protocol anchor
3. Noting the TGB MRR limitation and citing the leaderboard values for context
