# TGNN Link Prediction Benchmark

**CS 7643 Deep Learning — Georgia Tech — Spring 2025**

Benchmarking Temporal Graph Neural Networks for dynamic link prediction on the **tgbl-wiki-v2** dataset using the TGL framework (Zhou et al., VLDB 2022).

## Project Thesis

We evaluate three temporal GNN architectures — TGN, TGAT, and JODIE — under a standardized evaluation protocol using the Temporal Graph Benchmark (TGB). We compare legacy 1-negative AP/AUC evaluation against TGB's fixed-negative-set MRR, and include an EdgeBank heuristic baseline to contextualize learned model performance.

## Repo Layout

```
tgnn-link-prediction-benchmark/
├── src/
│   ├── data/           # Data conversion (tgb_to_tgl.py) and graph building
│   ├── models/         # TGL model code: modules, memory, layers, sampler
│   ├── baselines/      # EdgeBank baseline
│   ├── eval/           # TGBEvaluator (legacy AP/AUC + TGB MRR)
│   └── utils/          # Utilities and CSV logger
├── configs/
│   ├── models/         # tgn.yml, tgat.yml, jodie.yml
│   └── ablations/      # Ablation configs (future)
├── scripts/            # Training scripts, data prep, smoke test
├── experiments/
│   ├── results/        # results.csv (tracked)
│   ├── logs/           # TensorBoard / text logs (gitignored)
│   ├── checkpoints/    # Model checkpoints (gitignored)
│   └── figures/        # Generated plots (gitignored)
├── report/             # Final report LaTeX / PDF
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the C++ sampler
cd src/models && python setup.py build_ext --inplace && cd ../..

# 3. Prepare data
bash scripts/prepare_data.sh

# 4. Smoke test (1 epoch of TGAT)
bash scripts/smoke_test.sh
```

## Results

| Dataset | Model | Legacy AP | TGB MRR | Seeds |
|---------|-------|-----------|---------|-------|
| tgbl-wiki-v2 | EdgeBank | — | 0.580 | 1 |
| tgbl-wiki-v2 | **TGN** | 0.993 ± 0.000 | **0.395 ± 0.074** | 3 |
| tgbl-wiki-v2 | JODIE | 0.988 ± 0.000 | 0.327 ± 0.015 | 3 |
| tgbl-wiki-v2 | TGAT | 0.683 ± 0.001 | 0.167 ± 0.003 | 2 |
| tgbl-review-v2 | EdgeBank | — | 0.025 | 1 |
| tgbl-review-v2 | **TGN** | 0.952 ± 0.001 | **0.363 ± 0.089** | 3 |
| tgbl-review-v2 | JODIE | 0.943 ± 0.006 | 0.345 ± 0.013 | 3 |

**Key finding 1 — Evaluation protocol matters.** Under TGB's 999-negative protocol, all three TGNNs score below the parameter-free EdgeBank heuristic on tgbl-wiki-v2. Legacy 1-negative AP/AUC (>0.99 for TGN) dramatically overstate model quality.

**Key finding 2 — Edge-recurrence rate explains which class wins.** On tgbl-wiki-v2 (60% test edges recur from training), EdgeBank dominates by memorization. On tgbl-review-v2 (0% recurrence), EdgeBank collapses to near-random while TGNNs reach 0.34–0.36 MRR. This contrast suggests recurrence rate predicts, before any training, whether a learned temporal model is likely to outperform memorization.

### Reproducing the main results

```bash
# From checkpoints (no retraining, ~30 min):
PYTHONPATH=. bash scripts/reproduce_main_table.sh 0

# From scratch (~2 hours on a single A10G GPU):
PYTHONPATH=. bash scripts/run_experiments.sh 0
```

### Reproducing the cross-dataset experiment

```bash
# Prepare review data
python src/data/tgb_to_tgl.py --dataset tgbl-review-v2

# Train TGN and JODIE on review (3 seeds each, ~13 GPU-hours total)
for s in 1 2 3; do
  python scripts/train.py --data tgbl-review-v2 \
    --config configs/models/tgn_review.yml --seed $s --exp_name tgn_review --gpu 0
  python scripts/train.py --data tgbl-review-v2 \
    --config configs/models/jodie_review.yml --seed $s --exp_name jodie_review --gpu 0
done
```

## Data Preparation

The `tgbl-wiki-v2` dataset (157K edges, 8K nodes, 172-dim edge features) is downloaded from S3 and converted to TGL's CSR format:

```bash
bash scripts/prepare_data.sh
```

This creates `DATA/tgbl-wiki-v2/` with `edges.csv`, `edge_features.pt`, and `ext_full.npz`.

## Training

```bash
# TGN
python -m scripts.train --data tgbl-wiki-v2 --config configs/models/tgn.yml --gpu 0

# TGAT
python -m scripts.train --data tgbl-wiki-v2 --config configs/models/tgat.yml --gpu 0

# JODIE
python -m scripts.train --data tgbl-wiki-v2 --config configs/models/jodie.yml --gpu 0

# EdgeBank baseline
python -m src.baselines.edgebank --data tgbl-wiki-v2 --variant unlimited
```

## Reproducing Results

1. Run `bash scripts/prepare_data.sh`
2. Train each model with 3 seeds: `--seed 0`, `--seed 1`, `--seed 2`
3. Results are logged to `experiments/results/results.csv`

## Hardware Expectations

| Model | GPU Memory | Time/Epoch (est.) |
|-------|-----------|-------------------|
| TGAT  | ~4 GB     | ~2 min            |
| TGN   | ~6 GB     | ~3 min            |
| JODIE | ~2 GB     | ~1 min            |

Tested on NVIDIA A100 / RTX 3090. CPU-only training is possible but significantly slower.

## Citations

```bibtex
@article{zhou2022tgl,
  title={TGL: A General Framework for Temporal GNN Training on Billion-Scale Graphs},
  author={Zhou, Hongkuan and Zheng, Da and Nisa, Israt and Ioannidis, Vasileios and Song, Xiang and Karypis, George},
  journal={Proceedings of the VLDB Endowment},
  year={2022}
}

@article{huang2024temporal,
  title={Temporal Graph Benchmark for Machine Learning on Temporal Graphs},
  author={Huang, Shenyang and Poursafaei, Farimah and Danovitch, Jacob and Fey, Matthias and Hu, Weihua and Rossi, Emanuele and Leskovec, Jure and Bronstein, Michael and Rabusseau, Guillaume and Rabbany, Reihaneh},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
