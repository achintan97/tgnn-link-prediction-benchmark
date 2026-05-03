#!/bin/bash
# Reproduce main_table.csv from saved checkpoints (no retraining)
# Usage: bash scripts/reproduce_main_table.sh
set -e
echo "=== Reproducing main table from checkpoints ==="

# EdgeBank (no checkpoint needed)
PYTHONPATH=. python3 -m src.baselines.edgebank --data tgbl-wiki-v2 --variant unlimited

# Aggregate existing results
PYTHONPATH=. python3 -m src.utils.aggregate

echo "=== Done. See experiments/results/main_table.csv ==="
cat experiments/results/main_table.md
