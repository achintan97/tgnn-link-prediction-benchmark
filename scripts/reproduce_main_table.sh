#!/bin/bash
# Reproduce main_table.csv from existing checkpoints (no retraining)
# Usage: PYTHONPATH=. bash scripts/reproduce_main_table.sh [GPU]
set -e
GPU=${1:-0}
DATA="tgbl-wiki-v2"
PYTHON=${PYTHON:-python3.12}

echo "=== Reproducing Main Table from Checkpoints ==="

# EdgeBank
echo "=== EdgeBank ==="
PYTHONPATH=. $PYTHON -m src.baselines.edgebank --data $DATA --variant unlimited

# Neural models — eval only (loads checkpoint, skips training)
for MODEL in tgn tgat jodie; do
    CONFIG="configs/models/${MODEL}.yml"
    for SEED in 1 2 3; do
        CKPT="experiments/checkpoints/${MODEL}_seed${SEED}.pkl"
        if [ ! -f "$CKPT" ]; then
            echo "SKIP $MODEL seed=$SEED (no checkpoint at $CKPT)"
            continue
        fi
        echo ""
        echo "=== $MODEL seed=$SEED ==="
        CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYTHON scripts/train.py \
            --data $DATA --config $CONFIG --gpu 0 --seed $SEED \
            --exp_name $MODEL --eval_only \
            2>&1 | tee experiments/logs/${MODEL}_seed${SEED}_eval.log
    done
done

echo ""
echo "=== Aggregating results ==="
PYTHONPATH=. $PYTHON -m src.utils.aggregate
echo ""
cat experiments/results/main_table.md
