#!/bin/bash
# Run full experiment matrix: 3 models × 3 seeds + EdgeBank
set -e
GPU=${1:-0}
SEEDS="1 2 3"
DATA="tgbl-wiki-v2"
PYTHON=${PYTHON:-python3.12}

echo "=== tgbl-wiki-v2 Experiment Matrix ==="
echo "GPU: $GPU, Seeds: $SEEDS, Python: $PYTHON"

# EdgeBank
echo ""
echo "=== EdgeBank ==="
$PYTHON -m src.baselines.edgebank --data $DATA --variant unlimited

# Neural models
for MODEL in tgn tgat jodie; do
    CONFIG="configs/models/${MODEL}.yml"
    for SEED in $SEEDS; do
        CKPT="experiments/checkpoints/${MODEL}_seed${SEED}.pkl"
        if [ -f "$CKPT" ]; then
            echo "SKIP $MODEL seed=$SEED (checkpoint exists)"
            continue
        fi
        echo ""
        echo "=== $MODEL seed=$SEED ==="
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train.py \
            --data $DATA --config $CONFIG --gpu 0 --seed $SEED \
            --exp_name $MODEL \
            2>&1 | tee experiments/logs/${MODEL}_seed${SEED}.log
    done
done

echo ""
echo "=== All experiments complete ==="
ls -la experiments/checkpoints/
ls -la experiments/results/
