#!/usr/bin/env bash
# Smoke test: run TGAT for 1 epoch on tgbl-wiki-v2
set -euo pipefail

DATASET="tgbl-wiki-v2"
CONFIG="configs/models/tgat.yml"
GPU="${1:-0}"

echo "=== Smoke Test: TGAT on ${DATASET} (1 epoch) ==="

# Override epoch count to 1 via a temp config
TMPCONFIG=$(mktemp /tmp/smoke_XXXXXX.yml)
sed 's/epoch: [0-9]*/epoch: 1/' "${CONFIG}" > "${TMPCONFIG}"

python -m scripts.train \
    --data "${DATASET}" \
    --config "${TMPCONFIG}" \
    --gpu "${GPU}" \
    --eval_neg_samples 1

rm -f "${TMPCONFIG}"
echo "=== Smoke test complete ==="
