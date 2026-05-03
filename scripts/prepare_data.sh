#!/usr/bin/env bash
set -euo pipefail

DATASET="tgbl-wiki-v2"
DATA_DIR="DATA/${DATASET}"
S3_CSV="s3://achintan-gatech-coursework/temporal/wiki/dataset/tgbl-wiki_edgelist_v2.csv"
S3_VAL_NS="s3://achintan-gatech-coursework/temporal/wiki/dataset/tgbl-wiki_val_ns_v2.pkl"
S3_TEST_NS="s3://achintan-gatech-coursework/temporal/wiki/dataset/tgbl-wiki_test_ns_v2.pkl"

mkdir -p "${DATA_DIR}"

# Download raw CSV if not present
if [ ! -f "${DATA_DIR}/tgbl-wiki_edgelist_v2.csv" ]; then
    echo "Downloading raw CSV from S3..."
    aws s3 cp "${S3_CSV}" "${DATA_DIR}/tgbl-wiki_edgelist_v2.csv"
fi

# Download negative sample pickles
for f in "${S3_VAL_NS}" "${S3_TEST_NS}"; do
    fname=$(basename "$f")
    if [ ! -f "${DATA_DIR}/${fname}" ]; then
        echo "Downloading ${fname}..."
        aws s3 cp "$f" "${DATA_DIR}/${fname}"
    fi
done

# Convert to TGL format
echo "Converting TGB CSV to TGL format..."
python -m src.data.tgb_to_tgl \
    --input "${DATA_DIR}/tgbl-wiki_edgelist_v2.csv" \
    --output_dir "${DATA_DIR}"

# Build graph structures (CSR format for sampler)
echo "Building graph structures..."
python -m src.data.gen_graph --data "${DATASET}"

echo "Data preparation complete."
