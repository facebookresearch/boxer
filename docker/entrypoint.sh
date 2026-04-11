#!/bin/bash

set -e

CKPT_DIR="/boxer/ckpts"
SENTINEL="${CKPT_DIR}/.downloaded"

if [ ! -f "$SENTINEL" ]; then
    echo "WARNING: Checkpoints not found, proceeding to download..."
    bash /boxer/scripts/download_ckpts.sh
    touch "$SENTINEL"
    echo "INFO: Checkpoints downloaded and saved to $CKPT_DIR/"
fi

DATA_DIR="/boxer/sample_data"
DATA_SENTINEL="${DATA_DIR}/.downloaded"

if [ ! -f "$DATA_SENTINEL" ]; then
    echo "INFO: Sample Aria data not found, downloading..."
    bash /boxer/scripts/download_aria_data.sh
    touch "$DATA_SENTINEL"
    echo "INFO: Sample data downloaded to $DATA_DIR/"
fi

exec "$@"
