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

if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    chown -R "$HOST_UID:$HOST_GID" /boxer/ckpts /boxer/sample_data /boxer/output 2>/dev/null || true
fi

exec "$@"
