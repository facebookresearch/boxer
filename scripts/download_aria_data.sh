#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# Download sample Aria sequences from HuggingFace.
# Usage:
#   bash scripts/download_aria_data.sh              # download all sequences
#   bash scripts/download_aria_data.sh hohen_gen1   # download one sequence

set -e

DATA_DIR="sample_data"
BASE_URL="https://huggingface.co/datasets/facebook/boxer/resolve/main"

ALL_SEQS=("hohen_gen1" "nym10_gen1" "cook0_gen2")

FILES=(
    "main.vrs"
    "closed_loop_trajectory.csv"
    "online_calibration.jsonl"
    "semidense_observations.csv.gz"
    "semidense_points.csv.gz"
)

download_file() {
    local url="$1"
    local dest="$2"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    else
        curl -L -o "$dest" "$url"
    fi
}

download_seq() {
    local seq="$1"
    local seq_dir="$DATA_DIR/$seq"
    mkdir -p "$seq_dir"
    echo "Downloading $seq ..."
    for f in "${FILES[@]}"; do
        if [ -f "$seq_dir/$f" ]; then
            echo "  Already exists: $f"
            continue
        fi
        echo "  $f"
        download_file "$BASE_URL/$seq/$f" "$seq_dir/$f"
    done
}

if [ $# -gt 0 ]; then
    for seq in "$@"; do
        download_seq "$seq"
    done
else
    for seq in "${ALL_SEQS[@]}"; do
        download_seq "$seq"
    done
fi

echo "Done. Data saved to $DATA_DIR/"
