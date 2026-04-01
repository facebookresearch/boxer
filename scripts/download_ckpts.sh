#!/bin/bash
# Download BoxerNet, DinoV3, and OWLv2 checkpoints from HuggingFace.

set -e

CKPT_DIR="ckpts"
BASE_URL="https://huggingface.co/facebook/boxer/resolve/main"

FILES=(
    "boxernet_hw960in2x6d768.ckpt"
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
    "owlv2-base-patch16-ensemble.pt"
)

mkdir -p "$CKPT_DIR"

for f in "${FILES[@]}"; do
    if [ -f "$CKPT_DIR/$f" ]; then
        echo "Already exists: $CKPT_DIR/$f"
        continue
    fi
    echo "Downloading $f ..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$CKPT_DIR/$f" "$BASE_URL/$f"
    else
        curl -L -o "$CKPT_DIR/$f" "$BASE_URL/$f"
    fi
done

echo "Done. Checkpoints saved to $CKPT_DIR/"
