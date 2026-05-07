#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a BoxerNet training checkpoint to the lightweight OSS inference format.

The output checkpoint keeps only:
  - cfg
  - model

By default, it strips all `sam.*` weights and removes model config entries that
point at local training assets.

Usage:
    python scripts/export_boxernet_oss_ckpt.py \
        /path/to/last.ckpt \
        /path/to/boxernet_hw960in4x6d768-wssxpf9p.ckpt
"""

import argparse
import copy
import os

import torch


DEFAULT_STRIP_PREFIXES = ("sam.",)
DEFAULT_DROP_MODEL_CFG_KEYS = ("vggt_ckpt",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a BoxerNet training checkpoint to OSS inference format."
    )
    parser.add_argument("input_ckpt", help="Path to the source training checkpoint")
    parser.add_argument("output_ckpt", help="Path to the exported OSS checkpoint")
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=list(DEFAULT_STRIP_PREFIXES),
        help="Model key prefix to drop. Can be passed multiple times.",
    )
    parser.add_argument(
        "--drop-model-cfg-key",
        action="append",
        default=list(DEFAULT_DROP_MODEL_CFG_KEYS),
        help="Model cfg key to remove. Can be passed multiple times.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validating the exported model against BoxerNet.state_dict().",
    )
    return parser.parse_args()


def load_model_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"]
    raise KeyError("checkpoint must contain a 'model' or 'state_dict' mapping")


def strip_model_state(
    model_state: dict[str, torch.Tensor], prefixes: tuple[str, ...]
) -> tuple[dict[str, torch.Tensor], int]:
    stripped = {}
    removed = 0
    for key, value in model_state.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            removed += 1
            continue
        stripped[key] = value
    return stripped, removed


def clean_cfg(cfg: dict, drop_model_cfg_keys: tuple[str, ...]) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["with_sam"] = False
    for key in drop_model_cfg_keys:
        cfg["model"].pop(key, None)
    return cfg


def validate_exported_checkpoint(checkpoint: dict) -> None:
    from boxernet.boxernet import BoxerNet

    model = BoxerNet(checkpoint["cfg"]["model"])
    expected = model.state_dict()
    actual = checkpoint["model"]

    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    shape_mismatch = [
        (key, tuple(actual[key].shape), tuple(expected[key].shape))
        for key in sorted(expected_keys & actual_keys)
        if tuple(actual[key].shape) != tuple(expected[key].shape)
    ]

    if missing or extra or shape_mismatch:
        lines = ["exported checkpoint does not match BoxerNet.state_dict()"]
        if missing:
            lines.append(f"missing keys ({len(missing)}): {missing[:10]}")
        if extra:
            lines.append(f"extra keys ({len(extra)}): {extra[:10]}")
        if shape_mismatch:
            lines.append(
                f"shape mismatches ({len(shape_mismatch)}): {shape_mismatch[:5]}"
            )
        raise RuntimeError("\n".join(lines))


def main() -> None:
    args = parse_args()
    strip_prefixes = tuple(dict.fromkeys(args.strip_prefix))
    drop_model_cfg_keys = tuple(dict.fromkeys(args.drop_model_cfg_key))

    checkpoint = torch.load(args.input_ckpt, map_location="cpu", weights_only=False)
    model_state = load_model_state(checkpoint)
    stripped_model, removed = strip_model_state(model_state, strip_prefixes)
    exported = {
        "cfg": clean_cfg(checkpoint["cfg"], drop_model_cfg_keys),
        "model": stripped_model,
    }

    if not args.skip_validation:
        validate_exported_checkpoint(exported)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_ckpt)) or ".", exist_ok=True)
    torch.save(exported, args.output_ckpt)

    input_size_mb = os.path.getsize(args.input_ckpt) / 1024 / 1024
    output_size_mb = os.path.getsize(args.output_ckpt) / 1024 / 1024
    print(f"Input checkpoint:  {args.input_ckpt}")
    print(f"Output checkpoint: {args.output_ckpt}")
    print(f"Removed tensors:   {removed}")
    print(f"Kept tensors:      {len(stripped_model)}")
    print(f"Input size (MB):   {input_size_mb:.2f}")
    print(f"Output size (MB):  {output_size_mb:.2f}")


if __name__ == "__main__":
    main()
