#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import torch
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--opset", type=int, default=22)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = "/home/demo/code/projectaria_gen2_depth_from_stereo/FoundationStereo"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from core.foundation_stereo import FoundationStereo

    class FoundationStereoOnnx(FoundationStereo):
        @torch.no_grad()
        def forward(self, left, right):
            with torch.amp.autocast("cuda", enabled=True):
                return FoundationStereo.forward(
                    self, left, right, iters=self.args.valid_iters, test_mode=True
                )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    cfg_path = os.path.join(os.path.dirname(args.ckpt_path), "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg.save_path = args.save_path
    cfg.ckpt_dir = args.ckpt_path
    cfg.height = int(args.height)
    cfg.width = int(args.width)
    cfg.valid_iters = int(args.valid_iters)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    logging.warning("args:\n%s", cfg)
    logging.warning("Using pretrained model from %s", args.ckpt_path)

    model = FoundationStereoOnnx(cfg)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()

    left_img = torch.randn(1, 3, cfg.height, cfg.width, device="cuda").float()
    right_img = torch.randn(1, 3, cfg.height, cfg.width, device="cuda").float()

    torch.onnx.export(
        model,
        (left_img, right_img),
        args.save_path,
        dynamo=False,
        opset_version=int(args.opset),
        input_names=["left", "right"],
        output_names=["disp"],
        dynamic_axes={
            "left": {0: "batch_size"},
            "right": {0: "batch_size"},
            "disp": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()
