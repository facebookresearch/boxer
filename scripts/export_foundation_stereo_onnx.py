#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--opset", type=int, default=22)
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use the newer torch.export-based ONNX exporter.",
    )
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="Export ONNX with dynamic batch axis instead of a fixed batch size.",
    )
    parser.add_argument(
        "--no_constant_folding",
        action="store_true",
        help="Disable ONNX constant folding to reduce peak export memory.",
    )
    parser.add_argument(
        "--fp32_export",
        action="store_true",
        help="Disable model autocast while exporting the ONNX graph.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = "/home/demo/code/projectaria_gen2_depth_from_stereo/FoundationStereo"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import core.foundation_stereo as foundation_stereo_mod
    import core.geometry as geometry_mod
    import core.utils.utils as utils_mod

    def normalize_image_for_export(img):
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return ((img / 255.0 - mean) / std).contiguous()

    foundation_stereo_mod.normalize_image = normalize_image_for_export

    def bilinear_sampler_for_export(img, coords, mode="bilinear", mask=False, low_memory=False):
        h, w = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (w - 1) - 1
        grid = torch.cat([xgrid, ygrid], dim=-1).to(img.dtype)
        img = torch.nn.functional.grid_sample(
            img.contiguous(), grid.contiguous(), align_corners=True, mode=mode
        )
        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()
        return img

    utils_mod.bilinear_sampler = bilinear_sampler_for_export
    geometry_mod.bilinear_sampler = bilinear_sampler_for_export
    FoundationStereo = foundation_stereo_mod.FoundationStereo

    class FoundationStereoOnnx(FoundationStereo):
        @torch.no_grad()
        def forward(self, left, right):
            with torch.amp.autocast("cuda", enabled=not bool(args.fp32_export)):
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
    if args.fp32_export:
        cfg.mixed_precision = False
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    logging.warning("args:\n%s", cfg)
    logging.warning("Using pretrained model from %s", args.ckpt_path)

    model = FoundationStereoOnnx(cfg)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    class AdaptiveMaxPool2dOneForExport(nn.Module):
        def forward(self, x):
            return torch.amax(x, dim=(-2, -1), keepdim=True)

    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.AdaptiveMaxPool2d) and child.output_size == 1:
                setattr(module, name, AdaptiveMaxPool2dOneForExport())

    model.cuda().eval()

    batch_size = int(args.batch_size)
    left_img = torch.randn(batch_size, 3, cfg.height, cfg.width, device="cuda").float()
    right_img = torch.randn(batch_size, 3, cfg.height, cfg.width, device="cuda").float()

    export_kwargs = {}
    if args.dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            "left": {0: "batch_size"},
            "right": {0: "batch_size"},
            "disp": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        (left_img, right_img),
        args.save_path,
        dynamo=bool(args.dynamo),
        opset_version=int(args.opset),
        input_names=["left", "right"],
        output_names=["disp"],
        do_constant_folding=not bool(args.no_constant_folding),
        external_data=True,
        **export_kwargs,
    )


if __name__ == "__main__":
    main()
