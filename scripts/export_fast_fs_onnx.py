#!/usr/bin/env python3
"""Export Fast-FoundationStereo as a single ONNX model via legacy exporter."""

import argparse
import logging
import os
import sys

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf


FAST_FS_REPO = "/home/demo/code/Fast-FoundationStereo"
if FAST_FS_REPO not in sys.path:
    sys.path.insert(0, FAST_FS_REPO)

import core.foundation_stereo as _fs_module


def _build_gwc_volume_onnx(refimg_fea, targetimg_fea, maxdisp, num_groups, normalize=True):
    dtype = refimg_fea.dtype
    bsz, channels, height, width = refimg_fea.shape
    channels_per_group = channels // num_groups
    ref_volume = refimg_fea.unsqueeze(2).expand(bsz, channels, maxdisp, height, width)
    shifted = [
        F.pad(targetimg_fea, (d, 0, 0, 0), "constant", 0.0)[:, :, :, :width]
        for d in range(maxdisp)
    ]
    target_volume = torch.stack(shifted, dim=2)
    ref_volume = ref_volume.view(
        bsz, num_groups, channels_per_group, maxdisp, height, width
    )
    target_volume = target_volume.view(
        bsz, num_groups, channels_per_group, maxdisp, height, width
    )
    if normalize:
        ref_volume = F.normalize(ref_volume.float(), dim=2).to(dtype)
        target_volume = F.normalize(target_volume.float(), dim=2).to(dtype)
    return (ref_volume * target_volume).sum(dim=2).contiguous()


def _build_concat_volume_onnx(refimg_fea, targetimg_fea, maxdisp):
    bsz, channels, height, width = refimg_fea.shape
    ref_volume = refimg_fea.unsqueeze(2).expand(bsz, channels, maxdisp, height, width)
    shifted = [
        F.pad(targetimg_fea, (d, 0, 0, 0), "constant", 0.0)[:, :, :, :width]
        for d in range(maxdisp)
    ]
    target_volume = torch.stack(shifted, dim=2)
    return torch.cat((ref_volume, target_volume), dim=1).contiguous()


class FastFoundationStereoSingleOnnx(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, left_image, right_image):
        return self.model.forward(
            left_image,
            right_image,
            iters=self.model.args.valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", required=True, help="Path to serialized Fast FS .pth")
    p.add_argument("--save_path", required=True, help="Directory for ONNX and yaml")
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--valid_iters", type=int, default=8)
    p.add_argument("--max_disp", type=int, default=192)
    p.add_argument("--onnx_name", type=str, default="fast_foundationstereo")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assert args.height % 32 == 0 and args.width % 32 == 0
    os.makedirs(args.save_path, exist_ok=True)
    torch.autograd.set_grad_enabled(False)

    logging.info("Loading model from %s", args.model_dir)
    model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.args.mixed_precision = False
    model.cuda().eval()

    wrapper = FastFoundationStereoSingleOnnx(model).cuda().eval()
    left_img = torch.randn(1, 3, args.height, args.width, device="cuda")
    right_img = torch.randn(1, 3, args.height, args.width, device="cuda")

    onnx_name = (
        args.onnx_name if args.onnx_name.endswith(".onnx") else f"{args.onnx_name}.onnx"
    )
    onnx_path = os.path.join(args.save_path, onnx_name)
    logging.info("Exporting ONNX (%sx%s) -> %s", args.height, args.width, onnx_path)

    _fs_module.normalize_image = lambda img: img
    _fs_module.build_gwc_volume_optimized_pytorch1 = _build_gwc_volume_onnx
    _fs_module.build_concat_volume_optimized_pytorch1 = _build_concat_volume_onnx

    torch.onnx.export(
        wrapper,
        (left_img, right_img),
        onnx_path,
        opset_version=17,
        input_names=["left_image", "right_image"],
        output_names=["disparity"],
        do_constant_folding=True,
        dynamo=False,
    )

    cfg = OmegaConf.to_container(model.args)
    cfg["image_size"] = [args.height, args.width]
    config_name = os.path.splitext(onnx_name)[0] + ".yaml"
    config_path = os.path.join(args.save_path, config_name)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    logging.info("ONNX model  : %s", onnx_path)
    logging.info("Config      : %s", config_path)


if __name__ == "__main__":
    main()
