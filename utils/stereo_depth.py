import os
import re
import sys

import torch
import yaml

from utils.demo_utils import CKPT_PATH


DEFAULT_FOUNDATION_STEREO_CKPT = (
    "/home/demo/code/projectaria_gen2_depth_from_stereo/FoundationStereo/"
    "pretrained_models/11-33-40/model_best_bp2.pth"
)

FOUNDATION_STEREO_MODEL_CANDIDATES = [
    os.environ.get("BOXER_FS_CKPT", ""),
    os.path.join(CKPT_PATH, "fs_256wh_16it_bf16_dino_costagg_grus_encoder_fp16.engine"),
    os.path.join(
        os.path.dirname(CKPT_PATH),
        "tmp/foundation_stereo_trt_256/tensorrt_fp32.engine",
    ),
    os.path.join(
        os.path.dirname(CKPT_PATH),
        "tmp/foundation_stereo_trt_320/tensorrt_fp32_dynamo.engine",
    ),
    os.path.join(CKPT_PATH, "fs_320wh_16it_bf16_all_convtranspose_fp32.engine"),
    os.path.join(CKPT_PATH, "fs_384wh_12it_bf16_all_convtranspose_fp32.engine"),
    os.path.join(CKPT_PATH, "fs_512wh_8it_bf16_all_convtranspose_fp32.engine"),
]

FS_MODEL_PRESETS = {
    # Standalone benchmark latency with torch.cuda.synchronize(), May 27 2026:
    # preset      impl        hw   iters  mean ms  p50 ms  p90 ms  min ms  max ms
    # f256        foundation 256     16      85.7    77.8   124.3    64.7   168.1
    # f320        foundation 320     16     129.4   123.2   148.5   116.6   161.3
    # fast512ct   fast       512    n/a      98.7    82.1   146.9    71.1   181.9
    # fast512fp32 fast       512    n/a     105.2   101.4   134.0    78.7   173.3
    # f384        foundation 384     12     166.8   142.8   240.1   125.1   258.6
    # f512        foundation 512      8     262.4   233.3   347.7   203.8   374.9
    # ckpts/fast_foundationstereo_512.engine is excluded: it returns all-NaN disparity.
    "bf16": "/home/demo/Downloads/model_best_bp2.pth",
    "f256torchbf16": "/home/demo/Downloads/model_best_bp2.pth",
    "f256": os.path.join(
        CKPT_PATH, "fs_256wh_16it_bf16_dino_costagg_grus_encoder_fp16.engine"
    ),  # parity-checked, ~70 ms
    "f256fp32": os.path.join(
        os.path.dirname(CKPT_PATH),
        "tmp/foundation_stereo_trt_256/tensorrt_fp32.engine",
    ),
    "f320": os.path.join(
        os.path.dirname(CKPT_PATH),
        "tmp/foundation_stereo_trt_320/tensorrt_fp32_dynamo.engine",
    ),  # parity-checked, ~236 ms
    "f320fp32": os.path.join(
        os.path.dirname(CKPT_PATH),
        "tmp/foundation_stereo_trt_320/tensorrt_fp32_dynamo.engine",
    ),
    "f384": os.path.join(CKPT_PATH, "fs_384wh_12it_bf16_all_convtranspose_fp32.engine"),  # ~143 ms
    "f512": os.path.join(CKPT_PATH, "fs_512wh_8it_bf16_all_convtranspose_fp32.engine"),  # ~233 ms
    "fast512": os.path.join(
        CKPT_PATH, "fast_foundationstereo_512_bf16_all_convtranspose_fp32.engine"
    ),  # ~82 ms
    "fast512fp32": os.path.join(CKPT_PATH, "fast_foundationstereo_512_fp32.engine"),  # ~101 ms
    "fast512ct": os.path.join(
        CKPT_PATH, "fast_foundationstereo_512_bf16_all_convtranspose_fp32.engine"
    ),  # ~82 ms
}

FS_MODEL_PRESET_HELP = {
    "bf16": "FoundationStereo 256 PyTorch BF16 autocast (~113 ms)",
    "f256": "FoundationStereo 256 mixed BF16/FP16 TensorRT (parity-checked, ~70 ms)",
    "f256fp32": "FoundationStereo 256 FP32 TensorRT (parity-checked)",
    "f256torchbf16": "FoundationStereo 256 PyTorch BF16 autocast (~113 ms)",
    "f320": "FoundationStereo 320 FP32 TensorRT (parity-checked, ~236 ms)",
    "f320fp32": "FoundationStereo 320 FP32 TensorRT (parity-checked, ~236 ms)",
    "f384": "FoundationStereo 384 TensorRT (~143 ms)",
    "f512": "FoundationStereo 512 TensorRT (~233 ms)",
    "fast512": "Fast-FS 512 BF16 ConvTranspose-FP32 TensorRT (~82 ms)",
    "fast512fp32": "Fast-FS 512 FP32 TensorRT (~101 ms)",
    "fast512ct": "Fast-FS 512 BF16 ConvTranspose-FP32 TensorRT (~82 ms)",
}


def get_autocast_dtype_for_cuda():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def is_tensorrt_engine_path(path: str) -> bool:
    return path.endswith(".engine") or path.endswith(".plan")


def resolve_default_foundation_stereo_model() -> str:
    for path in FOUNDATION_STEREO_MODEL_CANDIDATES:
        if path and os.path.isfile(path):
            return path
    searched = "\n  ".join(
        path for path in FOUNDATION_STEREO_MODEL_CANDIDATES if path
    )
    raise FileNotFoundError(
        "No default FoundationStereo model was found. Pass --fs_ckpt explicitly "
        "or place a model at one of:\n  "
        f"{searched}"
    )


def resolve_fs_model_preset(name: str) -> str:
    try:
        path = FS_MODEL_PRESETS[name]
    except KeyError:
        choices = ", ".join(FS_MODEL_PRESETS)
        raise ValueError(f"Unknown --fsm preset {name!r}. Choices: {choices}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--fsm {name} points to missing model: {path}")
    return path


def infer_fs_hw_from_engine_path(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"fs_(\d+)wh_", base)
    if m is not None:
        return int(m.group(1))
    m = re.search(r"foundation_stereo_trt_(\d+)", os.path.dirname(path))
    if m is not None:
        return int(m.group(1))
    if m is None:
        raise ValueError(
            "Could not infer FoundationStereo resolution from engine filename. "
            "Expected something like fs_256wh_16it_...engine or fs_384wh_12it_...engine, "
            f"got: {base}"
        )


def resolve_fast_fs_config(model_path: str) -> dict:
    model_dir = os.path.dirname(model_path)
    base = os.path.splitext(os.path.basename(model_path))[0]
    candidates = [
        os.path.join(model_dir, f"{base}.yaml"),
        os.path.join(model_dir, "cfg.yaml"),
        os.path.join(model_dir, "config.yaml"),
        os.path.join(model_dir, "onnx.yaml"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    for name in sorted(os.listdir(model_dir)):
        if name.lower().startswith("cfg") and name.lower().endswith(".yaml"):
            p = os.path.join(model_dir, name)
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"No Fast-FoundationStereo yaml config found for {model_path}"
    )


def resolve_fs_hw(model_path: str, fs_impl: str) -> int:
    if fs_impl == "foundation":
        if is_tensorrt_engine_path(model_path):
            return infer_fs_hw_from_engine_path(model_path)
        return 256
    if fs_impl == "fast":
        cfg = resolve_fast_fs_config(model_path)
        image_size = cfg.get("image_size")
        if (
            not isinstance(image_size, (list, tuple))
            or len(image_size) != 2
            or int(image_size[0]) != int(image_size[1])
        ):
            raise ValueError(
                f"Fast-FoundationStereo config must define square image_size, got {image_size}"
            )
        return int(image_size[0])
    raise ValueError(f"Unsupported fs_impl: {fs_impl}")


def infer_fs_impl_from_model_path(model_path: str) -> str:
    base = os.path.basename(model_path).lower()
    if "fast_foundationstereo" in base or base.startswith("fast_"):
        return "fast"
    return "foundation"


def ensure_projectaria_fs_repo_on_path() -> None:
    fs_repo = "/home/demo/code/projectaria_gen2_depth_from_stereo"
    foundation_path = os.path.join(fs_repo, "FoundationStereo")
    if fs_repo not in sys.path:
        sys.path.insert(0, fs_repo)
    if foundation_path not in sys.path:
        sys.path.insert(0, foundation_path)


def tensorrt_dtype_to_torch(dtype):
    import tensorrt as trt

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.bfloat16: torch.bfloat16,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]
