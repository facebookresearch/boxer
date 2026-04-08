# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import io
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from owl.clip_tokenizer import CLIPTokenizer
from owl.owlv2_model import (
    CLIPEncoder,
    CLIPVisionEmbeddings,
    Owlv2BoxPredictionHead,
    Owlv2ClassPredictionHead,
)
from utils.taxonomy import load_text_labels

DEFAULT_TEXT_LABELS = load_text_labels("lvisplus")


class VisionDetectorWrapper(nn.Module):
    """Wraps OWLv2 vision encoder + detection heads.

    Takes pixel_values + pre-computed text embeddings, returns logits and pred_boxes.
    Pure PyTorch — no HuggingFace dependency.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        patch_size: int,
        num_positions: int,
        query_dim: int,
    ):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(hidden_size, patch_size, num_positions)
        self.pre_layernorm = nn.LayerNorm(hidden_size)
        self.encoder = CLIPEncoder(
            hidden_size, num_layers, num_heads, intermediate_size
        )
        self.post_layernorm = nn.LayerNorm(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.class_head = Owlv2ClassPredictionHead(hidden_size, query_dim)
        self.box_head = Owlv2BoxPredictionHead(hidden_size)
        self.sigmoid = nn.Sigmoid()
        num_patches = num_positions - 1  # exclude CLS token
        self.register_buffer("box_bias", torch.zeros(num_patches, 4))

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor]
    ) -> "VisionDetectorWrapper":
        """Build model from state_dict, inferring architecture from weight shapes."""
        hidden_size = state_dict["embeddings.patch_embedding.weight"].shape[0]
        patch_size = state_dict["embeddings.patch_embedding.weight"].shape[2]
        num_positions = state_dict["embeddings.position_embedding.weight"].shape[0]
        intermediate_size = state_dict["encoder.layers.0.mlp.fc1.weight"].shape[0]
        num_heads = hidden_size // 64  # standard CLIP head dim
        num_layers = (
            max(
                int(k.split(".")[2])
                for k in state_dict
                if k.startswith("encoder.layers.")
            )
            + 1
        )
        query_dim = state_dict["class_head.dense0.weight"].shape[0]
        model = cls(
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            patch_size,
            num_positions,
            query_dim,
        )
        model.load_state_dict(state_dict)
        return model

    def forward(
        self,
        pixel_values: torch.Tensor,
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Vision encoder
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)
        hidden_states = self.encoder(hidden_states)

        # Post layernorm
        image_embeds = self.post_layernorm(hidden_states)

        # Merge CLS token with patch tokens
        class_token_out = torch.broadcast_to(
            image_embeds[:, :1, :], image_embeds[:, :-1].shape
        )
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # image_feats: [batch, num_patches, hidden_dim]
        image_feats = image_embeds

        # query_embeds: [num_queries, embed_dim] -> [1, num_queries, embed_dim]
        query_embeds_batched = query_embeds.unsqueeze(0)
        query_mask_batched = query_mask.unsqueeze(0)

        # Class prediction
        pred_logits, _ = self.class_head(
            image_feats, query_embeds_batched, query_mask_batched
        )

        # Box prediction
        pred_boxes = self.box_head(image_feats)
        pred_boxes = pred_boxes + self.box_bias
        pred_boxes = self.sigmoid(pred_boxes)

        return pred_logits, pred_boxes


_CKPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ckpts",
    "owlv2-base-patch16-ensemble.pt",
)


def _per_class_nms(boxes, scores, labels, iou_threshold):
    """Per-class greedy NMS. boxes: (N,4) x1y1x2y2, scores/labels: (N,)."""
    keep = []
    for cls in labels.unique():
        cls_mask = labels == cls
        cls_idx = cls_mask.nonzero(as_tuple=True)[0]
        cls_scores = scores[cls_idx]
        cls_boxes = boxes[cls_idx]
        # Sort by score descending
        order = cls_scores.argsort(descending=True)
        cls_idx = cls_idx[order]
        cls_boxes = cls_boxes[order]
        # Greedy suppress
        suppressed = torch.zeros(len(cls_idx), dtype=torch.bool)
        for i in range(len(cls_idx)):
            if suppressed[i]:
                continue
            keep.append(cls_idx[i].item())
            # IoU of box i with all subsequent boxes
            ix1 = torch.max(cls_boxes[i, 0], cls_boxes[i + 1 :, 0])
            iy1 = torch.max(cls_boxes[i, 1], cls_boxes[i + 1 :, 1])
            ix2 = torch.min(cls_boxes[i, 2], cls_boxes[i + 1 :, 2])
            iy2 = torch.min(cls_boxes[i, 3], cls_boxes[i + 1 :, 3])
            inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
            area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (
                cls_boxes[i, 3] - cls_boxes[i, 1]
            )
            area_j = (cls_boxes[i + 1 :, 2] - cls_boxes[i + 1 :, 0]) * (
                cls_boxes[i + 1 :, 3] - cls_boxes[i + 1 :, 1]
            )
            iou = inter / (area_i + area_j - inter + 1e-6)
            suppressed[i + 1 :] |= iou > iou_threshold
    keep.sort()
    return keep


class OwlWrapper(nn.Module):
    """Runs OWLv2 open-set 2D BB detector.

    No transformers dependency. Pure PyTorch nn.Module for the vision detector,
    with bfloat16 support via explicit casting (same pattern as BoxerNet).

    Text embeddings are computed once at init and cached.
    Use set_text_prompts() to change prompts without re-creating the wrapper.
    """

    def __init__(
        self,
        device="cuda",
        text_prompts=None,
        min_confidence=0.2,
        precision="float32",
        warmup=True,
        nms_iou_threshold=0.5,
    ):
        super().__init__()
        _debug = os.environ.get("DEBUG", "0") == "1"
        _t0 = time.perf_counter()
        _tp = _t0

        def _dbg(label):
            nonlocal _tp
            if not _debug:
                return
            now = time.perf_counter()
            print(
                f"  [owl] {label}: {(now - _tp) * 1000:.0f}ms (total: {(now - _t0) * 1000:.0f}ms)",
                flush=True,
            )
            _tp = now

        if text_prompts is None:
            text_prompts = DEFAULT_TEXT_LABELS

        if device == "cuda":
            assert torch.cuda.is_available()
        elif device == "mps":
            assert torch.backends.mps.is_available()
        else:
            device = "cpu"

        # Load combined checkpoint
        if not os.path.exists(_CKPT_PATH):
            raise FileNotFoundError(
                f"OWLv2 checkpoint not found at {_CKPT_PATH}. "
                "See README for export instructions."
            )
        checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
        _dbg("torch.load (881MB)")

        config = checkpoint["config"]
        self.image_mean = torch.tensor(config["image_mean"]).view(1, 3, 1, 1)
        self.image_std = torch.tensor(config["image_std"]).view(1, 3, 1, 1)
        self.native_size = tuple(config["image_size"])  # (960, 960)

        # Load traced text encoder (always on CPU, runs once at init)
        self.text_encoder = torch.jit.load(
            io.BytesIO(checkpoint["text_encoder"]), map_location="cpu"
        )
        self.text_encoder.eval()
        _dbg("jit text_encoder")

        self.device = device
        self.text_prompts = text_prompts
        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold
        if precision is not None:
            self.use_bfloat16 = precision == "bfloat16" and device not in ("cpu", "mps")
        else:
            self.use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()

        # Load vision detector as pure nn.Module from state_dict
        self.vision_detector = VisionDetectorWrapper.from_state_dict(
            checkpoint["vision_detector_state_dict"]
        )
        if self.use_bfloat16:
            self.vision_detector.to(dtype=torch.bfloat16, device=device)
        else:
            self.vision_detector.to(device=device)
        self.vision_detector.eval()
        if device == "cuda":
            self.vision_detector = torch.compile(self.vision_detector)
        _dbg("vision_detector")

        # Load tokenizer from checkpoint data
        self.tokenizer = CLIPTokenizer(
            vocab=checkpoint["tokenizer_vocab"],
            merges=checkpoint["tokenizer_merges"],
            max_length=config["max_seq_length"],
        )
        _dbg("tokenizer")

        # Pre-compute and cache text embeddings (cached to disk by prompt hash)
        import hashlib

        prompt_hash = hashlib.md5("\n".join(text_prompts).encode()).hexdigest()[:12]
        cache_path = _CKPT_PATH.replace(".pt", f"_textemb_{prompt_hash}.pt")
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location=device, weights_only=False)
            if isinstance(cached, dict) and "embeddings" in cached:
                self.text_embeddings = cached["embeddings"]
            else:
                # Old format: raw tensor
                self.text_embeddings = cached
            if self.use_bfloat16:
                self.text_embeddings = self.text_embeddings.to(dtype=torch.bfloat16)
            _dbg(f"load cached text embeddings ({len(text_prompts)} prompts)")
        else:
            self.text_embeddings = self._encode_text(text_prompts)
            torch.save(
                {"prompts": text_prompts, "embeddings": self.text_embeddings.cpu()},
                cache_path,
            )
            self.text_embeddings = self.text_embeddings.to(device)
            _dbg(f"encode_text ({len(text_prompts)} prompts) + save cache")
        self.query_mask = torch.ones(len(text_prompts), dtype=torch.bool, device=device)

        print(
            f"Loaded OWLv2 on {device} with {len(text_prompts)} text prompts, precision={'bfloat16' if self.use_bfloat16 else 'float32'}"
        )

        # Warmup
        if warmup:
            self._warmup()
            _dbg("warmup")

    def _encode_text(self, prompts):
        """Tokenize and encode text prompts through the traced text encoder (runs on CPU)."""
        tokens = self.tokenizer(prompts)
        input_ids = tokens["input_ids"]  # CPU
        attention_mask = tokens["attention_mask"]  # CPU
        with torch.no_grad():
            embeds = self.text_encoder(input_ids, attention_mask)
        embeds = embeds.to(self.device)
        if self.use_bfloat16:
            embeds = embeds.to(dtype=torch.bfloat16)
        return embeds

    def set_text_prompts(self, prompts):
        """Update text prompts and re-compute cached embeddings."""
        self.text_prompts = prompts
        self.text_embeddings = self._encode_text(prompts)
        self.query_mask = torch.ones(len(prompts), dtype=torch.bool, device=self.device)

    def _warmup(self, steps=1):
        """Warmup the vision model with dummy inference."""
        H, W = self.native_size
        dummy = torch.zeros(1, 3, H, W, device=self.device)
        if self.use_bfloat16:
            dummy = dummy.to(dtype=torch.bfloat16)
        with torch.no_grad():
            for _ in range(steps):
                self.vision_detector(dummy, self.text_embeddings, self.query_mask)

    @torch.no_grad()
    def forward(self, image_torch, rotated=False, resize_to_HW=(906, 906)):
        _debug = os.environ.get("DEBUG", "0") == "1"
        assert len(image_torch.shape) == 4, "input image should be 4D tensor"
        assert image_torch.shape[0] == 1, "only batch size 1 is supported"
        if (
            image_torch.max() < 1.01
            or image_torch.max() > 255.0
            or image_torch.min() < 0.0
        ):
            print("warning: input image should be in [0, 255] as a float")

        if _debug:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        input_image = image_torch.clone()
        if rotated:
            input_image = torch.rot90(input_image, k=3, dims=(2, 3))  # 90 CW
        HH, WW = input_image.shape[2], input_image.shape[3]

        # Preprocess: resize to native model resolution, normalize
        interp_mode = "bilinear" if self.device == "mps" else "bicubic"
        pixel_values = F.interpolate(
            input_image,
            size=self.native_size,
            mode=interp_mode,
            align_corners=False,
        )
        pixel_values = pixel_values / 255.0
        mean = self.image_mean.to(pixel_values.device)
        std = self.image_std.to(pixel_values.device)
        pixel_values = (pixel_values - mean) / std
        pixel_values = pixel_values.to(self.device)
        if self.use_bfloat16:
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

        if _debug:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()

        # Forward pass (vision + detection)
        logits, pred_boxes = self.vision_detector(
            pixel_values,
            self.text_embeddings,
            self.query_mask,
        )

        if _debug:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()

        # Postprocess in float32 for numerical stability
        logits = logits.float()
        pred_boxes = pred_boxes.float()

        # Postprocess: sigmoid, threshold, convert boxes
        scores_all, labels_all = torch.max(logits[0], dim=-1)  # [num_patches]
        scores_all = torch.sigmoid(scores_all)

        keep = scores_all > self.min_confidence
        scores = scores_all[keep].cpu()
        labels = labels_all[keep].cpu()
        boxes_cxcywh = pred_boxes[0, keep]  # [N, 4] normalized cxcywh

        empty_return = torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0), None

        if len(boxes_cxcywh) == 0:
            return empty_return

        # Convert cxcywh -> xyxy, scale to original image size
        cx, cy, w, h = boxes_cxcywh.unbind(-1)
        x1 = (cx - w / 2) * WW
        y1 = (cy - h / 2) * HH
        x2 = (cx + w / 2) * WW
        y2 = (cy + h / 2) * HH
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).cpu()

        # Filter out very large or small boxes (false positives)
        too_big = (x2 - x1 > 0.9 * WW) | (y2 - y1 > 0.9 * HH)
        too_small = (x2 - x1 < 0.05 * WW) | (y2 - y1 < 0.05 * HH)
        keep = ~(too_big | too_small).cpu()
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(boxes) == 0:
            return empty_return

        # Per-class NMS
        if self.nms_iou_threshold < 1.0:
            keep = _per_class_nms(boxes, scores, labels, self.nms_iou_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        if len(boxes) == 0:
            return empty_return

        # Convert x1, y1, x2, y2 -> x1, x2, y1, y2 convention
        boxes = boxes[:, [0, 2, 1, 3]]

        if rotated:
            # Rotate boxes back by 90 degrees counter-clockwise
            x1, x2, y1, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            new_x1 = y1
            new_x2 = y2
            new_y1 = WW - x2
            new_y2 = WW - x1
            boxes = torch.stack([new_x1, new_x2, new_y1, new_y2], dim=-1)

        if _debug:
            _t3 = time.perf_counter()
            print(
                f"  [owl fwd] preprocess: {(_t1 - _t0) * 1000:.1f}ms  "
                f"model: {(_t2 - _t1) * 1000:.1f}ms  "
                f"postprocess: {(_t3 - _t2) * 1000:.1f}ms  "
                f"total: {(_t3 - _t0) * 1000:.1f}ms",
                flush=True,
            )

        return boxes, scores, labels, None
