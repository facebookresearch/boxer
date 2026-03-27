# pyre-unsafe
import io
import os
import time

import torch
import torch.nn.functional as F
from owl.clip_tokenizer import CLIPTokenizer
from utils.taxonomy import load_text_labels

DEFAULT_TEXT_LABELS = load_text_labels("lvisplus")


class VisionDetectorWrapper(torch.nn.Module):
    """Wraps OWLv2 vision encoder + detection heads.

    Takes pixel_values + pre-computed text embeddings, returns logits and pred_boxes.

    NOTE: We inline the vision model sub-components (embeddings, pre_layernorm,
    encoder) instead of calling vision_model.forward() directly. This avoids
    HuggingFace's internal `pixel_values.to(expected_input_dtype)` cast getting
    baked into the JIT trace as a hardcoded float32 cast, which would prevent
    bfloat16 from working at inference time via torch.autocast.
    """

    def __init__(self, model):
        super().__init__()
        self.embeddings = model.owlv2.vision_model.embeddings
        self.pre_layernorm = model.owlv2.vision_model.pre_layernorm
        self.encoder = model.owlv2.vision_model.encoder
        self.post_layernorm = model.owlv2.vision_model.post_layernorm
        self.layer_norm = model.layer_norm
        self.class_head = model.class_head
        self.box_head = model.box_head
        self.sigmoid = model.sigmoid
        # Pre-computed box bias for native resolution (no interpolation needed)
        self.register_buffer("box_bias", model.box_bias.clone())

    def forward(
        self,
        pixel_values: torch.Tensor,
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Vision encoder (inlined to avoid baked-in dtype cast)
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        last_hidden_state = encoder_outputs[0]

        # Post layernorm
        image_embeds = self.post_layernorm(last_hidden_state)

        # Merge CLS token with patch tokens
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # image_feats: [batch, num_patches, hidden_dim]
        image_feats = image_embeds

        # query_embeds: [num_queries, embed_dim] -> [1, num_queries, embed_dim]
        query_embeds_batched = query_embeds.unsqueeze(0)
        query_mask_batched = query_mask.unsqueeze(0)

        # Class prediction
        pred_logits, _ = self.class_head(image_feats, query_embeds_batched, query_mask_batched)

        # Box prediction
        pred_boxes = self.box_head(image_feats)
        pred_boxes = pred_boxes + self.box_bias
        pred_boxes = self.sigmoid(pred_boxes)

        return pred_logits, pred_boxes

_CKPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpts", "owlv2-base-patch16-ensemble.pt")


class OwlWrapper(torch.nn.Module):
    """
    Runs OWLv2 open-set 2D BB detector.
    No transformers dependency at runtime.

    Uses JIT-traced vision detector for float32, or nn.Module with explicit
    bfloat16 casting for ~2x speedup on CUDA (autocast doesn't penetrate
    JIT traced sub-modules).

    Text embeddings are computed once at init and cached.
    Use set_text_prompts() to change prompts without re-creating the wrapper.
    """

    def __init__(self, device="cuda", text_prompts=None, min_confidence=0.2, precision="float32", warmup=True):
        super().__init__()
        _debug = os.environ.get("DEBUG", "0") == "1"
        _t0 = time.perf_counter()
        _tp = _t0

        def _dbg(label):
            nonlocal _tp
            if not _debug:
                return
            now = time.perf_counter()
            print(f"  [owl] {label}: {(now - _tp)*1000:.0f}ms (total: {(now - _t0)*1000:.0f}ms)", flush=True)
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
        self.text_encoder = torch.jit.load(io.BytesIO(checkpoint["text_encoder"]), map_location="cpu")
        self.text_encoder.eval()
        _dbg("jit text_encoder")

        self.device = device
        self.text_prompts = text_prompts
        self.min_confidence = min_confidence
        self.use_bfloat16 = (precision == "bfloat16" and device not in ("cpu", "mps"))

        # Load vision detector: nn.Module for bfloat16 (explicit casting),
        # JIT trace for float32 (no benefit from bfloat16 through JIT).
        if self.use_bfloat16 and "vision_detector_state_dict" in checkpoint:
            # Build nn.Module shell and load weights from state_dict
            # (no HuggingFace dependency — weights come from checkpoint)
            self.vision_detector = _load_vision_module(
                checkpoint["vision_detector_state_dict"], config, device
            )
            self.vision_detector.to(dtype=torch.bfloat16, device=device)
            self.vision_detector.eval()
        else:
            vis_map_loc = "cpu" if device == "mps" else device
            self.vision_detector = torch.jit.load(
                io.BytesIO(checkpoint["vision_detector"]), map_location=vis_map_loc
            )
            self.vision_detector.eval()
            if device == "mps":
                self.vision_detector = self.vision_detector.to(device)
        _dbg("jit vision_detector")

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
            _dbg(f"load cached text embeddings ({len(text_prompts)} prompts)")
        else:
            self.text_embeddings = self._encode_text(text_prompts)
            torch.save({"prompts": text_prompts, "embeddings": self.text_embeddings.cpu()}, cache_path)
            self.text_embeddings = self.text_embeddings.to(device)
            _dbg(f"encode_text ({len(text_prompts)} prompts) + save cache")
        self.query_mask = torch.ones(len(text_prompts), dtype=torch.bool, device=device)

        print(f"Loaded OWLv2 on {device} with {len(text_prompts)} text prompts, precision={'bfloat16' if self.use_bfloat16 else 'float32'}")

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
        assert len(image_torch.shape) == 4, "input image should be 4D tensor"
        assert image_torch.shape[0] == 1, "only batch size 1 is supported"
        if (
            image_torch.max() < 1.01
            or image_torch.max() > 255.0
            or image_torch.min() < 0.0
        ):
            print("warning: input image should be in [0, 255] as a float")

        input_image = image_torch.clone()
        if rotated:
            input_image = torch.rot90(input_image, k=3, dims=(2, 3))  # 90 CW
        HH, WW = input_image.shape[2], input_image.shape[3]

        # Preprocess: resize to native model resolution, normalize
        interp_mode = "bilinear" if self.device == "mps" else "bicubic"
        pixel_values = F.interpolate(
            input_image, size=self.native_size, mode=interp_mode, align_corners=False,
        )
        pixel_values = pixel_values / 255.0
        mean = self.image_mean.to(pixel_values.device)
        std = self.image_std.to(pixel_values.device)
        pixel_values = (pixel_values - mean) / std
        pixel_values = pixel_values.to(self.device)
        if self.use_bfloat16:
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

        # Forward pass (vision + detection)
        logits, pred_boxes = self.vision_detector(
            pixel_values, self.text_embeddings, self.query_mask,
        )

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

        return boxes, scores, labels, None


def _load_vision_module(state_dict, config, device):
    """Load VisionDetectorWrapper as nn.Module from state_dict without HuggingFace.

    Reconstructs the model architecture from the HuggingFace config stored
    in the checkpoint, loads the saved weights, and returns a ready-to-use module.
    """
    from transformers import Owlv2Config, Owlv2ForObjectDetection

    # Use the pretrained config to get correct image_size (960 vs default 768)
    owl_config = Owlv2Config.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection(owl_config).eval()
    wrapper = VisionDetectorWrapper(model)
    wrapper.load_state_dict(state_dict)
    return wrapper
