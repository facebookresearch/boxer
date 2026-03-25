# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
DETIC wrapper for efficient open-vocabulary object detection with instance segmentation.

This wrapper provides a Python interface to the DETIC model using JIT-traced models,
matching the behavior of the C++ DeticRunner and following the interface pattern of
OwlWrapper and Sam3Wrapper.

Example usage:
    wrapper = DeticWrapper(device="cuda", min_confidence=0.5)
    boxes, scores, labels, masks = wrapper.forward(image_torch, threshold=0.5)
"""

import csv
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import os
from torchvision.ops import nms

# Default model dimensions matching C++ DeticRunnerSettings
DEFAULT_MODEL_WIDTH = 480
DEFAULT_MODEL_HEIGHT = 640
DEFAULT_MIN_CONFIDENCE = 0.5


class DeticWrapper(nn.Module):
    """
    Runs DETIC open-vocabulary object detection on input images using JIT-traced models.

    The wrapper loads pre-traced DETIC models from local paths and provides an interface
    compatible with OwlWrapper and Sam3Wrapper.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_width: int = DEFAULT_MODEL_WIDTH,
        model_height: int = DEFAULT_MODEL_HEIGHT,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        selected_classes: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize DETIC wrapper with JIT model from local path.

        Args:
            device: Device to run the model on ('cuda', 'mps', or 'cpu')
            model_width: Width of the model input (default: 480)
            model_height: Height of the model input (default: 640)
            min_confidence: Minimum confidence threshold for detections (default: 0.5)
            selected_classes: Optional list of class names to filter detections.
                              If None, all classes are returned.
        """
        super().__init__()

        # Validate device
        if device == "cuda":
            assert torch.cuda.is_available(), "CUDA not available"
        elif device == "mps":
            assert torch.backends.mps.is_available(), "MPS not available"
        else:
            device = "cpu"

        self.device = device
        self.model_width = model_width
        self.model_height = model_height
        self.min_confidence = min_confidence
        self.selected_classes = set(selected_classes) if selected_classes else None

        # Determine device string for model path
        device_str = "cuda:0" if device == "cuda" else "cpu"
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "detic", device_str,
        )

        # Load class ID to name mapping
        self.class_id_name_map = self._load_class_mapping()
        print(f"Loaded {len(self.class_id_name_map)} classes")

        if self.selected_classes:
            known_classes = set(self.class_id_name_map.values())
            unknown = self.selected_classes - known_classes
            if unknown:
                print(f"WARNING: {len(unknown)} label(s) not in Detic vocabulary: {sorted(unknown)}")
                print("Consider using --detector owl for open-vocabulary detection.")

        # Load JIT model
        model_path = os.path.join(base_dir, f"detic_{model_width}x{model_height}-1.pt")
        print(f"Loading model from: {model_path}")
        self.model = torch.jit.load(model_path)
        self.model.to(device)
        self.model.eval()
        print(f"Loaded DETIC model on {device}")

        # Warmup the model
        self._warmup_model(warmup_steps=5)

    def _load_class_mapping(self) -> Dict[int, str]:
        """
        Load class ID to name mapping from CSV file bundled in the repo.

        Returns:
            Dictionary mapping class index to class name
        """
        local_classes_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lvis_classes.csv"
        )

        class_map = {}
        with open(local_classes_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_map[int(row["Class Index"])] = row["Class Name"]

        return class_map

    def _warmup_model(self, warmup_steps: int = 5) -> None:
        """
        Warm up the model with dummy inference.

        Args:
            warmup_steps: Number of warmup iterations
        """
        print("Warming up the model")
        with torch.no_grad():
            dummy_input = torch.zeros(
                3, self.model_height, self.model_width, device=self.device
            )
            for i in range(warmup_steps):
                self.model.forward(dummy_input)
                print(f"Model warmup iteration {i} complete")

    @torch.no_grad()
    def forward(
        self,
        image_torch: torch.Tensor,
        rotated: bool = False,
        resize_to_HW: Optional[Tuple[int, int]] = None,
        nms_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
        """
        Run DETIC inference on a single image.

        Args:
            image_torch: Input image tensor of shape (1, C, H, W) in range [0, 255]
            rotated: Whether to rotate image 90 degrees clockwise before processing
            threshold: Confidence threshold for filtering predictions.
                       If None, uses min_confidence from initialization.
            resize_to_HW: Target (height, width) for resizing. If None, uses model dimensions.
            nms_threshold: IoU threshold for non-maximum suppression (default: 0.5)

        Returns:
            Tuple of (boxes, scores, labels, masks):
                - boxes: Bounding boxes in format (x1, x2, y1, y2), shape (N, 4)
                - scores: Confidence scores, shape (N,)
                - labels: Class names as list of strings
                - masks: Binary segmentation masks, shape (N, H, W)
        """
        assert len(image_torch.shape) == 4, "input image should be 4D tensor"
        assert image_torch.shape[0] == 1, "only batch size 1 is supported"
        if (
            image_torch.max() < 1.01
            or image_torch.max() > 255.0
            or image_torch.min() < 0.0
        ):
            print("warning: input image should be in [0, 255] as a float")

        threshold = self.min_confidence

        input_image = image_torch.clone()
        if rotated:
            input_image = torch.rot90(input_image, k=3, dims=(2, 3))  # 90 CW

        HH, WW = input_image.shape[2], input_image.shape[3]
        original_HH, original_WW = HH, WW

        # Determine target model dimensions
        target_height = resize_to_HW[0] if resize_to_HW else self.model_height
        target_width = resize_to_HW[1] if resize_to_HW else self.model_width

        need_resize = (target_width != WW) or (target_height != HH)

        # Convert to CHW format (model expects CHW, not NCHW)
        # Input is (1, C, H, W) -> need (C, H, W)
        input_tensor = input_image.squeeze(0)  # (C, H, W)

        # Permute if needed (model expects CHW with channels first)
        # Image tensor should already be in CHW format, but let's verify
        if input_tensor.shape[0] not in [1, 3]:
            # Likely HWC format, permute to CHW
            input_tensor = input_tensor.permute(2, 0, 1)

        # Handle grayscale by repeating to 3 channels
        if input_tensor.shape[0] == 1:
            input_tensor = input_tensor.repeat(3, 1, 1)

        # Resize if needed
        if need_resize:
            input_tensor = torch.nn.functional.interpolate(
                input_tensor.unsqueeze(0),
                size=(target_height, target_width),
                mode="nearest",
            ).squeeze(0)

        input_tensor = input_tensor.to(self.device)

        # Run model inference
        outputs = self.model.forward(input_tensor)

        # Parse outputs - model returns tuple of (boxes, classes, masks, confidences, ...)
        # Based on C++ code: elements[i*5+0]=box2D, elements[i*5+1]=class, elements[i*5+2]=mask, elements[i*5+3]=conf
        box2d_tensor = outputs[0].cpu()  # (N, 4) in format x1, y1, x2, y2
        class_tensor = outputs[1].cpu()  # (N,) class indices
        mask_tensor = outputs[2].cpu()  # (N, H, W) masks
        conf_tensor = outputs[3].cpu()  # (N,) confidence scores

        empty_return = (
            torch.zeros((0, 4)),
            torch.zeros(0),
            [],
            torch.zeros((0, original_HH, original_WW)),
        )

        if class_tensor.size(0) == 0:
            return empty_return

        # Filter by confidence threshold
        conf_mask = conf_tensor >= threshold
        box2d_tensor = box2d_tensor[conf_mask]
        class_tensor = class_tensor[conf_mask]
        mask_tensor = mask_tensor[conf_mask]
        conf_tensor = conf_tensor[conf_mask]

        if class_tensor.size(0) == 0:
            return empty_return

        # Filter by selected classes if specified
        if self.selected_classes:
            keep_indices = []
            for i in range(class_tensor.size(0)):
                class_idx = class_tensor[i].item()
                class_name = self.class_id_name_map.get(int(class_idx), "unknown")
                if class_name in self.selected_classes:
                    keep_indices.append(i)

            if len(keep_indices) == 0:
                return empty_return

            keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
            box2d_tensor = box2d_tensor[keep_tensor]
            class_tensor = class_tensor[keep_tensor]
            mask_tensor = mask_tensor[keep_tensor]
            conf_tensor = conf_tensor[keep_tensor]

        if class_tensor.size(0) == 0:
            return empty_return

        # Apply NMS
        nms_keep = nms(box2d_tensor.float(), conf_tensor.float(), nms_threshold)
        box2d_tensor = box2d_tensor[nms_keep]
        class_tensor = class_tensor[nms_keep]
        mask_tensor = mask_tensor[nms_keep]
        conf_tensor = conf_tensor[nms_keep]

        if class_tensor.size(0) == 0:
            return empty_return

        # Scale boxes and masks back to original resolution if resized
        if need_resize:
            width_scale = original_WW / target_width
            height_scale = original_HH / target_height

            # Scale boxes
            box2d_tensor[:, 0] *= width_scale  # x1
            box2d_tensor[:, 1] *= height_scale  # y1
            box2d_tensor[:, 2] *= width_scale  # x2
            box2d_tensor[:, 3] *= height_scale  # y2

            # Resize masks back to original resolution
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(1).float(),
                size=(original_HH, original_WW),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            mask_tensor = mask_tensor > 0.5

        # Convert boxes from (x1, y1, x2, y2) to (x1, x2, y1, y2) convention
        new_boxes = []
        for box in box2d_tensor:
            x1, y1, x2, y2 = box
            assert x1 <= x2 and y1 <= y2, f"box coordinates are not valid: {box}"
            new_boxes.append(torch.tensor([x1, x2, y1, y2], device=box.device))
        boxes = torch.stack(new_boxes, dim=0)

        # Handle rotation - rotate boxes and masks back
        if rotated:
            new_boxes = []
            for box in boxes:
                # rotate boxes back by 90 degrees counter-clockwise
                x1, x2, y1, y2 = box
                new_x1 = y1
                new_x2 = y2
                new_y1 = original_WW - x2
                new_y2 = original_WW - x1
                assert new_x1 <= new_x2 and new_y1 <= new_y2, (
                    f"box coordinates are not valid: {box}"
                )
                new_boxes.append(
                    torch.tensor([new_x1, new_x2, new_y1, new_y2], device=box.device)
                )
            boxes = torch.stack(new_boxes, dim=0)

            # Rotate masks back by 90 degrees counter-clockwise
            mask_tensor = torch.rot90(mask_tensor, k=1, dims=(1, 2))

        # Convert class indices to class names
        labels = []
        for i in range(class_tensor.size(0)):
            class_idx = int(class_tensor[i].item())
            class_name = self.class_id_name_map.get(class_idx, f"class_{class_idx}")
            labels.append(class_name)

        # Ensure output tensors are float32
        boxes = boxes.float()
        scores = conf_tensor.float()
        masks = mask_tensor.float()

        return boxes, scores, labels, masks
