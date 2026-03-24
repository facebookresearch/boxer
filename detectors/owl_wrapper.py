# pyre-unsafe
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from utils.taxonomy import load_text_labels

DEFAULT_TEXT_LABELS = load_text_labels("lvisplus")


class OwlWrapper(torch.nn.Module):
    """
    runs Owl open set 2D BB detector on input images. this assumes the text prompts are
    setup during initialization and only images are passed during forward() since computing
    text embeddings is slow.
    """

    def __init__(self, device="cuda", text_prompts=None, min_confidence=0.2):
        super().__init__()

        if text_prompts is None:
            text_prompts = DEFAULT_TEXT_LABELS

        if device == "cuda":
            assert torch.cuda.is_available()
        elif device == "mps":
            assert torch.backends.mps.is_available()
        else:
            device = "cpu"

        # model_name = "owlv2-base-patch16" # should be faster but I don't see a speedup.
        model_name = "owlv2-base-patch16-ensemble"

        model_name = f"google/{model_name}"

        processor = Owlv2Processor.from_pretrained(model_name, use_fast=True)
        model = Owlv2ForObjectDetection.from_pretrained(model_name).eval()

        model.to(device)
        model = model.eval()
        print(f"Loaded Owl model {model_name} on {device}")

        self.model = model
        self.device = device
        self.text_prompts = text_prompts
        self.processor = processor
        self.model_name = model_name
        self.min_confidence = min_confidence

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

        # Default size of 906x906. Doesn't seem to work well on much smaller images.
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlv2/image_processing_owlv2.py#L264C62-L264C65
        size = {"height": resize_to_HW[0], "width": resize_to_HW[1]}

        # with avg_timer(f"OwlWrapper: processor() HxW: {HH}x{WW}"):
        inputs = self.processor(
            text=self.text_prompts,
            images=input_image,
            return_tensors="pt",
            size=size,
        ).to(self.device)

        forward_HW = inputs["pixel_values"].shape[2:]
        # with avg_timer(f"OwlWrapper: forward() HxW: {forward_HW[0]}x{forward_HW[1]}"):
        inputs["interpolate_pos_encoding"] = True
        # MPS doesn't support bicubic interpolation (used by OWLv2 position
        # encoding), so monkey-patch to use bilinear during the forward pass.
        if self.device == "mps":
            import torch.nn.functional as F

            _orig_interpolate = F.interpolate

            def _mps_interpolate(*args, **kwargs):
                if kwargs.get("mode") == "bicubic":
                    kwargs["mode"] = "bilinear"
                return _orig_interpolate(*args, **kwargs)

            F.interpolate = _mps_interpolate
            try:
                outputs = self.model(**inputs)
            finally:
                F.interpolate = _orig_interpolate
        else:
            outputs = self.model(**inputs)

        # with avg_timer("OwlWrapper: post-process"):
        target_sizes = torch.tensor([(HH, WW)])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.min_confidence,
            text_labels=[self.text_prompts],
        )
        result = results[0]

        boxes = result["boxes"].cpu()
        scores = result["scores"].cpu()
        labels = result["labels"].cpu()

        empty_return = torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0), None

        if len(boxes) == 0:
            return empty_return

        # Filter out very large or small boxes. These are often false positives.
        xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        too_big = (xmax - xmin > 0.9 * WW) | (ymax - ymin > 0.9 * HH)
        too_small = (xmax - xmin < 0.05 * WW) | (ymax - ymin < 0.05 * HH)
        keep = ~too_big & ~too_small
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(boxes) == 0:
            return empty_return

        # Flip x1, y1, x2, y2 to x1, x2, y1, y2 convention.
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            assert x1 <= x2 and y1 <= y2, f"box coordinates are not valid: {box}"
            new_boxes.append(torch.tensor([x1, x2, y1, y2], device=box.device))
        boxes = torch.stack(new_boxes, dim=0)

        if rotated:
            new_boxes = []
            for box in boxes:
                # rotate boxes back by 90 degrees counter-clockwise
                x1, x2, y1, y2 = box
                new_x1 = y1
                new_x2 = y2
                new_y1 = WW - x2
                new_y2 = WW - x1
                assert new_x1 <= new_x2 and new_y1 <= new_y2, (
                    f"box coordinates are not valid: {box}"
                )
                new_boxes.append(
                    torch.tensor([new_x1, new_x2, new_y1, new_y2], device=box.device)
                )
            boxes = torch.stack(new_boxes, dim=0)

        return boxes, scores, labels, None
