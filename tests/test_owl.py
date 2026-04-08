#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Tests that the traced OWLv2 wrapper produces outputs matching the
original transformers-based pipeline.

Requires: pip install transformers  (only for this test)
"""

import unittest

import torch

from owl.clip_tokenizer import CLIPTokenizer
from owl.owl_wrapper import OwlWrapper
from utils.taxonomy import load_text_labels


def _has_transformers():
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Shared fixtures: one OwlWrapper + one forward() call for all tests.
# --------------------------------------------------------------------------

_shared_wrapper = None
_shared_forward_result = None
_shared_forward_img = None


def _get_wrapper():
    global _shared_wrapper
    if _shared_wrapper is None:
        _shared_wrapper = OwlWrapper(
            "cpu",
            text_prompts=["cat", "dog"],
            min_confidence=0.001,
            warmup=False,
        )
    return _shared_wrapper


def _get_forward_result():
    global _shared_forward_result, _shared_forward_img
    if _shared_forward_result is None:
        wrapper = _get_wrapper()
        torch.manual_seed(123)
        _shared_forward_img = torch.rand(1, 3, 480, 640) * 255
        _shared_forward_result = wrapper.forward(_shared_forward_img)
    return _shared_forward_img, _shared_forward_result


@unittest.skipUnless(_has_transformers(), "transformers not installed")
class TestOwlMatchesTransformers(unittest.TestCase):
    """Compare traced OWLv2 outputs against the original transformers model."""

    @classmethod
    def setUpClass(cls):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        cls.model_name = "google/owlv2-base-patch16-ensemble"
        cls.processor = Owlv2Processor.from_pretrained(cls.model_name, use_fast=True)
        cls.hf_model = Owlv2ForObjectDetection.from_pretrained(cls.model_name).eval()
        cls.wrapper = _get_wrapper()

    def test_tokenizer_matches(self):
        """Verify our CLIP tokenizer matches the transformers tokenizer."""
        ref_tok = self.processor.tokenizer
        my_tok = self.wrapper.tokenizer

        labels = load_text_labels("lvisplus")
        ref = ref_tok(
            labels,
            return_tensors="pt",
            padding="max_length",
            max_length=16,
            truncation=True,
        )
        mine = my_tok(labels)

        self.assertTrue(
            (ref["input_ids"] == mine["input_ids"]).all(),
            "Tokenizer input_ids mismatch",
        )
        self.assertTrue(
            (ref["attention_mask"] == mine["attention_mask"]).all(),
            "Tokenizer attention_mask mismatch",
        )

    def test_text_embeddings_match(self):
        """Verify traced text encoder produces same embeddings as transformers."""
        prompts = ["a photo of a cat", "chair", "table"]
        ref_tok = self.processor.tokenizer
        ref_inputs = ref_tok(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=16,
            truncation=True,
        )

        with torch.no_grad():
            text_out = self.hf_model.owlv2.text_model(
                input_ids=ref_inputs["input_ids"],
                attention_mask=ref_inputs["attention_mask"],
            )
            ref_embeds = self.hf_model.owlv2.text_projection(text_out[1])

        # Temporarily set prompts to match, then restore
        self.wrapper.set_text_prompts(prompts)
        our_embeds = self.wrapper.text_embeddings
        self.wrapper.set_text_prompts(["cat", "dog"])

        max_diff = (ref_embeds - our_embeds).abs().max().item()
        self.assertLess(
            max_diff, 1e-4, f"Text embedding max diff {max_diff:.2e} exceeds tolerance"
        )

    def test_vision_logits_match(self):
        """Verify traced vision detector logits match transformers on same preprocessed input."""
        prompts = ["cat", "dog"]
        img, _ = _get_forward_result()

        inputs = self.processor(
            text=prompts,
            images=img,
            return_tensors="pt",
            size={"height": 960, "width": 960},
        )
        inputs["interpolate_pos_encoding"] = False
        with torch.no_grad():
            hf_out = self.hf_model(**inputs)

        with torch.no_grad():
            traced_logits, traced_boxes = self.wrapper.vision_detector(
                inputs["pixel_values"],
                self.wrapper.text_embeddings,
                self.wrapper.query_mask,
            )

        logits_diff = (hf_out.logits - traced_logits).abs().max().item()
        boxes_diff = (hf_out.pred_boxes - traced_boxes).abs().max().item()

        self.assertLess(logits_diff, 1e-4, f"Logits max diff {logits_diff:.2e}")
        self.assertLess(boxes_diff, 1e-4, f"Boxes max diff {boxes_diff:.2e}")

    def test_detection_output_format(self):
        """Verify forward() returns correct types and shapes."""
        _, (boxes, scores, labels, masks) = _get_forward_result()

        self.assertIsInstance(boxes, torch.Tensor)
        self.assertIsInstance(scores, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertIsNone(masks)
        self.assertEqual(boxes.ndim, 2)
        self.assertEqual(boxes.shape[1], 4)
        self.assertEqual(scores.shape[0], boxes.shape[0])
        self.assertEqual(labels.shape[0], boxes.shape[0])

    def test_set_text_prompts(self):
        """Verify text prompts can be changed dynamically."""
        wrapper = self.wrapper
        self.assertEqual(wrapper.text_embeddings.shape[0], 2)

        wrapper.set_text_prompts(["cat", "dog", "bird"])
        self.assertEqual(wrapper.text_embeddings.shape[0], 3)
        self.assertEqual(len(wrapper.text_prompts), 3)
        self.assertEqual(wrapper.query_mask.shape[0], 3)

        # Restore original prompts
        wrapper.set_text_prompts(["cat", "dog"])

    def test_rotated_input(self):
        """Verify rotated input doesn't crash and returns valid output."""
        img, _ = _get_forward_result()
        boxes, scores, labels, masks = self.wrapper.forward(img, rotated=True)

        self.assertIsInstance(boxes, torch.Tensor)
        self.assertEqual(boxes.shape[1], 4) if len(boxes) > 0 else None

    def test_box_coordinate_convention(self):
        """Verify boxes use x1, x2, y1, y2 convention (not x1, y1, x2, y2)."""
        _, (boxes, scores, labels, _) = _get_forward_result()

        if len(boxes) > 0:
            x1, x2, y1, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            self.assertTrue((x1 <= x2).all(), "x1 > x2 in some boxes")
            self.assertTrue((y1 <= y2).all(), "y1 > y2 in some boxes")


class TestOwlWrapperStandalone(unittest.TestCase):
    """Tests that don't require transformers."""

    def test_no_transformers_imported(self):
        """Verify the wrapper doesn't import transformers at runtime."""
        import sys

        mods_before = set(sys.modules.keys())
        _ = _get_wrapper()
        mods_after = set(sys.modules.keys())
        new_mods = mods_after - mods_before
        self.assertNotIn("transformers", new_mods)

    def test_missing_models_error(self):
        """Verify helpful error when traced models are missing."""
        import owl.owl_wrapper as ow

        original = ow._CKPT_PATH
        ow._CKPT_PATH = "/nonexistent/path/model.pt"
        try:
            with self.assertRaises(FileNotFoundError) as ctx:
                OwlWrapper("cpu", text_prompts=["test"])
            self.assertIn("README", str(ctx.exception))
        finally:
            ow._CKPT_PATH = original


class TestTextEmbedder(unittest.TestCase):
    """Tests for the CLIP text embedder."""

    @classmethod
    def setUpClass(cls):
        from owl.clip_tokenizer import TextEmbedder

        cls.embedder = TextEmbedder()

    def test_output_shape(self):
        """Verify output shape is (N, 512)."""
        embeds = self.embedder.forward(["chair", "table"])
        self.assertEqual(embeds.shape, (2, 512))

    def test_normalized(self):
        """Verify embeddings are L2-normalized."""
        embeds = self.embedder.forward(["chair", "table", "lamp"])
        norms = embeds.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_semantic_similarity(self):
        """Verify semantically similar labels have higher cosine similarity."""
        embeds = self.embedder.forward(["sofa", "couch", "refrigerator"])
        sim = embeds @ embeds.T
        self.assertGreater(sim[0, 1].item(), sim[0, 2].item())

    def test_deterministic(self):
        """Verify same input gives same output."""
        e1 = self.embedder.forward(["chair"])
        e2 = self.embedder.forward(["chair"])
        self.assertTrue(torch.allclose(e1, e2))

    def test_matches_owl_text_encoder(self):
        """Verify TextEmbedder produces same embeddings as OWL checkpoint's text encoder."""
        import io
        import os

        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ckpts",
            "owlv2-base-patch16-ensemble.pt",
        )
        if not os.path.exists(ckpt_path):
            self.skipTest("OWL checkpoint not found")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        text_encoder = torch.jit.load(
            io.BytesIO(checkpoint["text_encoder"]), map_location="cpu"
        )
        text_encoder.eval()
        tokenizer = CLIPTokenizer(
            vocab=checkpoint["tokenizer_vocab"],
            merges=checkpoint["tokenizer_merges"],
            max_length=checkpoint["config"]["max_seq_length"],
        )

        prompts = ["chair", "table", "lamp", "a photo of a dog"]
        tokens = tokenizer(prompts)
        with torch.no_grad():
            owl_embeds = text_encoder(tokens["input_ids"], tokens["attention_mask"])
        owl_embeds = torch.nn.functional.normalize(owl_embeds, dim=-1)
        our_embeds = self.embedder.forward(prompts)
        max_diff = (owl_embeds - our_embeds).abs().max().item()
        self.assertLess(max_diff, 1e-5, f"Embedding max diff {max_diff:.2e}")


if __name__ == "__main__":
    unittest.main()
