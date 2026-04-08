# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Pure PyTorch reimplementation of OWLv2 vision encoder + detection heads.

Matches HuggingFace state_dict keys so we can load pretrained weights
without importing transformers at runtime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    """CLIP-style GELU approximation: x * sigmoid(1.702 * x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, num_positions: int):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.patch_embedding = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.position_embedding = nn.Embedding(num_positions, hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(num_positions).unsqueeze(0),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype

        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [B, num_patches, D]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, C = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(
        self, hidden_size: int, num_layers: int, num_heads: int, intermediate_size: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Owlv2ClassPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, query_dim: int):
        super().__init__()
        self.dense0 = nn.Linear(hidden_size, query_dim)
        self.logit_shift = nn.Linear(hidden_size, 1)
        self.logit_scale = nn.Linear(hidden_size, 1)
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.Tensor,
        query_embeds: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_class_embeds = self.dense0(image_embeds)
        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )
        query_embeds = query_embeds / (
            torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        )

        # Dot product: [B, patches, query_dim] x [B, queries, query_dim] -> [B, patches, queries]
        pred_logits = torch.einsum(
            "...pd,...qd->...pq", image_class_embeds, query_embeds
        )

        # Shift and scale operate on original image_embeds (hidden_size dim)
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = query_mask.unsqueeze(-2)
            pred_logits = torch.where(
                query_mask == 0,
                torch.finfo(pred_logits.dtype).min,
                pred_logits,
            )

        return pred_logits, image_class_embeds


class Owlv2BoxPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int = 4):
        super().__init__()
        self.dense0 = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(hidden_size, out_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        out = self.gelu(self.dense0(image_embeds))
        out = self.gelu(self.dense1(out))
        return self.dense2(out)
