# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os

import numpy as np
import torch
from torch import nn

from boxernet.dinov3_wrapper import (
    DINOV3_OUTPUT_DIM,
    DinoV3Wrapper,
    batch_dino,
)
from utils.gravity import gravity_align_T_world_cam
from utils.tw.camera import CameraTW
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW, rotation_from_euler

# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, y):
        # Pre-norm (apply norm before attention) as in https://arxiv.org/abs/2002.04745.
        x = self.norm(x)
        y = self.norm(y)
        # Compute Q,K,V matrices.
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)
        # Split into separate heads.
        B, N_q, _ = q.shape
        N_kv = k.shape[1]
        q = q.view(B, N_q, self.heads, -1).transpose(1, 2)
        k = k.view(B, N_kv, self.heads, -1).transpose(1, 2)
        v = v.view(B, N_kv, self.heads, -1).transpose(1, 2)
        # Compute softmax(Q*K^T / sqrt(D))*V
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        # Combine the multi-headed output.
        out = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.to_out(out)


class AttentionBlockV2(nn.Module):
    def __init__(self, dim=256, depth=6, heads=4, mlp_mult=4):
        """Vanilla transformer implementation.
        Inputs:
          dim - dimensionality of attention and mlp (shared for simplicity)
          depth - number of layers
          heads - number of attention heads
        """
        super().__init__()
        assert dim % heads == 0
        dim_head = int(dim // heads)
        dim_mlp = dim * mlp_mult
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                        ),
                        FeedForward(dim, dim_mlp),
                    ]
                )
            )

    def forward(self, x, y=None):
        """If y not provided this is self-attention; otherwise is cross-attention.

        For cross-attention, the queries should be independent, so changing the value of
        one query should not affect the other queries.

        Inputs:
            x: BxNxD batch of query tokens, D is dimenionality of each token (maps to Q)
            y: optional BxMxD batch input tokens, D is dim of each token (maps to K, V)
        Outputs:
            out: BxNxD tensor of transformed queries, same shape as x

        """
        if y is None:
            y = x  # Self-attention.

        for attn, ff in self.layers:
            x = attn(x, y) + x
            x = ff(x) + x
        return x


# ---------------------------------------------------------------------------
# Prediction Head -- Predicts 7 DoF params + Aleotoric Uncertainty per box
# ---------------------------------------------------------------------------


class AleHead(torch.nn.Module):
    """Aleatoric uncertainty head. Predicts 3D bounding boxes."""

    def __init__(
        self, in_dim, out_dim=7, hidden_dim=128, norm_chamfer=False, min_dim=0.05
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bbox_min = 0.02
        self.bbox_max = 4.0
        self.min_dim = min_dim
        self.norm_chamfer = norm_chamfer
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.logvar_head = nn.Linear(hidden_dim, 1)  # predicts log sigma^2

    def forward(self, batch, output):
        query = output["query"]

        h = self.net(query)
        mu = self.mean_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, -10, +3)

        params_v = mu

        B, M = params_v.shape[0], params_v.shape[1]
        device = params_v.device
        pr = params_v
        yaw_max = np.pi / 2

        # Directly predict X,Y,Z in voxel coords.
        center_v = pr[..., :3]

        # Build size
        pr[..., 3:6] = (
            torch.sigmoid(pr[..., 3:6]) * (self.bbox_max - self.bbox_min)
            + self.bbox_min
        )
        hh, ww, dd = pr[:, :, 3], pr[:, :, 4], pr[:, :, 5]
        if self.min_dim > 0:
            hh = hh.clamp(min=self.min_dim)
            ww = ww.clamp(min=self.min_dim)
            dd = dd.clamp(min=self.min_dim)
        bb3 = torch.zeros((B, M, 6)).to(device)
        bb3[:, :, 0] = -(hh / 2)
        bb3[:, :, 1] = hh / 2
        bb3[:, :, 2] = -(ww / 2)
        bb3[:, :, 3] = ww / 2
        bb3[:, :, 4] = -(dd / 2)
        bb3[:, :, 5] = dd / 2

        # Build T_world_object
        pr[..., 6] = yaw_max * torch.tanh(pr[..., 6])
        yaw = pr[:, :, 6]
        zeros = torch.zeros_like(yaw).to(device)
        e_angles = torch.stack([zeros, zeros, yaw], dim=-1)
        R = rotation_from_euler(e_angles.reshape(-1, 3))
        R = R.reshape(B, M, 3, 3)
        T_vo = PoseTW.from_Rt(R, center_v)

        # Build final ObbTW.
        sigma2 = torch.exp(logvar)
        prob = 1.0 / (1.0 + sigma2)

        inst_id = torch.arange(M).reshape(1, M, -1).repeat(B, 1, 1).to(device)
        sem_id = 32 + torch.zeros((B, M, 1)).to(device)
        obb_pr_v = ObbTW.from_lmc(
            bb3_object=bb3,
            T_world_object=T_vo,
            prob=prob,
            inst_id=inst_id,
            sem_id=sem_id,
        )
        T_wv = batch["T_world_voxel0"].unsqueeze(1)
        obb_pr_w = obb_pr_v.transform(T_wv)
        output["obbs_pr_w"] = obb_pr_w
        output["obbs_pr_params"] = params_v
        output["obbs_pr_logvar"] = logvar
        return batch, output


def image_to_patches(x, patch_size=14):
    """
    Args:
        x: (B, 1, H, W) image tensor
    Returns:
        patches: (B, N, patch_size*patch_size), where N = num_patches
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        "H and W must be divisible by patch_size"
    )

    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # shape: (B, 1, H//p, W//p, p, p)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.view(B, -1, patch_size * patch_size)  # (B, N, p*p)

    return patches


def masked_median(x: torch.Tensor, mask: torch.Tensor, dim: int):
    # Replace masked-out values with NaN
    x_masked = x.masked_fill(~mask, float("nan"))
    # Sort along the dimension, keeping NaNs at the end
    sorted_x, _ = torch.sort(x_masked, dim=dim)

    # Count valid (non-NaN) elements along dim
    valid_counts = mask.sum(dim=dim)

    # Get index of the median for each entry
    median_indices = (valid_counts - 1) // 2  # floor for even counts

    # Gather median values using advanced indexing
    # Only supports dim=1 in this example, but can be generalized
    if dim != 1:
        raise NotImplementedError("This version only supports dim=1")

    batch_indices = torch.arange(x.size(0), device=x.device)
    medians = sorted_x[batch_indices, median_indices]

    return medians


def sdp_to_patches(sdp_w, cam, T_wr, H, W, patch_size):
    """
    Convert semi-dense world points to a patch-based depth representation.

    Args:
        sdp_w: (B, N, 3) world points
        cam: CameraTW camera intrinsics
        T_wr: PoseTW world-to-rig transform
        H, W: image dimensions
        patch_size: size of each patch (e.g., 16 for DINO)

    Returns:
        (B, 1, fH, fW) median depth per patch, where fH=H//patch_size, fW=W//patch_size
        Invalid patches have value -1
    """
    assert sdp_w.ndim == 3, f"Expected 3D (BxNx3) sdp_w, got {sdp_w.ndim}D tensor"
    B, N, _ = sdp_w.shape
    device = sdp_w.device

    # Sneaky Bug Fix!!! Center world points around camera position to avoid float32
    # precision loss. T_wr.t can have large values (e.g., z~278), and sdp_w is nearby,
    # so the subtraction in T_cw * sdp_w loses precision in float32 (especially on GPU).
    center = T_wr.t.clone()
    if center.ndim == 1:
        center = center.unsqueeze(0)
    sdp_w = sdp_w - center.unsqueeze(-2)  # (B, N, 3)
    T_wr_data = T_wr._data.clone()
    T_wr_data[..., -3:] = 0.0  # zero out translation
    T_wr = PoseTW(T_wr_data)

    # Transform world points to camera coordinates and project to image
    T_cw = cam.T_camera_rig @ T_wr.inverse()
    sdp_c = T_cw * sdp_w
    sdp_uv, valid = cam.project(sdp_c)

    # Get pixel coordinates and depth values
    uu = sdp_uv[..., 0].round().long().clamp(0, W - 1)
    vv = sdp_uv[..., 1].round().long().clamp(0, H - 1)
    depth_vals = sdp_c[..., 2].clone()

    # Build valid mask: must be valid projection with positive depth
    valid_mask = valid & (depth_vals > 0)

    # Flatten for scatter operation
    bb = torch.arange(B, device=device).reshape(B, 1).expand(B, N)
    flat_idx = (bb * H * W + vv * W + uu).reshape(-1)
    depth_flat = depth_vals.reshape(-1)
    valid_flat = valid_mask.reshape(-1)

    # Filter to only valid points
    valid_idx = flat_idx[valid_flat]
    valid_depth = depth_flat[valid_flat]

    # Scatter depths and counts to image grid
    # Use index_add_ instead of scatter_reduce_ for MPS compatibility
    sdp_img_flat = torch.zeros(B * H * W, device=device)
    count_flat = torch.zeros(B * H * W, device=device)

    sdp_img_flat.index_add_(0, valid_idx, valid_depth)
    count_flat.index_add_(0, valid_idx, torch.ones_like(valid_depth))

    # Compute mean depth per pixel
    sdp_img = sdp_img_flat.reshape(B, 1, H, W)
    count_img = count_flat.reshape(B, 1, H, W)
    sdp_img = torch.where(count_img > 0, sdp_img / count_img, torch.zeros_like(sdp_img))

    # Convert to patches and compute median per patch
    sdp_patches = image_to_patches(
        sdp_img, patch_size
    )  # (B, num_patches, patch_size^2)
    mask = sdp_patches > 0.0

    fH, fW = H // patch_size, W // patch_size
    num_patches = fH * fW

    sdp_median = masked_median(
        sdp_patches.reshape(B * num_patches, -1),
        mask.reshape(B * num_patches, -1),
        dim=1,
    )
    sdp_median = sdp_median.reshape(B, 1, fH, fW)
    sdp_median[torch.isnan(sdp_median)] = -1

    return sdp_median


def generate_patch_centers(B, fH, fW, patch_size, device):
    """
    Generate coordinates (d, m) for rays in a batch of posed images.

    Args:
        B : int
            Batch size (number of images)
        fH, fW : int
            Height and width of the image grid
        patch_size : int
            Size of patches
        device : str
            "cpu" or "cuda"
    Returns:
        uv : (B, fH*fW, 2) tensor

    """
    # get patch centers in pixel coords.
    yy, xx = torch.meshgrid(
        torch.arange(0, fH * patch_size, step=patch_size, device=device),
        torch.arange(0, fW * patch_size, step=patch_size, device=device),
        indexing="ij",
    )
    xx, yy = xx + patch_size / 2, yy + patch_size / 2
    uv = torch.stack([xx, yy], dim=-1)  # (fW, fH, 2)
    uv = torch.reshape(uv, (fH * fW, 2))  # (fH*fW, 2)
    # repeat for batch
    uv = uv.unsqueeze(0).expand(B, fH * fW, 2).contiguous()  # (B, fH*fW, 2)
    return uv


def generate_plucker_encoding(B, fH, fW, patch_size, cam, T_vc):
    """
    Generate Plücker line encodings (d, m) for rays in a batch of posed images.

    Args:
        B : int
            Batch size (number of images)
        fH, fW : int
            Height and width of the image grid
        patch_size : int
            Size of patches
        cam : CameraTW
            Camera intrinsics
        T_vc : PoseTW
            voxel-to-camera transform
    Returns:
        plucker_voxel : (B, fH*fW, 6) tensor
            Plücker line representations (d, m) in voxel coordinates,
            where d = direction, m = origin × direction.
    """
    device = cam.device

    # (B, fH*fW, 2)
    uv = generate_patch_centers(B, fH, fW, patch_size, device)

    # Unproject rays to get direction vectors in camera frame
    dirs_cam, valid = cam.unproject(uv)  # (B, fH*fW, 3)

    # Ray origin in camera frame (typically [0, 0, 0])
    origins_cam = torch.zeros_like(dirs_cam)

    # Transform to voxel frame
    dirs_voxel = T_vc.rotate(dirs_cam)  # (B, fH*fW, 3)

    # Normalize.
    dirs_voxel = dirs_voxel / (dirs_voxel.norm(dim=-1, keepdim=True) + 1e-8)

    origins_voxel = T_vc * origins_cam  # (B, fH*fW, 3)

    # Compute moment: m = o × d
    m_voxel = torch.cross(origins_voxel, dirs_voxel, dim=-1)  # (B, fH*fW, 3)

    # Stack to get Plücker 6D line encoding: (d, m)
    plucker_voxel = torch.cat([dirs_voxel, m_voxel], dim=-1)  # (B, fH*fW, 6)

    # set invalid rays to zero
    plucker_voxel[~valid] = 0.0
    # set any nans to zero
    plucker_voxel[torch.isnan(plucker_voxel)] = 0.0

    return plucker_voxel


def smart_load(model_dict, ckpt_dict):
    # Strip _orig_mod. prefix from checkpoint keys (added by torch.compile)
    cleaned_ckpt_dict = {}
    for key, val in ckpt_dict.items():
        clean_key = key.replace("_orig_mod.", "")
        cleaned_ckpt_dict[clean_key] = val
    if len(cleaned_ckpt_dict) != len(ckpt_dict):
        print("  Warning: key collisions after stripping _orig_mod. prefix")
    ckpt_dict = cleaned_ckpt_dict

    new_ckpt_dict = {}
    loaded_count = 0
    loaded_params = 0
    shape_mismatch_count = 0
    shape_mismatch_params = 0
    shape_mismatch_keys = []
    missing_count = 0
    missing_params = 0
    missing_keys = []
    for key in ckpt_dict:
        if key not in model_dict:
            missing_count += 1
            missing_params += ckpt_dict[key].numel()
            missing_keys.append(key)
            continue
        ckpt_shape = ckpt_dict[key].shape
        model_shape = model_dict[key].shape
        if ckpt_shape != model_shape:
            shape_mismatch_count += 1
            shape_mismatch_params += ckpt_dict[key].numel()
            shape_mismatch_keys.append((key, ckpt_shape, model_shape))
            continue
        new_ckpt_dict[key] = ckpt_dict[key]
        loaded_count += 1
        loaded_params += ckpt_dict[key].numel()
    # Compute model params not loaded.
    not_loaded_count = 0
    not_loaded_params = 0
    not_loaded_keys = []
    for key in model_dict:
        if key not in new_ckpt_dict:
            not_loaded_count += 1
            not_loaded_params += model_dict[key].numel()
            not_loaded_keys.append(key)
    # Print summary.
    total_ckpt = len(ckpt_dict)
    total_model = len(model_dict)
    total_ckpt_params = sum(p.numel() for p in ckpt_dict.values())
    total_model_params = sum(p.numel() for p in model_dict.values())

    def fmt(n):
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.2f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    load_pct = (
        100.0 * loaded_params / total_model_params if total_model_params > 0 else 0
    )
    has_issues = shape_mismatch_count > 0 or missing_count > 0 or not_loaded_count > 0
    print(
        f"==> Loaded checkpoint: {loaded_count}/{total_model} tensors, "
        f"{fmt(loaded_params)}/{fmt(total_model_params)} params ({load_pct:.1f}%)"
    )
    if has_issues:
        if shape_mismatch_count > 0:
            print(
                f"  Shape mismatch: {shape_mismatch_count} tensors, {fmt(shape_mismatch_params)} params"
            )
            for key, cs, ms in shape_mismatch_keys[:20]:
                print(f"    {key}: ckpt={list(cs)} model={list(ms)}")
        if missing_count > 0:
            print(
                f"  Not in model: {missing_count} tensors, {fmt(missing_params)} params"
            )
            for key in missing_keys[:20]:
                print(f"    {key}: {list(ckpt_dict[key].shape)}")
        if not_loaded_count > 0:
            print(
                f"  Not from ckpt: {not_loaded_count} tensors, {fmt(not_loaded_params)} params"
            )
            for key in not_loaded_keys[:20]:
                print(f"    {key}: {list(model_dict[key].shape)}")
    return new_ckpt_dict


class BoxerNet(nn.Module):
    """
    2D box prompt on top of image + semidense points.
    """

    default_cfg = {
        "dim": 768,
        "in_depth": 4,
        "cross_depth": 6,
    }
    base_default_cfg = {"trainable": True, "combine_images": True}

    def __init__(self, cfg):
        super().__init__()
        if not isinstance(cfg, dict):
            cfg = dict(cfg)
        merged = {**self.base_default_cfg, **self.default_cfg, **cfg}
        self.cfg = merged
        cfg = merged

        self.dim = cfg["dim"]
        self.heads = self.dim // 64
        self.in_depth = cfg["in_depth"]
        self.cross_depth = cfg["cross_depth"]
        self.with_ray = cfg.get("with_ray", False)
        assert self.cross_depth >= 1, "cross_depth must be at least 1"

        dinov3_model_name = "dinov3_vits16plus"
        self.dino = DinoV3Wrapper(dinov3_model_name)
        if torch.cuda.is_available():
            self.dino = self.dino.cuda()
        for param in self.dino.parameters():
            param.requires_grad = False
        self.patch_size = 16
        dino_dim = DINOV3_OUTPUT_DIM[dinov3_model_name]

        self.in_dim = dino_dim  # dino feature
        self.in_dim += 1  # median per patch depth
        if self.with_ray:
            self.in_dim += 6  # ray direction

        self.query_dim = 4  # (xmin, xmin, ymin, ymax)

        self.head = AleHead(self.dim, 7, norm_chamfer=cfg["norm_chamfer"])

        self.input2emb = torch.nn.Linear(self.in_dim, self.dim)
        self.query2emb = torch.nn.Linear(self.query_dim, self.dim)
        if self.in_depth > 0:
            self.self_attn = AttentionBlockV2(self.dim, self.in_depth, self.heads)
        self.cross_attn = AttentionBlockV2(self.dim, self.cross_depth, self.heads)

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, device="cuda"):
        if device == "cuda":
            assert torch.cuda.is_available()
        elif device == "mps":
            assert torch.backends.mps.is_available()
        else:
            device = "cpu"

        ckpt_path = os.path.expanduser(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise IOError(f'Cannot find checkpoint file "{ckpt_path}"')

        print(f'Loading checkpoint from "{ckpt_path}"')
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg = ckpt["cfg"]
        hw = cfg["dataset"]["image_hw"]

        model = cls(cfg["model"])
        model_dict = model.state_dict()
        new_ckpt_dict = smart_load(model_dict, ckpt["model"])
        model.load_state_dict(new_ckpt_dict, strict=False)

        model = model.to(device)
        model = model.eval()
        model.hw = hw
        model.device = device
        return model

    def process_camera(self, datum):
        img = datum["img0"]
        cam = datum["cam0"]
        T_wr = datum["T_world_rig0"]
        rotated = datum["rotated0"]
        assert img.max() <= 1.0, "input image should be in [0,1]"
        if img.ndim == 3:
            img = img.unsqueeze(0)
        cam_data = cam._data
        T_wr_data = T_wr._data
        if cam_data.ndim == 1:
            cam_data = cam_data.unsqueeze(0)
        assert T_wr_data.shape[-1] == 12
        if T_wr_data.ndim == 1:
            T_wr_data = T_wr_data.unsqueeze(0)

        assert isinstance(rotated, torch.Tensor)
        assert rotated.ndim == 1
        out = {}
        out["img0"] = img
        out["cam0"] = CameraTW(cam_data)
        out["T_world_rig0"] = PoseTW(T_wr_data)
        out["rotated0"] = rotated
        return out

    def prepare_inputs(self, datum):
        inputs = {}
        inputs.update(self.process_camera(datum))

        sdp_w = datum["sdp_w"]
        assert sdp_w.shape[1] == 3, "sdp_w should be Nx3"
        if sdp_w.ndim == 2:
            sdp_w = sdp_w.unsqueeze(0)
        inputs["sdp_w_padded"] = sdp_w

        bb2d = datum["bb2d"]
        assert bb2d.shape[-1] == 4
        if bb2d.ndim == 2:
            bb2d = bb2d.unsqueeze(0)
        inputs["bb2d"] = bb2d

        for key in inputs:
            if "rotated" in key:
                inputs[key] = inputs[key].to(self.device).bool()
            else:
                inputs[key] = inputs[key].to(self.device).float()

        return inputs

    def encode(self, batch):
        out = {}

        torch_img = batch["img0"]
        cam = batch["cam0"]
        T_wr = batch["T_world_rig0"]
        T_wc = T_wr @ cam.T_camera_rig.inverse()
        if "T_world_voxel0" not in batch:
            T_wv = gravity_align_T_world_cam(T_wc, z_grav=True)
            rpy = T_wv.to_euler()  # T_wv should only have yaw value
            assert torch.allclose(
                torch.zeros(1, device=rpy.device), rpy[:, :2], atol=1e-4
            )
            batch["T_world_voxel0"] = T_wv
        T_wv = batch["T_world_voxel0"]
        rotated = batch["rotated0"]
        B, _, H, W = torch_img.shape

        with torch.no_grad():
            # Run DinoV3.
            dino_feat = batch_dino(self.dino, torch_img, rotated)
            out["dino0"] = dino_feat.clone()
            _, _, fH, fW = dino_feat.shape
            dino_feat = dino_feat.reshape(B, -1, fH * fW).permute(0, 2, 1)

            # Render semi-dense points as patches.
            sdp_w = batch["sdp_w_padded"]
            sdp_median = sdp_to_patches(sdp_w, cam, T_wr, H, W, self.dino.patch_size)
            out["sdp_patch0"] = sdp_median.clone()
            sdp_input = sdp_median.reshape(B, -1, fH * fW).permute(0, 2, 1)
            x = torch.cat([dino_feat, sdp_input], dim=-1)

            # Get ray directions in voxel frame.
            if self.with_ray:
                T_vc = T_wv.inverse() @ T_wc
                # print a loud warning if T_vc.t is not close to zero (loose tolerance).
                # This is because we define the gravity aligned coordinate frame to have
                # a (0,0,0) translation relative to the camera center.
                if not torch.allclose(T_vc.t, torch.zeros_like(T_vc.t), atol=1e-2):
                    print(
                        f"WARNING: T_vc.t is not close to zero: {T_vc.t}. This is not expected."
                    )
                ray_enc = generate_plucker_encoding(
                    B, fH, fW, self.dino.patch_size, cam, T_vc
                )
                out["ray_enc0"] = ray_enc.clone()
                x = torch.cat([x, ray_enc], dim=-1)

        input_enc = self.input2emb(x)

        if self.in_depth > 0:
            input_enc = self.self_attn(input_enc)

        out["input_enc"] = input_enc

        return out

    def query(self, batch, output):
        # Get all 3D GT boxes.
        cam = batch["cam0"]
        img = batch["img0"]
        bb2d = batch["bb2d"]
        B = img.shape[0]
        H, W = img.shape[-2:]
        M = bb2d.shape[1]
        device = bb2d.device
        input_enc = output["input_enc"]

        if "obbs_valid" not in batch:
            batch["obbs_valid"] = torch.ones((B, M, 1), dtype=torch.bool, device=device)

        cH = int(cam[0].size[0].round().int())
        cW = int(cam[0].size[1].round().int())
        assert cW == W and cH == H

        with torch.no_grad():
            # normalize 2DBB corners to [0, 1]
            query_tokens = bb2d.clone()
            query_tokens[..., :2] = (query_tokens[..., :2] + 0.5) / W
            query_tokens[..., 2:] = (query_tokens[..., 2:] + 0.5) / H

        # Use query tokens to attend to input tokens.
        query = self.query2emb(query_tokens)
        query = self.cross_attn(query, input_enc)

        # Run head.
        output["query"] = query
        batch, output = self.head(batch, output)

        return output

    @torch.no_grad()
    def forward(self, datum):
        inputs = self.prepare_inputs(datum)
        out = self.encode(inputs)
        out = self.query(inputs, out)
        return out
