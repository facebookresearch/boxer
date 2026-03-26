# BoxerNet

Given a single RGB frame with camera intrinsics, an egomotion pose, and sparse depth, BoxerNet predicts a full 7-DoF 3D bounding box (position, size, yaw) for each 2D detection.

## Architecture

```
                        ┌─────────────┐
   RGB Image ──────────►│   DINOv3    │──► patch features (fH×fW × 384)
                        └─────────────┘          │
                                                 ├─ concat ──► input tokens
                        ┌─────────────┐          │
   Semi-Dense Points ──►│ SDP Patches │──► patch depths  (fH×fW × 1)
                        └─────────────┘

   input tokens ──► [Self-Attention × N] ──► input encoding

   2D Boxes (xmin, xmax, ymin, ymax) ──► query tokens
                                              │
                          ┌───────────────────┘
                          ▼
               [Cross-Attention × M]  (queries attend to input encoding)
                          │
                          ▼
                     [AleHead MLP]  ──► 7-DoF OBB (dx, dy, dz, w, h, d, yaw)
                                       + aleatoric uncertainty (log σ²)
```

**BoxerNet** (`boxernet.py`) is a transformer that:
1. **Encodes** the scene: DINOv3 visual features + semi-dense depth patches are projected to a shared embedding space, then refined with self-attention.
2. **Queries** per detection: each 2D bounding box becomes a query token that cross-attends to the scene encoding.
3. **Predicts** 3D boxes: an MLP head outputs a 7-DoF oriented bounding box (center offset, dimensions, yaw) plus an aleatoric uncertainty estimate.

## Files

| File | Description |
|------|-------------|
| `boxernet.py` | BoxerNet model (encode → cross-attend → predict) |
| `alehead.py` | AleHead: 7-DoF OBB + uncertainty prediction head |
| `attention_utils.py` | Transformer blocks (self/cross-attention) |
| `dinov3_wrapper.py` | DINOv3 backbone wrapper |
