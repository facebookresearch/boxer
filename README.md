# Boxer

Boxer lifts 2D object detections into 3D oriented bounding boxes (OBBs) from posed images and semi-dense point clouds. Given a single RGB frame with camera intrinsics, an egomotion pose, and sparse depth, BoxerNet predicts a full 7-DoF 3D bounding box (position, size, yaw) for each 2D detection.

## Architecture

```
                        ┌─────────────┐
   RGB Image ──────────►│   DINOv3     │──► patch features (fH×fW × 384)
                        └─────────────┘          │
                                                 ├─ concat ──► input tokens
                        ┌─────────────┐          │
   Semi-Dense Points ──►│  SDP→Patches │──► patch depths  (fH×fW × 1)
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

**BoxerNet** (`boxernet/boxernet.py`) is a transformer that:
1. **Encodes** the scene: DINOv3 visual features + semi-dense depth patches are projected to a shared embedding space, then refined with self-attention.
2. **Queries** per detection: each 2D bounding box becomes a query token that cross-attends to the scene encoding.
3. **Predicts** 3D boxes: an MLP head outputs a 7-DoF oriented bounding box (center offset, dimensions, yaw) plus an aleatoric uncertainty estimate.

The pipeline supports optional **3D tracking** (`--track`) for temporal consistency and **3D fusion** (`--fuse`) for merging detections across frames.

## Installation

```bash
conda create -n boxer python=3.12
conda activate boxer

pip install torch torchvision
pip install opencv-python tqdm
pip install sentence-transformers   # for semantic track merging

# 2D detectors (install at least one)
pip install transformers            # for OWLv2 (default)
pip install detic                   # for DETIC (alternative)

```

### Model Checkpoint

Download the BoxerNet checkpoint and place it at `~/data/boxer/`:

```bash
mkdir -p ~/data/boxer
# Place checkpoint file, e.g.:
# ~/data/boxer/bxr_alln2nw12bs12hw960in2x6d768ni1_Nov20.ckpt
```

## Usage

```bash
# Basic: run on an Aria sequence with OWLv2 detector
python run_boxer.py --input /path/to/aria/sequence

# Use DETIC detector instead of OWL
python run_boxer.py --input /path/to/sequence --detector detic

# Custom text prompts
python run_boxer.py --input /path/to/sequence --labels=chair,table,lamp

# Run with online 3D tracking (adds a third visualization panel)
python run_boxer.py --input /path/to/sequence --track

# Run with post-hoc 3D box fusion
python run_boxer.py --input /path/to/sequence --fuse

# ScanNet sequence
python run_boxer.py --input /path/to/scannet/scene0000_00

# Omni3D dataset
python run_boxer.py --input SUNRGBD

# Adjust thresholds
python run_boxer.py --input /path/to/sequence --thresh2d 0.3 --thresh3d 0.6

# Skip visualization (faster, just writes CSV)
python run_boxer.py --input /path/to/sequence --skip_viz

# Use bfloat16 for faster inference on supported GPUs
python run_boxer.py --input /path/to/sequence --precision bfloat16
```

### Outputs

Results are written to `~/viz_boxer/<sequence_name>/`:
- `boxer_3dbbs.csv` — per-frame 3D bounding boxes
- `boxer_2dbbs.csv` — per-frame 2D detections
- `boxer_3dbbs_tracked.csv` — tracked 3D boxes (with `--track`)
- `boxer_viz_final.mp4` — visualization video

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | | Path to input sequence |
| `--detector` | `owl` | 2D detector: `owl` or `detic` |
| `--labels` | `lvisplus` | Comma-separated text prompts, or a taxonomy name |
| `--thresh2d` | `0.2` | 2D detection confidence threshold |
| `--thresh3d` | `0.5` | 3D box confidence threshold |
| `--track` | off | Enable online 3D box tracking |
| `--fuse` | off | Run post-hoc 3D box fusion |
| `--skip_viz` | off | Skip visualization, only write CSVs |
| `--precision` | `float32` | Inference precision (`float32` or `bfloat16`) |
| `--camera` | `rgb` | Aria camera stream (`rgb`, `slaml`, `slamr`) |
| `--pinhole` | off | Rectify fisheye to pinhole |
| `--detector_hw` | `800` | Resize for 2D detector |
| `--ckpt` | see code | Path to BoxerNet checkpoint |
| `--output_dir` | `~/viz_boxer` | Output directory |
| `--gt2d` | off | Use ground-truth 2D boxes as input |
| `--no_sdp` | off | Disable semi-dense point input |
| `--force_cpu` | off | Force CPU inference |

## Project Structure

```
boxer/
├── run_boxer.py              # Main entry point
├── boxernet/
│   ├── boxernet.py           # BoxerNet model (encode → cross-attend → predict)
│   ├── alehead.py            # AleHead: 7-DoF OBB + uncertainty prediction head
│   ├── attention_utils.py    # Transformer blocks (self/cross-attention)
│   └── dinov3_wrapper.py     # DINOv3 backbone wrapper
├── detectors/
│   ├── owl_wrapper.py        # OWLv2 open-vocabulary detector
│   └── detic_wrapper.py      # DETIC detector
├── loaders/
│   ├── aria_loader.py        # Aria glasses data loader
│   ├── ca_loader.py          # CA-1M dataset loader
│   ├── omni_loader.py        # Omni3D dataset loader
│   └── scannet_loader.py     # ScanNet dataset loader
└── utils/
    ├── obb.py                # ObbTW tensor wrapper + IoU computation
    ├── camera.py             # CameraTW: camera intrinsics + projection
    ├── pose.py               # PoseTW: SE(3) poses + quaternion math
    ├── track_3d_boxes.py     # Online 3D bounding box tracker
    ├── fuse_3d_boxes.py      # Post-hoc 3D box fusion
    ├── render.py             # 3D box rendering on images
    ├── file_io.py            # CSV I/O for OBBs and calibration
    ├── condense_text.py      # Sentence-transformer wrapper for semantic matching
    └── ...
```

## License

See the [LICENSE](LICENSE) file for details.
