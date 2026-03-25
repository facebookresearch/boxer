# Overview
Boxer lifts 2D object detections into static, global, fused 3D oriented bounding boxes (OBBs) from posed images and semi-dense point clouds, focused on indoor object detection.

This repo contains the code and pre-trained model needed to run Boxer on a variety of input data sources (inference only code). 

We provide examples for running on a variety of data sources:
* Project Aria (Gen 1)
* Project Aria (Gen 2)
* CA-1M
* ScanNet
* SUN-RGBD (single view)

# Input Requirements

For single image lifting with BoxerNet, we require an image, intrinsics calibration (we tested with both Pinhole and Fisheye624 camera models), and the gravity direction. Depth is optional but improves performance significantly.

For lifting a video sequence we need the same as above, but needs the full 6 DoF pose (as opposed to simply the gravity direction) to lift the 3DBBs into the world coordinate frame.

# Boxer

Given a single RGB frame with camera intrinsics, an egomotion pose, and sparse depth, BoxerNet predicts a full 7-DoF 3D bounding box (position, size, yaw) for each 2D detection.

## BoxerNet Architecture

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

# 3D interactive viewer (optional, for --viz_3d)
pip install moderngl moderngl-window imgui[glfw] pyrr

# 2D detector: export OWLv2 checkpoint (one-time, requires transformers)
pip install transformers
python detectors/export_owl.py      # saves to ~/data/boxer/owlv2-base-patch16-ensemble.pt
pip uninstall transformers           # optional: no longer needed at runtime

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
| `--detector` | `owl` | 2D detector (`owl`) |
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
│   ├── owl_wrapper.py        # OWLv2 open-vocabulary detector (JIT-traced, no transformers needed)
│   ├── clip_tokenizer.py     # Minimal CLIP BPE tokenizer
│   └── export_owl.py         # One-time export script (requires transformers)
├── loaders/
│   ├── base_loader.py        # Base loader interface
│   ├── aria_loader.py        # Aria glasses data loader
│   ├── ca_loader.py          # CA-1M dataset loader
│   ├── omni_loader.py        # Omni3D dataset loader
│   └── scannet_loader.py     # ScanNet dataset loader
├── tests/                    # Unit tests (see tests/README.md)
├── tw/                       # TensorWrapper types
│   ├── tensor_wrapper.py     # TensorWrapper base class
│   ├── camera.py             # CameraTW: camera intrinsics + projection
│   ├── obb.py                # ObbTW tensor wrapper + IoU computation
│   └── pose.py               # PoseTW: SE(3) poses + quaternion math
└── utils/
    ├── tensor_utils.py       # Tensor manipulation helpers
    ├── gravity.py            # Gravity alignment utilities
    ├── hungarian.py          # Pure-Python Hungarian algorithm + connected components
    ├── track_3d_boxes.py     # Online 3D bounding box tracker
    ├── fuse_3d_boxes.py      # Post-hoc 3D box fusion
    ├── render.py             # 3D box rendering on images
    ├── viz_boxer.py          # Boxer visualization pipeline
    ├── video.py              # Video I/O utilities
    ├── image.py              # Image utilities
    ├── file_io.py            # CSV I/O for OBBs and calibration
    ├── condense_text.py      # Sentence-transformer wrapper for semantic matching
    ├── taxonomy.py           # Label taxonomy definitions
    ├── settings.py           # Global settings and constants
    ├── demo_utils.py         # Demo helper functions
    └── orbit_viewer.py       # Orbit-based 3D viewer
```

## Tests

See [tests/README.md](tests/README.md) for setup, running, and coverage details.

## FAQ

Q: Can I run it on an arbitrary image without any other info?
A: Theoretically yes, but you would need to estimate the intrinsics and gravity direction. We didn't test that.

Q: Do you plan to release the training or evaluation code?
A: No, we do not, because that would require more long term maintenance from the authors. You can email the first author or leave a github issue if you have any questions about re-implementing these.



## License

See the [LICENSE](LICENSE) file for details.
