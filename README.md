# Overview
Boxer lifts 2D object detections into static, global, fused 3D oriented bounding boxes (OBBs) from posed images and semi-dense point clouds, focused on indoor object detection.

This repo contains the code and pre-trained model needed to run Boxer on a variety of input data sources (inference only code).

![Boxer System Architecture](assets/boxer_system.jpg)

We provide examples for running on a variety of data sources:
* Project Aria (Gen 1)
* Project Aria (Gen 2)
* CA-1M
* ScanNet
* SUN-RGBD (single view)

# Input Requirements

For single image lifting with BoxerNet, we require an image, intrinsics calibration (we tested with both Pinhole and Fisheye624 camera models), and the gravity direction. Depth is optional but improves performance significantly.

For lifting a video sequence we need the same as above, but needs the full 6 DoF pose (as opposed to simply the gravity direction) to lift the 3DBBs into the world coordinate frame.

# BoxerNet

See [boxernet/README.md](boxernet/README.md) for architecture details.


## Installation

```bash
conda create -n boxer python=3.12
conda activate boxer

pip install 'numpy>=1.24' 'torch>=2.0' 'opencv-python>=4.8' tqdm

# ffmpeg (required for video output)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

# Aria data (optional, for Project Aria sequences)
pip install projectaria-tools

# 3D interactive viewer (optional, for --viz_3d)
pip install moderngl moderngl-window imgui[glfw] pyrr

```

### Model Checkpoint

Download the checkpoints and place them in the `ckpts/` directory:

```bash
mkdir -p ckpts
# Place checkpoint files:
# ckpts/bxr_alln2nw12bs12hw960in2x6d768ni1_Nov20.ckpt
# ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
# ckpts/owlv2-base-patch16-ensemble.pt
```

### Sample Data

The repo includes sample sequences in `sample_data/` for Aria (`sor01`, `hohen`), CA-1M, and ScanNet. Helper scripts are provided to set up additional data:

```bash
# ScanNet: download a sample scene using the official scannet_frames_25k subset
python scripts/download_scannet_sample.py

# Omni3D SUN-RGBD: extract 20 sample images from your local SUNRGBD data
python scripts/download_omni3d_sample.py

# CA-1M: extract a sample sequence from your local CA-1M data
python scripts/download_ca1m_sample.py
```

Note: ScanNet data is subject to the [ScanNet Terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf).
For the full dataset, request access at [github.com/ScanNet/ScanNet](https://github.com/ScanNet/ScanNet).
For ground-truth 3D box annotations, see [Scan2CAD](https://github.com/skanti/Scan2CAD).

## Usage

The pipeline supports optional **online 3D tracking** (`--track`) for temporal consistency and **offline 3D fusion** (`--fuse`) for merging detections across frames after all detections have been made.

```bash

# Basic: run on a sample Aria sequence (place data in sample_data/)
python run_boxer.py --input sor01

# Disable visualization (faster, just writes CSV)
python run_boxer.py --input sor01 --no_viz

# Custom text prompts
python run_boxer.py --input sor01 --labels=chair,table,lamp

# Run with online 3D tracking (adds a third visualization panel)
python run_boxer.py --input sor01 --track

# Run with post-hoc 3D box fusion
python run_boxer.py --input sor01 --fuse

# ScanNet sequence
python run_boxer.py --input scene0084_02

# CA-1M sequence
python run_boxer.py --input ca1m-val-45261179

# Omni3D dataset
python run_boxer.py --input SUNRGBD

# Adjust thresholds
python run_boxer.py --input /path/to/sequence --thresh2d 0.3 --thresh3d 0.6

# Aria sequence from sample_data/
python run_boxer.py --input hohen

# Use bfloat16 for faster inference on supported GPUs
python run_boxer.py --input /path/to/sequence --precision bfloat16
```

### Outputs

Results are written to `output/<sequence_name>/`:
- `boxer_3dbbs.csv` — per-frame 3D bounding boxes
- `owl_2dbbs.csv` — per-frame 2D detections
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
| `--no_viz` | off | Disable visualization (on by default) |
| `--precision` | `float32` | Inference precision (`float32` or `bfloat16`) |
| `--camera` | `rgb` | Aria camera stream (`rgb`, `slaml`, `slamr`) |
| `--pinhole` | off | Rectify fisheye to pinhole |
| `--detector_hw` | `960` | Resize for 2D detector |
| `--ckpt` | see code | Path to BoxerNet checkpoint |
| `--output_dir` | `output/` | Output directory |
| `--gt2d` | off | Use ground-truth 2D boxes as input |
| `--no_sdp` | off | Disable semi-dense point input |
| `--force_cpu` | off | Force CPU inference |

## Project Structure

```
boxer/
├── run_boxer.py              # Main entry point
├── boxernet/
│   ├── boxernet.py           # BoxerNet model (encode → cross-attend → predict)
│   └── dinov3_wrapper.py     # DINOv3 backbone wrapper
├── owl/
│   ├── owl_wrapper.py        # OWLv2 open-vocabulary detector (JIT-traced, no transformers needed)
│   └── clip_tokenizer.py     # CLIP BPE tokenizer + text embedder
├── loaders/
│   ├── base_loader.py        # Base loader interface
│   ├── aria_loader.py        # Aria glasses data loader
│   ├── ca_loader.py          # CA-1M dataset loader
│   ├── omni_loader.py        # Omni3D dataset loader
│   └── scannet_loader.py     # ScanNet dataset loader
├── scripts/
│   ├── download_scannet_sample.py   # Download ScanNet sample data
│   ├── download_omni3d_sample.py    # Extract Omni3D SUN-RGBD sample
│   └── download_ca1m_sample.py      # Extract CA-1M sample data
├── tests/                    # Unit tests (see tests/README.md)
├── tw/                       # TensorWrapper types (see tw/README.md)
│   ├── tensor_wrapper.py     # TensorWrapper base class
│   ├── camera.py             # CameraTW: camera intrinsics + projection
│   ├── obb.py                # ObbTW tensor wrapper + IoU computation
│   ├── pose.py               # PoseTW: SE(3) poses + quaternion math
│   └── tensor_utils.py       # String/tensor conversions, array helpers
└── utils/
    ├── fuse_3d_boxes.py      # 3D box fusion + Hungarian algorithm
    ├── track_3d_boxes.py     # Online 3D bounding box tracker
    ├── file_io.py            # CSV I/O for OBBs and calibration
    ├── image.py              # Image utilities + 3D/2D box rendering
    ├── viewer.py             # Interactive 3D visualization
    ├── orbit_viewer.py       # Orbit-based 3D viewer
    ├── gravity.py            # Gravity alignment utilities
    ├── taxonomy.py           # Label taxonomy definitions
    ├── demo_utils.py         # Demo helpers, paths, timing
    └── video.py              # Video I/O utilities
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
