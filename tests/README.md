# Tests

## Setup

```bash
pip install pytest pytest-cov scipy
```

`scipy` is only needed as a reference for validating our pure-Python replacements (Hungarian algorithm, connected components).

## Running Tests

From the repo root:

```bash
# Run all tests with coverage report
tests/run_tests.sh

# Run without auto-opening the HTML report
tests/run_tests.sh --no-open

# Run a specific test file
python -m pytest tests/test_fusion.py -v
```

## Coverage

After running `tests/run_tests.sh`, the HTML coverage report is at `tests/htmlcov/index.html`.

## Test Files

| File | What it tests |
|------|---------------|
| `test_attention.py` | FeedForward, Attention, and AttentionBlockV2 modules (shapes, gradients, masking, determinism) |
| `test_camera.py` | CameraTW project/unproject round-trips, fisheye projection, backward passes |
| `test_iou.py` | Exact and sampling-based 3D IoU (iou_exact7, iou_mc9) |
| `test_obb.py` | ObbTW construction, point containment, voxel grid sampling, batched IoU |
| `test_gravity.py` | Gravity alignment and vector rejection utilities |
| `test_pose.py` | PoseTW SE(3) pose and quaternion math |
| `test_fusion.py` | Hungarian algorithm, connected components, 3D box fusion (rotation handling, min_detections, clustering) |
| `test_boxernet_utils.py` | BoxerNet utility functions: image_to_patches, masked_median, patch centers |
| `test_file_io.py` | OBB CSV and 2D bounding box CSV read/write round-trips |
| `test_image.py` | Image conversion utilities: normalize, torch2cv2, rotate, string2color |
| `test_taxonomy.py` | Semantic label dictionaries, color validation, text label loading |
| `test_tensor_utils.py` | String/tensor conversions, find_nearest, pad_points |
| `test_tensor_wrapper.py` | TensorWrapper base class operations |
| `test_track_3d_boxes.py` | 3D box tracker: track creation, promotion, aging, confidence filtering |
