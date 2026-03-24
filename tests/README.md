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
python -m pytest tests/test_scipy_replacements.py -v
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
| `test_scipy_replacements.py` | Hungarian (linear sum assignment) and union-find connected components against scipy reference |
