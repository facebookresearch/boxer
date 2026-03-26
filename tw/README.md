# Tensor Wrappers

The `tw/` package contains `TensorWrapper` — a base class that wraps a raw `torch.Tensor` with named fields and geometric operations — along with three domain-specific subclasses used throughout Boxer.

## Files

### `tensor_wrapper.py` — Base class

`TensorWrapper` stores a single `_data` tensor and exposes standard torch operations (indexing, slicing, `.to()`, `.cuda()`, `.reshape()`, etc.) that propagate through the wrapper. Includes:

- `autocast` decorator: automatically casts method arguments to matching device/dtype
- `autoinit` decorator: builds the `_data` tensor from keyword arguments in subclass constructors
- `smart_cat` / `smart_stack`: concatenate or stack mixed lists of `TensorWrapper` and `Tensor` objects
- DataLoader collate integration for batching `TensorWrapper` instances

### `pose.py` — SE(3) poses (`PoseTW`)

`PoseTW` wraps a `(*, 3, 4)` tensor representing rigid-body transforms (rotation + translation). Provides:

- Compose (`@`), inverse, and transform operations
- Quaternion and rotation matrix conversions (`quat_to_rotmat`, `rotmat_to_quat`)
- Euler angle construction (`rotation_from_euler`)
- SO(3) exponential/logarithmic maps
- Pose interpolation (SLERP rotation + linear translation)
- Geodesic distance between poses

### `camera.py` — Camera intrinsics (`CameraTW`)

`CameraTW` wraps a tensor of camera intrinsic parameters supporting both pinhole and Fisheye624 camera models. Provides:

- `project`: 3D points to 2D pixels (with optional Jacobians)
- `unproject`: 2D pixels + depth to 3D rays/points
- Fisheye624 distortion/undistortion
- Fisheye-to-pinhole rectification via `cv2.remap`
- Focal length, principal point, and image size accessors

### `obb.py` — Oriented bounding boxes (`ObbTW`)

`ObbTW` wraps a tensor encoding 7-DoF oriented bounding boxes (center, dimensions, yaw) plus metadata (semantic ID, confidence, text label). Provides:

- 8-corner computation and 3D rendering line orders
- Point containment and voxel grid sampling
- 3D IoU: exact analytical (`iou_exact7`) and Monte Carlo sampling (`iou_mc7`, `iou_mc9`)
- World-frame to camera-frame transforms via `PoseTW`
- 2D bounding box projection via `CameraTW`
