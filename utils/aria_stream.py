import ipaddress
import os
import re
import subprocess
import threading
import time
from typing import Optional

import numpy as np
import torch

import aria
import aria.sdk_gen2 as sdk_gen2
from projectaria_tools.core.sensor_data import (
    FrontendOutput,
    ImageData,
    ImageDataRecord,
)

from utils.tw.pose import PoseTW


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARIA_LAST_IP_PATH = os.path.join(REPO_ROOT, ".aria_last_ip.txt")

GEN2_CAMERA_ID_TO_LABEL = {
    1: "slam-front-left",
    2: "slam-front-right",
    4: "slam-side-left",
    8: "slam-side-right",
    16: "camera-et-left",
    32: "camera-et-right",
    64: "camera-rgb",
}


def ensure_aria_tools_on_path() -> None:
    aria_dir = os.path.dirname(os.path.abspath(aria.__file__))
    tools_dir = os.path.join(aria_dir, "tools")
    if not os.path.exists(os.path.join(tools_dir, "adb")):
        return
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if tools_dir not in path_parts:
        os.environ["PATH"] = tools_dir + os.pathsep + os.environ.get("PATH", "")


def load_cached_aria_ip() -> Optional[str]:
    try:
        with open(ARIA_LAST_IP_PATH, "r", encoding="ascii") as f:
            ip = f.read().strip()
    except OSError:
        return None
    if not ip:
        return None
    try:
        ipaddress.ip_address(ip)
    except ValueError:
        return None
    return ip


def save_cached_aria_ip(ip: str) -> None:
    try:
        with open(ARIA_LAST_IP_PATH, "w", encoding="ascii") as f:
            f.write(ip + "\n")
    except OSError:
        pass


def detect_aria_usb_device() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["lsusb"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None

    patterns = [
        "aria gen 2",
        "oculus vr, inc. aria gen 2",
        "meta aria gen 2",
    ]
    for line in proc.stdout.splitlines():
        lower = line.lower()
        if any(pat in lower for pat in patterns):
            return line.strip()
    return None


def _iface_priority_for_aria_peer(iface: str) -> int:
    if iface.startswith("enx") or iface.startswith("usb"):
        return 0
    if iface.startswith("eth"):
        return 1
    return 5


def find_usb_ncm_device_ip() -> Optional[str]:
    """Best-effort discovery of the Aria peer on the USB-NCM link."""
    try:
        addr = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if addr.returncode != 0:
        return None

    candidate_peers: list[tuple[int, str]] = []
    for line in addr.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        iface = parts[1]
        if not (
            iface.startswith("enx")
            or iface.startswith("usb")
            or iface.startswith("eth")
        ):
            continue
        cidr = parts[3]
        try:
            network = ipaddress.ip_interface(cidr).network
        except ValueError:
            continue
        if not network.is_private:
            continue

        try:
            neigh = subprocess.run(
                ["ip", "neigh", "show", "dev", iface],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if neigh.returncode != 0:
            continue

        for neigh_line in neigh.stdout.splitlines():
            match = re.match(r"^(\d+\.\d+\.\d+\.\d+)\s+.*\blladdr\b", neigh_line)
            if not match or "FAILED" in neigh_line:
                continue
            peer_ip = match.group(1)
            try:
                if ipaddress.ip_address(peer_ip) in network:
                    candidate_peers.append(
                        (_iface_priority_for_aria_peer(iface), peer_ip)
                    )
            except ValueError:
                continue

    if not candidate_peers:
        return None
    candidate_peers.sort(key=lambda item: item[0])
    return candidate_peers[0][1]


def has_usb_ncm_ipv4_interface() -> bool:
    try:
        addr = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if addr.returncode != 0:
        return False
    for line in addr.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        iface = parts[1]
        if iface.startswith("enx") or iface.startswith("usb") or iface.startswith("eth"):
            return True
    return False


def connect_with_ip_fallback(
    device_client: sdk_gen2.DeviceClient,
    explicit_ip: Optional[str],
    explicit_serial: Optional[str],
) -> tuple[object, sdk_gen2.DeviceTarget, str]:
    if explicit_ip:
        target = sdk_gen2.DeviceTarget(ip=explicit_ip)
        return device_client.connect(target), target, f"ip={explicit_ip}"
    if explicit_serial:
        target = sdk_gen2.DeviceTarget(serial=explicit_serial)
        return device_client.connect(target), target, f"serial={explicit_serial}"

    aria_usb_line = detect_aria_usb_device()
    if aria_usb_line is None:
        raise SystemExit(
            "Aria Gen 2 is not enumerated over USB on this host. "
            "Check the cable, port, and headset USB mode, then confirm it appears in `lsusb`."
        )

    candidates: list[str] = []
    discovered_ip = find_usb_ncm_device_ip()
    if discovered_ip:
        candidates.append(discovered_ip)
    elif not has_usb_ncm_ipv4_interface():
        raise SystemExit(
            "No USB-NCM IPv4 interface is up for the glasses. "
            "Reconnect the headset over a data-capable USB cable and wait for the USB network device to appear."
        )
    cached_ip = load_cached_aria_ip()
    if cached_ip and cached_ip not in candidates:
        candidates.append(cached_ip)

    if not candidates:
        raise SystemExit(
            "Could not discover an Aria USB-NCM peer IP and no cached IP was found. "
            "The glasses are not enumerated as a USB network device on this host. "
            "Reconnect the glasses over a data-capable USB cable, make sure USB-NCM is up, "
            "or pass --ip directly."
        )

    last_error = None
    for ip in candidates:
        target = sdk_gen2.DeviceTarget(ip=ip)
        try:
            device = device_client.connect(target)
            save_cached_aria_ip(ip)
            desc = f"cached/discovered USB-NCM ip={ip}"
            return device, target, desc
        except RuntimeError as err:
            last_error = err
            continue

    raise RuntimeError(
        f"Failed to connect using cached/discovered IPs {candidates}: {last_error}"
    )


def _append_ts_history(history: list[int], ts_ns: int, max_len: int = 64) -> None:
    history.append(int(ts_ns))
    if len(history) > max_len:
        del history[: len(history) - max_len]


def _hz_from_ts_history(history: list[int]) -> float:
    if len(history) < 2:
        return 0.0
    ts = np.asarray(history, dtype=np.int64)
    deltas_s = np.diff(ts).astype(np.float64) / 1e9
    valid = deltas_s[deltas_s > 0.0]
    if valid.size == 0:
        return 0.0
    return float(1.0 / np.median(valid))


class StreamState:
    """Thread-safe slots populated by streaming callbacks, read on main thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame: Optional[tuple[np.ndarray, int]] = None
        self.T_world_rig: Optional[PoseTW] = None
        self.T_camera_rig: Optional[PoseTW] = None
        self.rgb_image_size: Optional[tuple[int, int]] = None
        self.rgb_intrinsics: Optional[list[float]] = None
        self.rgb_ts_history: list[int] = []
        self.debug_geometry_logged = False
        self.rectify_debug_logged = False

    def snapshot(self):
        with self.lock:
            return (
                self.frame,
                self.T_world_rig,
                self.rgb_intrinsics,
                self.T_camera_rig,
                self.rgb_image_size,
            )

    def stream_hz(self) -> float:
        with self.lock:
            return _hz_from_ts_history(self.rgb_ts_history)


class LiveFsState:
    def __init__(self):
        self.lock = threading.Lock()
        self.left_frame: Optional[tuple[np.ndarray, int]] = None
        self.right_frame: Optional[tuple[np.ndarray, int]] = None
        self.left_calib = None
        self.right_calib = None
        self.T_world_device: Optional[np.ndarray] = None
        self.slam_ts_history: list[int] = []

    def snapshot(self):
        with self.lock:
            return (
                self.left_frame,
                self.right_frame,
                self.left_calib,
                self.right_calib,
                None if self.T_world_device is None else self.T_world_device.copy(),
            )

    def stream_hz(self) -> float:
        with self.lock:
            return _hz_from_ts_history(self.slam_ts_history)


def make_live_fs_callbacks(state: LiveFsState):
    def device_calib_cb(device_calib):
        left = device_calib.get_camera_calib("slam-front-left")
        right = device_calib.get_camera_calib("slam-front-right")
        if left is None or right is None:
            return
        with state.lock:
            state.left_calib = left
            state.right_calib = right
        print(
            f"==> FS calib received: slam-front-left {left.get_image_size()}, "
            f"slam-front-right {right.get_image_size()}",
            flush=True,
        )

    def slam_cb(image_data: ImageData, image_record: ImageDataRecord):
        camera_id = int(image_record.camera_id)
        if camera_id not in (1, 2):
            return
        arr = image_data.to_numpy_array()
        frame = (arr, int(image_record.capture_timestamp_ns))
        with state.lock:
            if camera_id == 1:
                state.left_frame = frame
                _append_ts_history(state.slam_ts_history, frame[1])
            else:
                state.right_frame = frame

    def vio_cb(vio: FrontendOutput):
        M_odo_bi = np.asarray(
            vio.transform_odometry_bodyimu.to_matrix(), dtype=np.float32
        )
        M_bi_dev = np.asarray(
            vio.transform_bodyimu_device.to_matrix(), dtype=np.float32
        )
        with state.lock:
            state.T_world_device = M_odo_bi @ M_bi_dev

    return device_calib_cb, slam_cb, vio_cb


def make_callbacks(state: StreamState):
    def device_calib_cb(device_calib):
        rgb = device_calib.get_camera_calib("camera-rgb")
        T_dev_cam_mat = np.asarray(
            rgb.get_transform_device_camera().to_matrix(), dtype=np.float32
        )
        T_dev_cam = PoseTW.from_matrix(torch.from_numpy(T_dev_cam_mat))
        T_cam_dev = T_dev_cam.inverse().float()
        iw, ih = rgb.get_image_size()
        factory_params = list(rgb.get_projection_params())
        with state.lock:
            state.T_camera_rig = T_cam_dev
            state.rgb_image_size = (int(iw), int(ih))
            state.rgb_intrinsics = factory_params
        print(
            f"==> Device calib received: camera-rgb {iw}x{ih}, "
            f"model={rgb.get_model_name()}, focal={factory_params[0]:.2f}, "
            f"n_params={len(factory_params)}"
        )

    def vio_cb(vio: FrontendOutput):
        M_odo_bi = np.asarray(
            vio.transform_odometry_bodyimu.to_matrix(), dtype=np.float32
        )
        M_bi_dev = np.asarray(
            vio.transform_bodyimu_device.to_matrix(), dtype=np.float32
        )
        M_odo_dev = M_odo_bi @ M_bi_dev
        T_wr = PoseTW.from_matrix(torch.from_numpy(M_odo_dev)).float()
        with state.lock:
            state.T_world_rig = T_wr

    def rgb_cb(image_data: ImageData, image_record: ImageDataRecord):
        try:
            arr = image_data.to_numpy_array()
        except RuntimeError as exc:
            now = time.time()
            last_warn = getattr(rgb_cb, "_last_decode_warn", 0.0)
            if now - last_warn > 2.0:
                print(f"==> Skipping undecodable RGB frame: {exc}", flush=True)
                rgb_cb._last_decode_warn = now
            return
        ts_ns = int(image_record.capture_timestamp_ns)
        with state.lock:
            state.frame = (arr, ts_ns)
            _append_ts_history(state.rgb_ts_history, ts_ns)

    return device_calib_cb, vio_cb, rgb_cb


def make_slam_probe_callback():
    counts: dict[str, int] = {}
    seen: set[str] = set()
    lock = threading.Lock()

    def slam_cb(image_data: ImageData, image_record: ImageDataRecord):
        name = GEN2_CAMERA_ID_TO_LABEL.get(
            int(image_record.camera_id), str(image_record.camera_id)
        )
        arr = image_data.to_numpy_array()
        with lock:
            counts[name] = counts.get(name, 0) + 1
            first = name not in seen
            if first:
                seen.add(name)
        if first:
            print(
                f"==> First {name}: shape={arr.shape}, "
                f"ts={int(image_record.capture_timestamp_ns)}",
                flush=True,
            )

    return slam_cb, counts
