#! /usr/bin/env python3
"""Measure Aria Gen2 RGB streaming and simple display FPS."""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.sensor_data import ImageData, ImageDataRecord

from scripts.live_boxer import connect_with_ip_fallback, ensure_aria_tools_on_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile_name", type=str, default="profile9")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--wifi", action="store_true")
    p.add_argument("--ip", type=str, default=None)
    p.add_argument("--serial", type=str, default=None)
    p.add_argument("--no_display", action="store_true")
    p.add_argument("--display_max_width", type=int, default=960)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_aria_tools_on_path()

    latest_frame = [None]
    latest_seq = [0]
    frame_lock = threading.Lock()
    callback_count = 0
    callback_ts_ns: list[int] = []
    first_shape = None

    def rgb_callback(image_data: ImageData, image_record: ImageDataRecord) -> None:
        nonlocal callback_count, first_shape
        try:
            img = image_data.to_numpy_array()
        except RuntimeError:
            return
        ts_ns = int(image_record.capture_timestamp_ns)
        with frame_lock:
            latest_frame[0] = (img, ts_ns)
            latest_seq[0] += 1
            callback_count += 1
            callback_ts_ns.append(ts_ns)
            if first_shape is None:
                first_shape = img.shape

    device_client = sdk_gen2.DeviceClient()
    device_client.set_client_config(sdk_gen2.DeviceClientConfig())
    device, _target, target_desc = connect_with_ip_fallback(
        device_client, args.ip, args.serial
    )
    print(f"==> Connecting to device ({target_desc})", flush=True)

    streaming_config = sdk_gen2.HttpStreamingConfig()
    streaming_config.profile_name = args.profile_name
    streaming_config.streaming_interface = (
        sdk_gen2.StreamingInterface.WIFI_STA
        if args.wifi
        else sdk_gen2.StreamingInterface.USB_NCM
    )
    device.set_streaming_config(streaming_config)
    device.start_streaming()

    srv_cfg = sdk_gen2.HttpServerConfig()
    srv_cfg.address = "0.0.0.0"
    srv_cfg.port = 6768

    stream_receiver = receiver.StreamReceiver(
        enable_image_decoding=True, enable_raw_stream=False
    )
    stream_receiver.set_server_config(srv_cfg)
    stream_receiver.register_rgb_callback(rgb_callback)
    stream_receiver.start_server()

    win = "Aria Gen2 RGB Benchmark"
    if not args.no_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    display_count = 0
    unique_display_count = 0
    last_display_seq = -1
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < float(args.seconds):
            with frame_lock:
                latest = latest_frame[0]
                seq = latest_seq[0]
            if latest is None:
                time.sleep(0.001)
                continue
            img, ts_ns = latest
            if not args.no_display:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = bgr.shape[:2]
                if w > int(args.display_max_width):
                    scale = float(args.display_max_width) / float(w)
                    bgr = cv2.resize(bgr, (int(args.display_max_width), int(h * scale)))
                cv2.putText(
                    bgr,
                    f"ts={ts_ns}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(win, bgr)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            display_count += 1
            if seq != last_display_seq:
                unique_display_count += 1
                last_display_seq = seq
    finally:
        device.stop_streaming()
        stream_receiver.stop_server()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - t0, 1e-6)
    cb_fps_wall = callback_count / elapsed
    unique_display_fps = unique_display_count / elapsed
    display_fps = display_count / elapsed
    sensor_fps = 0.0
    if len(callback_ts_ns) >= 2:
        ts = np.asarray(callback_ts_ns, dtype=np.int64)
        deltas_s = np.diff(ts).astype(np.float64) / 1e9
        valid = deltas_s[deltas_s > 0.0]
        if valid.size:
            sensor_fps = 1.0 / float(np.median(valid))
    print(
        "==> RGB stream benchmark "
        f"profile={args.profile_name} shape={first_shape} elapsed={elapsed:.2f}s "
        f"callbacks={callback_count} callback_fps={cb_fps_wall:.2f} "
        f"sensor_fps_median={sensor_fps:.2f} "
        f"display_loops_fps={display_fps:.2f} unique_display_fps={unique_display_fps:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
