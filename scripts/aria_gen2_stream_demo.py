"""Aria Gen2 streaming demo: receive RGB frames + VIO pose over HTTP.

Connects to an Aria Gen2 device, starts a streaming profile, and runs a local
HTTP receiver that prints VIO odometry pose and shows the live RGB frame in
an OpenCV window. Press 'q' or Esc in the window to quit.

Canonical sample: aria/gen2_samples/device_streaming.py (installed with the SDK).
"""

import threading

import cv2
import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.sensor_data import (
    FrontendOutput,
    ImageData,
    ImageDataRecord,
)


# Use "mp_streaming_demo" for VIO + eye gaze + hand tracking together.
# "profile9" is a lighter RGB + VIO profile.
STREAMING_PROFILE = "profile9"
RECEIVER_ADDRESS = "0.0.0.0"
RECEIVER_PORT = 6768
DISPLAY_MAX_WIDTH = 960  # downscale 12MP frames for display

# RGB callback runs on a worker thread; cv2.imshow must be called from the
# main thread (especially on macOS), so we hand off the latest frame here.
_latest_frame: list = [None]
_frame_lock = threading.Lock()


def rgb_callback(image_data: ImageData, image_record: ImageDataRecord) -> None:
    img = image_data.to_numpy_array()  # H x W x 3, RGB
    with _frame_lock:
        _latest_frame[0] = (img, image_record.capture_timestamp_ns)


def vio_callback(vio: FrontendOutput) -> None:
    T = vio.transform_odometry_bodyimu
    t = T.translation()
    R = T.rotation().log()  # axis-angle
    print(f"VIO @ {vio.capture_timestamp_ns}: t={t}, R={R}")


def main() -> None:
    device_client = sdk_gen2.DeviceClient()
    device_client.set_client_config(sdk_gen2.DeviceClientConfig())
    device = device_client.connect()

    streaming_config = sdk_gen2.HttpStreamingConfig()
    streaming_config.profile_name = STREAMING_PROFILE
    streaming_config.streaming_interface = sdk_gen2.StreamingInterface.USB_NCM
    device.set_streaming_config(streaming_config)
    device.start_streaming()

    srv_cfg = sdk_gen2.HttpServerConfig()
    srv_cfg.address = RECEIVER_ADDRESS
    srv_cfg.port = RECEIVER_PORT

    stream_receiver = receiver.StreamReceiver(
        enable_image_decoding=True, enable_raw_stream=False
    )
    stream_receiver.set_server_config(srv_cfg)
    stream_receiver.register_rgb_callback(rgb_callback)
    stream_receiver.register_vio_callback(vio_callback)
    stream_receiver.start_server()

    win = "Aria Gen2 RGB"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Streaming. Press 'q' or Esc in the window to stop.")

    try:
        while True:
            with _frame_lock:
                latest = _latest_frame[0]
            if latest is not None:
                img, ts_ns = latest
                # Aria images come back as RGB; OpenCV expects BGR.
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = bgr.shape[:2]
                if w > DISPLAY_MAX_WIDTH:
                    scale = DISPLAY_MAX_WIDTH / w
                    bgr = cv2.resize(bgr, (DISPLAY_MAX_WIDTH, int(h * scale)))
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
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        device.stop_streaming()
        stream_receiver.stop_server()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
