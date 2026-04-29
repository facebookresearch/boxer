"""One-shot probe: dump device calib + first VIO online_calib so we can pick rgb_idx."""

import time

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver

state = {"calib": None, "vio": None, "rgb_record": None}


def device_calib_cb(device_calib):
    state["calib"] = device_calib


def vio_cb(vio):
    if state["vio"] is None:
        state["vio"] = vio


def rgb_cb(image_data, image_record):
    if state["rgb_record"] is None:
        arr = image_data.to_numpy_array()
        state["rgb_record"] = (arr.shape, arr.dtype, str(arr[0, 0]))


device_client = sdk_gen2.DeviceClient()
device_client.set_client_config(sdk_gen2.DeviceClientConfig())
device = device_client.connect()

cfg = sdk_gen2.HttpStreamingConfig()
cfg.profile_name = "profile9"
cfg.streaming_interface = sdk_gen2.StreamingInterface.USB_NCM
device.set_streaming_config(cfg)
device.start_streaming()

srv = sdk_gen2.HttpServerConfig()
srv.address = "0.0.0.0"
srv.port = 6768

rx = receiver.StreamReceiver(enable_image_decoding=True, enable_raw_stream=False)
rx.set_server_config(srv)
rx.register_device_calib_callback(device_calib_cb)
rx.register_vio_callback(vio_cb)
rx.register_rgb_callback(rgb_cb)
rx.start_server()

print("Waiting up to 15s for device calib + vio + rgb...")
deadline = time.time() + 15
while time.time() < deadline and (
    state["calib"] is None or state["vio"] is None or state["rgb_record"] is None
):
    time.sleep(0.1)

print("\n--- DeviceCalibration ---")
calib = state["calib"]
if calib is None:
    print("  (none received)")
else:
    print(f"  type: {type(calib)}")
    print(f"  dir (filtered): {[m for m in dir(calib) if not m.startswith('_')]}")
    try:
        labels = calib.get_camera_labels()
        print(f"  camera labels: {labels}")
    except Exception as e:
        print(f"  get_camera_labels error: {e}")
    try:
        rgb = calib.get_camera_calib("camera-rgb")
        print(f"  camera-rgb: {rgb}")
        print(f"    image_size: {rgb.get_image_size()}")
        print(f"    model_name: {rgb.get_model_name()}")
        print(f"    focal_lengths: {rgb.get_focal_lengths()}")
        print(f"    principal_point: {rgb.get_principal_point()}")
        print(f"    proj_params: {rgb.get_projection_params()}")
        T = rgb.get_transform_device_camera()
        print(f"    T_device_camera: {T}")
        print(f"    T_device_camera.translation(): {T.translation()}")
        print(f"    T_device_camera.rotation().to_matrix(): {T.rotation().to_matrix()}")
    except Exception as e:
        print(f"  get_camera_calib(camera-rgb) error: {e}")

print("\n--- VIO online_calib.cam_parameters ---")
vio = state["vio"]
if vio is None:
    print("  (none received)")
else:
    oc = vio.online_calib
    print(f"  num_cameras: {oc.num_cameras()}")
    for i, cam in enumerate(oc.cam_parameters):
        print(
            f"  cam[{i}]: type={cam.type!r}, image_size={cam.image_size}, "
            f"intrinsics_len={len(cam.intrinsics)}, intrinsics[:6]={list(cam.intrinsics[:6])}"
        )

print("\n--- First RGB frame numpy info ---")
print(f"  {state['rgb_record']}")

print("\nStopping...")
device.stop_streaming()
time.sleep(1)
rx.stop_server()
print("Done.")
