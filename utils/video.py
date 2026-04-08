# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import os
import shutil
import subprocess


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def safe_delete_folder(folder, extensions=None, keep_folder=False, recursive=False):
    """Deletion utility that will only delete files with matching extension,
    making it safer than shutil.rmtree() which could delete some important
    system files on accident."""
    if extensions is None:
        extensions = [".jpg", ".mp4", ".wav", ".mp3", ".png"]
    if not os.path.exists(folder):
        return
    found = os.listdir(folder)
    for item in found:
        full_path = os.path.join(folder, item)
        if item.endswith(tuple(extensions)):
            os.remove(full_path)
        if recursive and os.path.isdir(full_path):
            safe_delete_folder(full_path, extensions, keep_folder, recursive)
    if not keep_folder:
        if len(os.listdir(folder)) == 0:
            os.rmdir(folder)
        else:
            print(
                "Warning, was unable to remove directory %s because it wasn't empty"
                % folder
            )


def find_ffmpeg(home, ffmpeg_binary=None):
    options = [
        "/usr/bin/ffmpeg",  # Better version on AWS than conda
        "ffmpeg",  # Normal installation will have this in the path (e.g. local machine)
        "/usr/local/fbprojects/fb-motion2/ffmpeg/bin/ffmpeg",  # OD comes with it pre-installed here.
        "/usr/local/fbprojects/ffmpeg-ref/ffmpeg/bin/ffmpeg",  # Some OD comes with this path.
        "%s/ffmpeg/ffmpeg/bin/ffmpeg" % home,
    ]  # Devserver README instruction location.
    if ffmpeg_binary is None:
        for option in options:
            if cmd_exists(option):
                ffmpeg_binary = option
                break
    else:
        assert cmd_exists(ffmpeg_binary), "Provided --ffmpeg_binary does not exist"

    if ffmpeg_binary is None:
        raise IOError(
            "Cannot auto find ffmpeg binary, please see the README for installation help"
        )
    return ffmpeg_binary


def get_video_codec():
    ffmpeg_binary = find_ffmpeg(os.path.expanduser("~"))
    cmd = f"{ffmpeg_binary} -y -v quiet -codecs | grep 264"
    result = str(subprocess.check_output(cmd, shell=True))
    # "libfdk_aac" is better than "aac" because it works with quicktime
    # If libfdk_aac is available, lets use it. On fedora desktop, it is not
    # available unless you install from source, so use "aac" instead.
    if "libx264" in result:  # available on OD-GPU
        codec = "libx264"
    elif "h264" in result:  # available on fedora
        codec = "h264"
    else:
        raise ValueError("Cannot find a valid audio decoder")
    return codec


def make_mp4(
    input_dir,
    framerate,
    ffmpeg_binary=None,
    output_dir=None,
    image_glob="image*.png",
    output_name="out.mp4",
    crf=18,
    preset="slow",
) -> str:
    """Create a mp4 from a directory of images. Output to the input_dir by default."""
    ffmpeg_binary = find_ffmpeg(os.path.expanduser("~"), ffmpeg_binary)
    if output_dir is None:
        output_dir = input_dir
    mp4_path = os.path.join(output_dir, output_name)
    video_codec = get_video_codec()
    cmd = (
        f"{ffmpeg_binary} -hide_banner -loglevel error -y "
        f"-framerate {framerate} -pattern_type glob "
        f"-i '{input_dir}/{image_glob}' "
        f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
        f"-c:v {video_codec} -preset {preset} -crf {crf} "
        f"-profile:v high -level 4.1 "
        f"-pix_fmt yuv420p -movflags +faststart "
        f"{mp4_path}"
    )
    print(cmd)
    subprocess.run(cmd, shell=True)
    return mp4_path
