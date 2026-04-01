#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Download a sample ScanNet scene for use with Boxer.

Uses the official scannet_frames_25k subset provided by the ScanNet authors.
This subset is publicly hosted and contains color, depth, pose, and intrinsics
for 25k frames across ScanNet scenes.

Usage:
    # Download default scene (scene0084_02) to sample_data/
    python scripts/download_scannet_sample.py

    # Download a specific scene
    python scripts/download_scannet_sample.py --scene scene0339_00

    # Download to a custom directory
    python scripts/download_scannet_sample.py --output_dir /path/to/data

Requirements:
    ScanNet data is released under a restricted Terms of Use. You must agree
    to the terms at http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf before
    downloading. This script uses the publicly available scannet_frames_25k
    subset for demonstration purposes.

    For the full dataset or Scan2CAD annotations, visit:
    - ScanNet: https://github.com/ScanNet/ScanNet
    - Scan2CAD: https://github.com/skanti/Scan2CAD
"""

import argparse
import os
import sys
import tempfile
import urllib.request
import zipfile

SCANNET_FRAMES_25K_URL = (
    "http://kaldir.vc.in.tum.de/scannet/scannet_frames_25k/{scene_id}.zip"
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "sample_data")
DEFAULT_SCENE = "scene0084_02"

# Expected directory structure after extraction
EXPECTED_SUBDIRS = ["color", "depth", "intrinsic", "pose"]


def download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress


def verify_scene(scene_dir: str) -> bool:
    """Check that the downloaded scene has the expected structure."""
    frames_dir = os.path.join(scene_dir, "frames")
    if not os.path.isdir(frames_dir):
        # Some archives extract directly with color/depth/pose at top level
        # Check if we need to wrap in frames/
        if all(os.path.isdir(os.path.join(scene_dir, d)) for d in EXPECTED_SUBDIRS):
            print("Reorganizing: wrapping extracted dirs in frames/")
            os.makedirs(frames_dir, exist_ok=True)
            for d in EXPECTED_SUBDIRS:
                src = os.path.join(scene_dir, d)
                dst = os.path.join(frames_dir, d)
                os.rename(src, dst)
        else:
            return False

    ok = True
    for subdir in EXPECTED_SUBDIRS:
        path = os.path.join(frames_dir, subdir)
        if not os.path.isdir(path):
            print(f"  Missing: {subdir}/")
            ok = False
        else:
            n = len(os.listdir(path))
            print(f"  {subdir}/: {n} files")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Download a sample ScanNet scene for Boxer."
    )
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help=f"ScanNet scene ID to download (default: {DEFAULT_SCENE})",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if scene directory already exists",
    )
    args = parser.parse_args()

    scene_dir = os.path.join(args.output_dir, args.scene)

    # Check if already downloaded
    if os.path.isdir(scene_dir) and not args.force:
        frames_dir = os.path.join(scene_dir, "frames")
        if os.path.isdir(frames_dir) and os.listdir(frames_dir):
            print(f"Scene already exists at {scene_dir}")
            print("Use --force to re-download.")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    url = SCANNET_FRAMES_25K_URL.format(scene_id=args.scene)

    # Download to temp file, then extract
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, f"{args.scene}.zip")

        try:
            download_file(url, zip_path)
        except urllib.error.HTTPError as e:
            if e.code == 403 or e.code == 401:
                print(f"\nAccess denied (HTTP {e.code}).")
                print("The scannet_frames_25k subset may require authentication.")
                print("\nTo download ScanNet data manually:")
                print("1. Visit https://github.com/ScanNet/ScanNet")
                print(
                    "2. Fill out the Terms of Use agreement to receive a download link"
                )
                print("3. Use the provided download script:")
                print(
                    f"   python download-scannet.py -o {args.output_dir} "
                    f"--id {args.scene} --type _2d-instance-filt.zip"
                )
                sys.exit(1)
            elif e.code == 404:
                print(f"\nScene '{args.scene}' not found in scannet_frames_25k.")
                print("This subset contains ~25k frames from select ScanNet scenes.")
                print(f"Try a different scene ID, or download the full dataset from:")
                print("  https://github.com/ScanNet/ScanNet")
                sys.exit(1)
            else:
                raise
        except urllib.error.URLError as e:
            print(f"\nDownload failed: {e.reason}")
            print("Check your internet connection and try again.")
            sys.exit(1)

        # Extract
        print(f"Extracting to {scene_dir}...")
        os.makedirs(scene_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Check archive structure — files may be nested under scene_id/
            top_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}

            if args.scene in top_dirs:
                # Archive contains scene_id/ prefix — extract to output_dir
                zf.extractall(args.output_dir)
            else:
                # Archive contents are at root — extract directly into scene_dir
                zf.extractall(scene_dir)

    # Verify
    print(f"\nVerifying {scene_dir}...")
    if verify_scene(scene_dir):
        print(f"\nDone! Scene ready at: {scene_dir}")
        print(f"\nRun Boxer on it:")
        print(f"  python run_boxer.py --input {args.scene}")
    else:
        print(f"\nWarning: Scene directory structure looks incomplete.")
        print("The loader expects: frames/{{color,depth,pose,intrinsic}}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
