#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Download a sample CA-1M sequence for use with Boxer.

CA-1M is a dataset of exhaustively annotated indoor scenes released as part of
the Cubify Anything project. Data is in WebDataset format (tar archives per
capture) and is released under CC-by-NC-ND 4.0.

Usage:
    # Download default sequence (42898570, val split) to sample_data/
    python scripts/download_ca1m_sample.py

    # Download a specific video
    python scripts/download_ca1m_sample.py --video-id 45261548

    # Download from train split
    python scripts/download_ca1m_sample.py --video-id 45261548 --split train

    # Download to a custom directory
    python scripts/download_ca1m_sample.py --output-dir /path/to/data

More info: https://github.com/apple/ml-cubifyanything
"""

import argparse
import glob
import os
import ssl
import sys
import tarfile
import tempfile
import urllib.request

CA1M_URL = (
    "https://ml-site.cdn-apple.com/datasets/ca1m/{split}/ca1m-{split}-{video_id}.tar"
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "sample_data")
DEFAULT_VIDEO_ID = "42898570"
DEFAULT_SPLIT = "val"


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

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress


def verify_extracted(data_dir: str) -> bool:
    """Check that the extracted directory has the expected CA-1M structure."""
    world_files = glob.glob(
        os.path.join(data_dir, "**/world.gt/instances.json"), recursive=True
    )
    image_files = glob.glob(
        os.path.join(data_dir, "**/*.wide/image.png"), recursive=True
    )
    has_world = len(world_files) > 0
    has_images = len(image_files) > 0
    if has_world:
        print("  world.gt/instances.json: found")
    else:
        print("  world.gt/instances.json: MISSING")
    if has_images:
        print(f"  frames: {len(image_files)}")
    else:
        print("  frames: MISSING")
    return has_world and has_images


def main():
    parser = argparse.ArgumentParser(
        description="Download a sample CA-1M sequence for Boxer."
    )
    parser.add_argument(
        "--video-id",
        default=DEFAULT_VIDEO_ID,
        help=f"CA-1M video ID to download (default: {DEFAULT_VIDEO_ID})",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["val", "train"],
        help=f"Dataset split (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data already exists",
    )
    args = parser.parse_args()

    seq_name = f"ca1m-{args.split}-{args.video_id}"
    seq_dir = os.path.join(args.output_dir, seq_name)

    # Check if already extracted
    if os.path.isdir(seq_dir) and not args.force:
        world_files = glob.glob(
            os.path.join(seq_dir, "**/world.gt/instances.json"), recursive=True
        )
        if world_files:
            print(f"Already downloaded and extracted: {seq_dir}")
            print("Use --force to re-download.")
            return

    print("NOTE: CA-1M data is released under CC-BY-NC-ND 4.0.")
    print("      https://creativecommons.org/licenses/by-nc-nd/4.0/")
    print()
    answer = input("Do you accept the CA-1M license terms? [y/N] ").strip().lower()
    if answer != "y":
        print("License not accepted. Aborting.")
        sys.exit(0)
    print()

    os.makedirs(seq_dir, exist_ok=True)
    url = CA1M_URL.format(split=args.split, video_id=args.video_id)

    # Download to temp file, extract, then delete tar
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, f"{seq_name}.tar")
        try:
            download_file(url, tar_path)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"\nVideo '{args.video_id}' not found in {args.split} split.")
                print("Check the video ID and split. Valid IDs can be found in:")
                print(
                    "  https://github.com/apple/ml-cubifyanything (data/val.txt, data/train.txt)"
                )
                sys.exit(1)
            else:
                print(f"\nDownload failed (HTTP {e.code}): {e.reason}")
                sys.exit(1)
        except urllib.error.URLError as e:
            print(f"\nDownload failed: {e.reason}")
            print("Check your internet connection and try again.")
            sys.exit(1)

        # Extract
        print(f"\nExtracting to {seq_dir}...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(seq_dir, filter="data")

    # Verify
    print(f"\nVerifying {seq_dir}...")
    if verify_extracted(seq_dir):
        print(f"\nDone! Sequence ready at: {seq_dir}")
        print("\nRun Boxer on it:")
        print(f"  python run_boxer.py --input {seq_name} --max_n=100")
    else:
        print("\nWarning: extracted directory structure looks incomplete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
