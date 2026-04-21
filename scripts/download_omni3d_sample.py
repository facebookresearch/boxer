#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Create a small sample of the Omni3D SUN-RGBD dataset for testing Boxer.

Downloads only the selected subset of images from Princeton's SUNRGBD zip
using HTTP range requests (curl), avoiding multi-GB full downloads.

Usage:
    # Create 20-image sample (default)
    python scripts/download_omni3d_sample.py

    # Custom count
    python scripts/download_omni3d_sample.py --num_images 10
"""

import argparse
import json
import os
import ssl
import struct
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import zlib

OMNI3D_JSON_URL = "https://dl.fbaipublicfiles.com/omni3d_data/Omni3D_json.zip"
SUNRGBD_ZIP_URL = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "sample_data", "Omni3D")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "sample_data", "Omni3D")


def download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    print(f"Downloading: {url}")

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
    print()


class RemoteZip:
    """Read individual files from a remote zip archive using HTTP range requests.

    Only downloads the central directory once, then extracts files on demand.
    """

    def __init__(self, url: str):
        self.url = url
        self.entries = {}  # name -> (comp_method, comp_size, local_offset)
        self._ready = False

    def _curl_range(self, start: int, end: int) -> bytes:
        result = subprocess.run(
            ["curl", "-s", "--range", f"{start}-{end}", self.url],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed for range {start}-{end}")
        return result.stdout

    def open(self) -> bool:
        """Fetch the central directory. Returns False if range requests unsupported."""
        # HEAD request
        result = subprocess.run(
            ["curl", "-sI", self.url], capture_output=True, text=True
        )
        if result.returncode != 0:
            return False

        content_length = None
        accepts_ranges = False
        for line in result.stdout.splitlines():
            lower = line.lower()
            if lower.startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
            if lower.startswith("accept-ranges:") and "bytes" in lower:
                accepts_ranges = True

        if not content_length or not accepts_ranges:
            return False

        # Read End of Central Directory (last 65 KB)
        eocd_size = min(65536, content_length)
        eocd_data = self._curl_range(content_length - eocd_size, content_length - 1)

        eocd_pos = eocd_data.rfind(b"\x50\x4b\x05\x06")
        if eocd_pos == -1 or len(eocd_data) - eocd_pos < 22:
            return False

        cd_size = struct.unpack_from("<I", eocd_data, eocd_pos + 12)[0]
        cd_offset = struct.unpack_from("<I", eocd_data, eocd_pos + 16)[0]

        # Handle Zip64
        if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
            loc_pos = eocd_data.rfind(b"\x50\x4b\x06\x07")
            if loc_pos == -1 or len(eocd_data) - loc_pos < 20:
                return False
            eocd64_offset = struct.unpack_from("<Q", eocd_data, loc_pos + 8)[0]
            eocd64 = self._curl_range(eocd64_offset, eocd64_offset + 63)
            if len(eocd64) < 56 or eocd64[:4] != b"\x50\x4b\x06\x06":
                return False
            cd_size = struct.unpack_from("<Q", eocd64, 40)[0]
            cd_offset = struct.unpack_from("<Q", eocd64, 48)[0]

        print(f"  Fetching zip central directory ({cd_size / 1024:.0f} KB)...")
        cd_data = self._curl_range(cd_offset, cd_offset + cd_size - 1)

        # Parse all Central Directory entries
        pos = 0
        while pos + 46 <= len(cd_data):
            if cd_data[pos : pos + 4] != b"\x50\x4b\x01\x02":
                break
            comp_method = struct.unpack_from("<H", cd_data, pos + 10)[0]
            comp_size = struct.unpack_from("<I", cd_data, pos + 20)[0]
            name_len = struct.unpack_from("<H", cd_data, pos + 28)[0]
            extra_len = struct.unpack_from("<H", cd_data, pos + 30)[0]
            comment_len = struct.unpack_from("<H", cd_data, pos + 32)[0]
            local_offset = struct.unpack_from("<I", cd_data, pos + 42)[0]

            name = cd_data[pos + 46 : pos + 46 + name_len].decode(
                "utf-8", errors="replace"
            )

            # Zip64 extra field
            if extra_len > 0 and (
                comp_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF
            ):
                extra = cd_data[pos + 46 + name_len : pos + 46 + name_len + extra_len]
                epos = 0
                while epos + 4 <= len(extra):
                    tag = struct.unpack_from("<H", extra, epos)[0]
                    sz = struct.unpack_from("<H", extra, epos + 2)[0]
                    if tag == 0x0001:
                        fo = epos + 4
                        if struct.unpack_from("<I", cd_data, pos + 24)[0] == 0xFFFFFFFF:
                            fo += 8
                        if comp_size == 0xFFFFFFFF and fo + 8 <= epos + 4 + sz:
                            comp_size = struct.unpack_from("<Q", extra, fo)[0]
                            fo += 8
                        if local_offset == 0xFFFFFFFF and fo + 8 <= epos + 4 + sz:
                            local_offset = struct.unpack_from("<Q", extra, fo)[0]
                        break
                    epos += 4 + sz

            self.entries[name] = (comp_method, comp_size, local_offset)
            pos += 46 + name_len + extra_len + comment_len

        print(f"  Found {len(self.entries)} entries in zip")
        self._ready = True
        return True

    def find(self, suffix: str) -> str | None:
        """Find the first entry name ending with the given suffix."""
        for name in self.entries:
            if name.endswith(suffix):
                return name
        return None

    def find_all(self, prefix: str) -> list[str]:
        """Find all entry names starting with the given prefix."""
        return [n for n in self.entries if n.startswith(prefix)]

    def extract(self, entry_name: str, dest_path: str) -> bool:
        """Download and extract a single file to dest_path."""
        if entry_name not in self.entries:
            return False

        comp_method, comp_size, local_offset = self.entries[entry_name]
        if comp_size == 0:
            return False

        # Download local file header + compressed data in one request.
        # Local header is 30 + name_len + extra_len bytes; we overshoot by
        # fetching 30 + 512 (generous header) + comp_size in one range.
        max_header = 30 + 512
        blob = self._curl_range(local_offset, local_offset + max_header + comp_size - 1)
        if len(blob) < 30:
            return False

        local_name_len = struct.unpack_from("<H", blob, 26)[0]
        local_extra_len = struct.unpack_from("<H", blob, 28)[0]
        data_start = 30 + local_name_len + local_extra_len
        raw = blob[data_start : data_start + comp_size]

        if comp_method == 0:
            content = raw
        elif comp_method == 8:
            content = zlib.decompress(raw, -zlib.MAX_WBITS)
        else:
            return False

        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(content)
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Create a sample Omni3D SUN-RGBD subset for Boxer."
    )
    parser.add_argument(
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help=f"Path to full Omni3D data (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=20,
        help="Number of sample images to include (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image selection (default: 42)",
    )
    args = parser.parse_args()

    # Step 1: Get the Omni3D JSON (download or use existing)
    json_path = os.path.join(args.data_root, "SUNRGBD_val.json")

    if not os.path.exists(json_path):
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, "SUNRGBD_val.json")

        print(
            "SUNRGBD_val.json not found locally, downloading via curl range requests..."
        )
        rz = RemoteZip(OMNI3D_JSON_URL)
        if rz.open():
            entry = rz.find("SUNRGBD_val.json")
            if entry:
                comp_size = rz.entries[entry][1]
                print(
                    f"  Downloading SUNRGBD_val.json ({comp_size / (1024 * 1024):.1f} MB)..."
                )
                if rz.extract(entry, json_path):
                    print(f"  Saved to {json_path}")
                else:
                    print("  Failed to extract SUNRGBD_val.json")
                    sys.exit(1)
            else:
                print("  SUNRGBD_val.json not found in Omni3D_json.zip")
                sys.exit(1)
        else:
            print("  Range requests not supported, falling back to full download...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, "Omni3D_json.zip")
                try:
                    download_file(OMNI3D_JSON_URL, zip_path)
                except Exception as e:
                    print(f"\nFailed to download Omni3D JSON: {e}")
                    sys.exit(1)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    target = None
                    for name in zf.namelist():
                        if name.endswith("SUNRGBD_val.json"):
                            target = name
                            break
                    if target is None:
                        print("SUNRGBD_val.json not found in Omni3D_json.zip")
                        sys.exit(1)
                    with zf.open(target) as src, open(json_path, "wb") as dst:
                        dst.write(src.read())
                    print(f"Extracted SUNRGBD_val.json to {json_path}")

    # Step 2: Load JSON and select images
    print(f"Loading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    all_images = data["images"]
    all_annotations = data["annotations"]
    categories = data["categories"]

    print(f"Full dataset: {len(all_images)} images, {len(all_annotations)} annotations")

    # Select images deterministically
    import random

    random.seed(args.seed)
    selected = random.sample(all_images, min(args.num_images, len(all_images)))
    selected_ids = {img["id"] for img in selected}

    # Filter annotations to selected images
    selected_anns = [a for a in all_annotations if a["image_id"] in selected_ids]

    print(f"Selected {len(selected)} images with {len(selected_anns)} annotations")

    # Step 3: Download image/depth/extrinsics for selected scenes from Princeton zip
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect the scene directory prefixes we need
    scene_prefixes = set()
    for img in selected:
        file_path = img["file_path"]  # e.g. SUNRGBD/kv2/.../image/0000269.jpg
        scene_dir = os.path.dirname(os.path.dirname(file_path))  # .../scene_folder
        scene_prefixes.add(scene_dir)

    # Check which scenes are already downloaded
    needed = set()
    for prefix in scene_prefixes:
        scene_out = os.path.join(args.output_dir, prefix)
        if not os.path.isdir(os.path.join(scene_out, "image")):
            needed.add(prefix)

    if not needed:
        print(f"All {len(scene_prefixes)} scenes already downloaded")
    else:
        print(
            f"\nDownloading {len(needed)} scenes from SUNRGBD.zip via range requests..."
        )
        print("  (central directory is ~63 MB, one-time cost)")

        rz_sun = RemoteZip(SUNRGBD_ZIP_URL)
        if not rz_sun.open():
            print("Error: could not read SUNRGBD.zip (range requests not supported)")
            sys.exit(1)

        downloaded = 0
        for i, prefix in enumerate(sorted(needed)):
            # Find all entries under this scene's image/, depth/, extrinsics/ dirs
            files_to_get = []
            for subdir in ["image", "depth", "extrinsics"]:
                subdir_prefix = prefix + "/" + subdir + "/"
                files_to_get.extend(rz_sun.find_all(subdir_prefix))

            if not files_to_get:
                print(f"  [{i + 1}/{len(needed)}] Warning: no files found for {prefix}")
                continue

            print(
                f"  [{i + 1}/{len(needed)}] {os.path.basename(prefix)} ({len(files_to_get)} files)"
            )
            for entry_name in files_to_get:
                dest = os.path.join(args.output_dir, entry_name)
                if not os.path.exists(dest):
                    rz_sun.extract(entry_name, dest)
            downloaded += 1

        print(f"Downloaded {downloaded} scenes")

    # Step 5: Write trimmed JSON
    trimmed = {
        "info": data.get("info", {}),
        "images": selected,
        "annotations": selected_anns,
        "categories": categories,
    }
    out_json = os.path.join(args.output_dir, "SUNRGBD_val.json")
    with open(out_json, "w") as f:
        json.dump(trimmed, f)
    print(f"Wrote trimmed JSON: {out_json} ({len(selected)} images)")

    print(f"\nDone! Sample data at: {args.output_dir}")
    print("\nRun Boxer on it:")
    print("  python run_boxer.py --input SUNRGBD --max_n 5")


if __name__ == "__main__":
    main()
