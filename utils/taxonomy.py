#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

import os
from typing import List, Union

# Directory containing label files (owl/ directory, sibling of utils/)
_LABELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "owl"
)


def load_text_labels(
    label_list: Union[str, List[str]], verbose: bool = False
) -> List[str]:
    """Load text labels from a plain-text file (one label per line).

    Args:
        label_list: Name of the label set to load (e.g. "lvisplus"),
                    or a list of custom label strings.
        verbose: If True, print loading messages.

    Returns:
        List of text label strings
    """
    if label_list is None:
        label_list = ["default"]
    elif isinstance(label_list, str):
        label_list = [label_list]

    first_label = label_list[0]
    txt_path = os.path.join(_LABELS_DIR, f"{first_label}_classes.csv")
    if not os.path.exists(txt_path):
        return label_list

    if verbose:
        print(f"Loading text labels from {txt_path}")
    with open(txt_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    if not labels:
        raise ValueError(
            f"Invalid or empty labels in {txt_path}. Expected one label per line."
        )

    return labels


BOXY_SEM2NAME = {
    -1: "Invalid",  # pad entries.
    0: "Unknown",  # dustbin
    2: "Floor",
    3: "Wall",
    5: "Door",
    6: "Window",
    7: "Couch",
    8: "Chair",
    9: "Table",
    10: "Screen",
    11: "Bed",
    12: "Lamp",
    13: "Plant",
    14: "Storage",
    17: "Refrigerator",
    19: "WallArt",
    21: "Mirror",
    # 26: "Detic",
    27: "Sink",
    28: "Toilet",
    29: "WasherDryer",
    31: "Stairs",
    32: "Anything",
}

SSI_SEM2NAME = {
    -1: "Invalid",
    0: "Unknown",
    1: "Other",
    2: "Floor",
    3: "Wall",
    4: "Ceiling",
    5: "Door",
    6: "Window",
    7: "Couch",
    8: "Chair",
    9: "Table",
    10: "Screen",
    11: "Bed",
    12: "Lamp",
    13: "Plant",
    14: "Storage",
    17: "Refrigerator",
    18: "Whiteboard",
    19: "WallArt",
    21: "Mirror",
    27: "Sink",
    28: "Toilet",
    29: "WasherDryer",
    31: "Stairs",
    32: "Anything",
}
SSI_NAME2SEM = {val: key for key, val in SSI_SEM2NAME.items()}

# //fbsource/arvr/projects/surreal/ar/slam/ssi/object_mapping/visualization/BoxVisualizationHelpers.cpp
SSI_COLORS = {
    "Unknown": (0.5, 0.5, 0.8),
    "Couch": (0.1, 0.5, 0.1),
    "Chair": (0.2, 0.6, 1.0),
    "Table": (1.0, 1.0, 0.0),
    "Bed": (0.55, 0.9, 0.0),
    "Floor": (1.0, 0.75, 0.75),
    "Lamp": (1.0, 0.8, 0.25),
    "Plant": (0.0, 1.0, 0.0),
    "WallArt": (0.6, 0.3, 0.95),
    "Mirror": (1.0, 0.6, 0.0),
    "Window": (0.5, 1.0, 1.0),
    "Door": (0.95, 0.25, 0.85),
    "Storage": (0.7, 0.4, 0.05),
    "Wall": (1.0, 1.0, 1.0),
    "Screen": (1.0, 0.0, 0.0),
    "Toilet": (0.7, 0.55, 0.1),  # mustard
    "WasherDryer": (0.7, 0.5, 0.5),  # gray red
    "Refrigerator": (0.5, 0.7, 0.6),  # gray green
    "Sink": (0.1, 0.45, 0.7),  # gray blue
    "Stairs": (0.25, 0.41, 0.88),  # blue ish
    "Island": (0.1, 0.2, 0.85),
    "Laptop": (0.9, 0.2, 0.3),
    "Anything": (0.55, 0.9, 0.55),
    "Invalid": (0.5, 0.5, 0.5),
    "Other": (0.5, 0.5, 1.0),
    "Ceiling": (0.5, 1.0, 0.5),  # green
    "Whiteboard": (0.3, 1.0, 0.2),  # light green
}

# Alternative color scheme: darker, saturated colors visible against white backgrounds.
# Uses perceptually distinct hues at ~60-70% lightness for good contrast on white.
SSI_COLORS_ALT = {
    "Unknown": (0.40, 0.40, 0.65),
    "Couch": (0.15, 0.55, 0.15),
    "Chair": (0.12, 0.47, 0.71),
    "Table": (0.70, 0.55, 0.00),
    "Bed": (0.40, 0.65, 0.00),
    "Floor": (0.74, 0.35, 0.35),
    "Lamp": (0.80, 0.52, 0.00),
    "Plant": (0.17, 0.63, 0.17),
    "WallArt": (0.50, 0.20, 0.80),
    "Mirror": (0.84, 0.44, 0.00),
    "Window": (0.00, 0.60, 0.60),
    "Door": (0.78, 0.15, 0.65),
    "Storage": (0.55, 0.30, 0.00),
    "Wall": (0.55, 0.55, 0.55),
    "Screen": (0.84, 0.15, 0.15),
    "Toilet": (0.58, 0.40, 0.00),
    "WasherDryer": (0.55, 0.30, 0.30),
    "Refrigerator": (0.25, 0.55, 0.40),
    "Sink": (0.10, 0.35, 0.60),
    "Stairs": (0.20, 0.35, 0.75),
    "Island": (0.10, 0.15, 0.70),
    "Laptop": (0.75, 0.15, 0.25),
    "Anything": (0.00, 0.80, 0.40),
    "Invalid": (0.40, 0.40, 0.40),
    "Other": (0.35, 0.35, 0.75),
    "Ceiling": (0.30, 0.65, 0.30),
    "Whiteboard": (0.20, 0.65, 0.15),
}

TEXT2COLORS = {
    "unknown": (0.5, 0.5, 0.8),
    "couch": (0.1, 0.5, 0.1),
    "sofa": (0.1, 0.5, 0.1),
    "chair": (0.2, 0.6, 1.0),
    "bench": (0.2, 0.6, 1.0),
    "table": (1.0, 1.0, 0.0),
    "desk": (1.0, 1.0, 0.0),
    "bed": (0.55, 0.9, 0.0),
    "mattress": (0.55, 0.9, 0.0),
    "floor": (1.0, 0.75, 0.75),
    "ground": (1.0, 0.75, 0.75),
    "lamp": (1.0, 0.8, 0.25),
    "light": (1.0, 0.8, 0.25),
    "plant": (0.0, 1.0, 0.0),
    "wallart": (0.6, 0.3, 0.95),
    "picture frame": (0.6, 0.3, 0.95),
    "photo": (0.6, 0.3, 0.95),
    "mirror": (1.0, 0.6, 0.0),
    "window frame": (0.5, 1.0, 1.0),
    "door frame": (0.95, 0.25, 0.85),
    "storage": (0.7, 0.4, 0.05),
    "cabinet": (0.7, 0.4, 0.05),
    "wall": (1.0, 1.0, 1.0),
    "screen": (1.0, 0.0, 0.0),
    "television": (1.0, 0.0, 0.0),
    "toilet": (0.7, 0.55, 0.1),  # mustard
    "washerdryer": (0.7, 0.5, 0.5),  # gray red
    "refrigerator": (0.5, 0.7, 0.6),  # gray green
    "sink": (0.1, 0.45, 0.7),  # gray blue
    "stairs": (0.25, 0.41, 0.88),  # blue ish
    "anything or other": (0.55, 0.9, 0.55),
    "anything": (0.55, 0.9, 0.55),
    "ceiling": (0.5, 1.0, 0.5),  # green
}
