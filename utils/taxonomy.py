#! /usr/bin/env python3
# pyre-ignore-all-errors

import os
from typing import List, Union

# Directory containing label files (same directory as this module)
_LABELS_DIR = os.path.dirname(os.path.abspath(__file__))


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
    txt_path = os.path.join(_LABELS_DIR, f"{first_label}_labels.txt")
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
BOXY_NAME2SEM = {val: key for key, val in BOXY_SEM2NAME.items()}


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

# Viridis colormap: uniformly spaced samples from matplotlib's viridis for each category.
# Generated via: [matplotlib.cm.viridis(i / (N-1))[:3] for i in range(N)]
# Categories ordered alphabetically for deterministic assignment.
_VIRIDIS_CATEGORIES = sorted([k for k in SSI_COLORS.keys() if k not in ("Invalid",)])
# Pre-sampled viridis values for 25 categories (N=25, uniform spacing).
_VIRIDIS_25 = [
    (0.267, 0.004, 0.329),
    (0.283, 0.073, 0.397),
    (0.282, 0.140, 0.458),
    (0.268, 0.204, 0.505),
    (0.244, 0.267, 0.535),
    (0.217, 0.326, 0.551),
    (0.191, 0.382, 0.556),
    (0.166, 0.435, 0.557),
    (0.144, 0.486, 0.553),
    (0.127, 0.534, 0.545),
    (0.120, 0.580, 0.533),
    (0.134, 0.625, 0.512),
    (0.178, 0.665, 0.480),
    (0.243, 0.701, 0.440),
    (0.319, 0.733, 0.394),
    (0.402, 0.761, 0.343),
    (0.490, 0.785, 0.287),
    (0.581, 0.805, 0.226),
    (0.672, 0.821, 0.162),
    (0.761, 0.833, 0.104),
    (0.845, 0.840, 0.068),
    (0.920, 0.843, 0.066),
    (0.978, 0.840, 0.118),
    (0.993, 0.861, 0.217),
    (0.993, 0.906, 0.144),
]
SSI_COLORS_VIRIDIS = {"Invalid": (0.40, 0.40, 0.40)}
for _i, _name in enumerate(_VIRIDIS_CATEGORIES):
    SSI_COLORS_VIRIDIS[_name] = _VIRIDIS_25[_i % len(_VIRIDIS_25)]

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

EVL_TAXONOMY_V1_5 = {  # id to (class, names)
    0: (
        "chair",
        [
            "chair",
            "rocking chair",
            "dining chair",
            "deck chair",
            "office chair",
            "stool",
            "ottoman",
            "folding chair",
            "foot stool",
            "beanbag chair",
            "barber chair",
            "high chair",
            "swing",
            "massage chair",
        ],
    ),
    1: (
        "wallart",
        [
            "picture_frame",
            "picture frame",
            "wallart",
            "wall_art",
            "wall art",
            "clock",
            "mount",
            "wall artwork",
        ],
    ),
    2: ("sofa", ["sofa", "couch", "armchair", "reclining chair"]),
    3: ("lamp", ["lamp", "light", "lighting"]),
    4: ("window", ["window", "curtain", "window location", "window opening"]),
    5: ("table", ["table", "chairtable", "side table", "desk"]),
    6: ("bed", ["bed"]),
    7: (
        "detic",
        ["detic"],
    ),  # change "pillow" to "detic" since detic detects pillow too
    8: ("floor_mat", ["floor_mat", "floor mat", "carpet", "rug", "area rug"]),
    9: ("mirror", ["mirror", "reflective surface"]),
    10: (
        "plant",
        ["plant", "flower_pot", "flower pot", "vase", "house plant", "christmas tree"],
    ),
    11: (
        "storage",
        [
            "storage",
            "dresser",
            "cabinet",
            "shelf",
            "vanity",
            "clothes_rack",
            "clothes rack",
            "cabinet",
            "shelf",
            "countertop",
            "counter top",
            "clothing",
        ],
    ),
    12: ("container", ["container", "jar", "bottle", "bowl", "cup", "mug"]),
    13: (
        "door",
        [
            "door",
            "sliding door",
            "door location",
            "door opening",
            "sliding door opening",
        ],
    ),
    14: (
        "screen",
        [
            "screen",
            "monitor",
            "tv",
            "laptop",
            "television",
            "computer monitor",
            "computer laptop",
            "laptop computer",
        ],
    ),
    15: ("sink", ["sink"]),
    16: ("toilet", ["toilet"]),
    17: ("washerdryer", ["washerdryer", "washer", "dryer"]),
    18: ("refrigerator", ["refrigerator", "refridgerator", "fridge"]),
    19: (
        "other_appliance",
        [
            "other_appliance",
            "appliance",
            "stove",
            "air_conditioner",
            "air conditioner",
            "small appliance",
            "large appliance",
            "range",
        ],
    ),
    20: (
        "other_small_object",
        [
            "other_small_object",
            "plate",
            "exercise_weight",
            "electronic_device",
            "cutlery",
            "battery_charger",
            "candler_holder",
            "food object",
            "decorative accessory",
            "fabric accessory",
            "book",
            "kitchen utensil",
            "christmas decoration",
        ],
    ),
    21: (
        "other_medium_object",
        [
            "other_medium_object",
            "computer",
            "fan",
            "unknown",
            "living thing",
            "trash can",
            "decoration",
            "decoration - other",
            "object",
            "object - other",
            "unlabeled",
        ],
    ),
    22: (
        "other_big_object",
        [
            "other_big_object",
            "cart",
            "tent",
            "ladder",
            "island",
            "fireplace",
            "elevator",
            "ghost",
            "adacent room",
            "room",
            "dry erase board",
            "whiteboard",
            "reflective",
            "garage door",
            "garage door location",
            "garage door opening",
            "sliding door location",
            "furniture",
            "furniture - other",
            "structural - other",
        ],
    ),
}
EVL_REMAP = {
    0: 8,  # chair
    1: 19,  # wallart
    2: 7,  # sofa
    3: 12,  # lamp
    4: 6,  # window
    5: 9,  # table
    6: 11,  # bed
    7: 32,  # detic->anything
    8: 32,  # floor_mat->anything
    9: 21,  # mirror
    10: 13,  # plant
    11: 14,  # storage
    12: 32,  # container->anything
    13: 5,  # door
    14: 10,  # screen
    15: 27,  # sink
    16: 28,  # toilet
    17: 29,  # washerdryer
    18: 17,  # refrigerator
    19: 32,  # other_appliance->anything
    20: 32,  # other_small_object->anything
    21: 32,  # other_medium_object->anything
    22: 32,  # other_big_object->anything
}
