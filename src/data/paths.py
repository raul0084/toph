"""Locate KITTI Tracking files in the layout the zips unpack to (data/data_tracking_calib/...,
data/training/label_02)."""
from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

_LAYOUT = {
    "calib": "data_tracking_calib/{split}/calib",
    "label_02": "{split}/label_02",  # labels only exist for training
    "oxts": "data_tracking_oxts/{split}/oxts",
    "image_02": "data_tracking_image_2/{split}/image_02",
}


def kitti_dir(kind: str, split: str = "training", root: Path = DATA_ROOT) -> Path:
    p = root / _LAYOUT[kind].format(split=split)
    if not p.is_dir():
        raise FileNotFoundError(f"KITTI {kind} for {split} not found at {p}")
    return p


@lru_cache(maxsize=None)
def image_shape(seq: str, split: str = "training", root: Path = DATA_ROOT) -> tuple[int, int] | None:
    """(height, width) of a sequence's images, read from the first PNG header. None if no images."""
    try:
        first = min(kitti_dir("image_02", split, root).joinpath(seq).glob("*.png"))
    except (FileNotFoundError, ValueError):
        return None
    with open(first, "rb") as f:
        header = f.read(24)
    width, height = struct.unpack(">II", header[16:24])
    return height, width
