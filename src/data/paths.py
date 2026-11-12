"""Locate KITTI Tracking files. Supports the layout the zips unpack to (data/data_tracking_calib/...,
data/training/label_02) and a flat data/kitti_tracking/<split>/<kind> layout."""
from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

_CANDIDATES = {
    "calib": ["data_tracking_calib/{split}/calib", "kitti_tracking/{split}/calib"],
    "label_02": ["training/label_02", "data_tracking_label_2/{split}/label_02", "kitti_tracking/{split}/label_02"],
    "oxts": ["data_tracking_oxts/{split}/oxts", "kitti_tracking/{split}/oxts"],
    "image_02": ["data_tracking_image_2/{split}/image_02", "kitti_tracking/{split}/image_02"],
}


def kitti_dir(kind: str, split: str = "training", root: Path = DATA_ROOT) -> Path:
    for pattern in _CANDIDATES[kind]:
        if kind == "label_02" and pattern == "training/label_02" and split != "training":
            continue
        p = root / pattern.format(split=split)
        if p.is_dir():
            return p
    raise FileNotFoundError(f"KITTI {kind} for {split} not found under {root}")


def sequences(split: str = "training", root: Path = DATA_ROOT) -> list[str]:
    return sorted(p.stem for p in kitti_dir("calib", split, root).glob("*.txt"))


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
