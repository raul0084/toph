"""KITTI Tracking file formats: calibration, labels, and our GPU-side detection files.

Coordinate frames:
    LiDAR:  x forward, y left, z up. OpenPCDet boxes: center (x, y, z), dims (dx=l, dy=w, dz=h), heading.
    Camera (rectified): x right, y down, z forward. KITTI boxes: bottom-center (x, y, z), dims (h, w, l), ry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
    "frame", "track_id", "type", "truncated", "occluded", "alpha",
    "x1", "y1", "x2", "y2", "h", "w", "l", "x", "y", "z", "ry",
]
DETECTION_COLUMNS = ["frame", "class_id", "score", "x", "y", "z", "dx", "dy", "dz", "heading"]


@dataclass(frozen=True)
class Calib:
    P2: np.ndarray           # (3, 4) left color camera projection (rectified)
    R_rect: np.ndarray       # (3, 3) rectification rotation
    Tr_velo_cam: np.ndarray  # (3, 4) LiDAR -> unrectified camera

    @classmethod
    def from_file(cls, path: str | Path) -> Calib:
        """Parse a KITTI Tracking calib file ("P2:", "R_rect", "Tr_velo_cam")."""
        raw = {}
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                raw[parts[0].rstrip(":")] = np.array(parts[1:], dtype=np.float64)
        return cls(
            P2=raw["P2"].reshape(3, 4),
            R_rect=raw["R_rect"].reshape(3, 3),
            Tr_velo_cam=raw["Tr_velo_cam"].reshape(3, 4),
        )

    def lidar_to_rect(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        return _hom(pts) @ (self.R_rect @ self.Tr_velo_cam).T

    def rect_to_img(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project rectified camera points to pixels. Returns (uv (N, 2), depth (N,))."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        uvw = _hom(pts) @ self.P2.T
        return uvw[:, :2] / uvw[:, 2:3], uvw[:, 2]


def _hom(pts: np.ndarray) -> np.ndarray:
    return np.hstack([pts, np.ones((len(pts), 1))])


def load_labels(path: str | Path) -> pd.DataFrame:
    """Load a KITTI Tracking label_02/<seq>.txt (or a tracker result file with a trailing score)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    cols = LABEL_COLUMNS + (["score"] if df.shape[1] == len(LABEL_COLUMNS) + 1 else [])
    if df.shape[1] != len(cols):
        raise ValueError(f"{path}: expected {len(LABEL_COLUMNS)} or {len(LABEL_COLUMNS) + 1} columns, got {df.shape[1]}")
    df.columns = cols
    return df


def load_detections(path: str | Path, class_names: list[str]) -> pd.DataFrame:
    """Load a detection file written by scripts/gpu/run_detection.py (LiDAR frame)."""
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=DETECTION_COLUMNS)
    df["frame"] = df["frame"].astype(int)
    df["class_id"] = df["class_id"].astype(int)
    df["type"] = np.asarray(class_names, dtype=object)[df["class_id"].to_numpy()]
    return df
