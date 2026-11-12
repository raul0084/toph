"""Matching predicted tracks to new detections.

Boxes here are tracker-order [x, y, z, ry, l, w, h] in the KITTI camera frame
(bottom-center, y pointing down, ry = rotation about the vertical axis).

Why it works:
- After the Kalman prediction, each track's box should sit almost on top of its true detection.
  A similarity score (3D IoU, or negative center distance) turns that into a numeric table.
- The Hungarian algorithm (scipy.optimize.linear_sum_assignment) picks the one-to-one pairing with
  the best total score. Greedy "closest first" can steal a detection from the track that needed it
  in crowded scenes; Hungarian can't.
- Gating (rejecting pairs below a threshold) stops it from forcing a match between a track and an
  unrelated detection just because they are the least-bad option. Those become births/misses instead.
- IoU vs. distance: IoU is scale-aware and great for cars. For pedestrians (~0.6 m wide), a 0.3 m
  offset already drops IoU near zero, so center distance is the more forgiving metric.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


def bev_corners(boxes: np.ndarray) -> np.ndarray:
    """(N, 7) tracker boxes -> (N, 4, 2) footprint corners in the camera x-z (ground) plane."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    x, z, ry, l, w = b[:, 0], b[:, 2], b[:, 3], b[:, 4], b[:, 5]
    sx = np.array([1, 1, -1, -1]) / 2
    sz = np.array([1, -1, -1, 1]) / 2
    lx, lz = l[:, None] * sx, w[:, None] * sz
    c, s = np.cos(ry)[:, None], np.sin(ry)[:, None]
    # Same rotation as src.data.boxes.camera_box_corners: object x-axis -> (cos ry, -sin ry) in (x, z).
    return np.stack([x[:, None] + c * lx + s * lz, z[:, None] - s * lx + c * lz], axis=-1)


def iou_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Na, Nb) 3D IoU = BEV polygon intersection * vertical overlap / union volume."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 7)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 7)
    out = np.zeros((len(a), len(b)))
    if len(a) == 0 or len(b) == 0:
        return out
    pa = [Polygon(c) for c in bev_corners(a)]
    pb = [Polygon(c) for c in bev_corners(b)]
    # Cheap pre-filter: boxes whose centers are farther apart than their half-diagonals can't overlap.
    ra = np.hypot(a[:, 4], a[:, 5]) / 2
    rb = np.hypot(b[:, 4], b[:, 5]) / 2
    dist = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 2] - b[None, :, 2])
    vol_a, vol_b = a[:, 4] * a[:, 5] * a[:, 6], b[:, 4] * b[:, 5] * b[:, 6]
    for i, j in zip(*np.nonzero(dist < ra[:, None] + rb[None, :])):
        with np.errstate(invalid="ignore"):  # GEOS raises a spurious FP flag; area is correct
            inter_area = pa[i].intersection(pb[j]).area
        # Camera y points down: box spans [y - h, y].
        top = max(a[i, 1] - a[i, 6], b[j, 1] - b[j, 6])
        bottom = min(a[i, 1], b[j, 1])
        inter = inter_area * max(0.0, bottom - top)
        out[i, j] = inter / (vol_a[i] + vol_b[j] - inter)
    return out


def center_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Na, Nb) Euclidean distance between 3D box centers (geometric centers, not bottom)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 7)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 7)
    ca = a[:, :3] - np.c_[np.zeros(len(a)), a[:, 6] / 2, np.zeros(len(a))]
    cb = b[:, :3] - np.c_[np.zeros(len(b)), b[:, 6] / 2, np.zeros(len(b))]
    return np.linalg.norm(ca[:, None] - cb[None], axis=-1)


def similarity(tracks: np.ndarray, dets: np.ndarray, metric: str) -> np.ndarray:
    """Higher = better match. 'iou' -> 3D IoU, 'dist' -> negative center distance."""
    if metric == "iou":
        return iou_3d(tracks, dets)
    if metric == "dist":
        return -center_distance(tracks, dets)
    raise ValueError(f"unknown metric {metric!r}")


def associate(sim: np.ndarray, threshold: float):
    """Hungarian matching on a (n_tracks, n_dets) similarity matrix with gating.

    Returns (matches as (K, 2) array of [track_idx, det_idx], unmatched track idxs, unmatched det idxs).
    A pair is only kept if sim >= threshold (e.g. IoU >= 0.1, or -dist >= -2.0).
    """
    n_t, n_d = sim.shape
    if n_t == 0 or n_d == 0:
        return np.zeros((0, 2), dtype=int), np.arange(n_t), np.arange(n_d)
    valid = sim >= threshold
    # Gate BEFORE solving: every out-of-gate pair gets the same cost, worse than any valid pair.
    # Otherwise the solver minimizes total cost including hopeless pairs (e.g. 13 m vs 10 m) and
    # can sacrifice a good match to make an impossible one slightly cheaper.
    cost = np.where(valid, -sim, 0.0)
    if valid.any():
        cost[~valid] = cost[valid].max() + 1.0
    rows, cols = linear_sum_assignment(cost)
    keep = valid[rows, cols]
    matches = np.column_stack([rows[keep], cols[keep]]).astype(int)
    unmatched_t = np.setdiff1d(np.arange(n_t), matches[:, 0])
    unmatched_d = np.setdiff1d(np.arange(n_d), matches[:, 1])
    return matches, unmatched_t, unmatched_d
