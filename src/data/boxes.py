"""3D box geometry: LiDAR <-> KITTI camera boxes, corners, image projection, FOV filtering."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.data.kitti import Calib

# Most KITTI Tracking sequences are 1242x375; a few differ by a few pixels.
DEFAULT_IMAGE_SHAPE = (375, 1242)  # (height, width)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def lidar_boxes_to_camera(boxes: np.ndarray, calib: Calib) -> np.ndarray:
    """(N, 7) [x, y, z, dx, dy, dz, heading] LiDAR center boxes ->
    (N, 7) [x, y, z, h, w, l, ry] KITTI camera bottom-center boxes.

    Same math as OpenPCDet's boxes3d_lidar_to_kitti_camera, reordered to KITTI's h, w, l.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    l, w, h, heading = boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6]
    bottom = boxes[:, :3].copy()
    bottom[:, 2] -= h / 2
    xyz = calib.lidar_to_rect(bottom)
    ry = wrap_angle(-heading - np.pi / 2)
    return np.column_stack([xyz, h, w, l, ry])


def camera_box_corners(boxes_cam: np.ndarray) -> np.ndarray:
    """(N, 7) [x, y, z, h, w, l, ry] -> (N, 8, 3) corners in the camera frame.
    Corners 0-3 are the bottom face, 4-7 the top; 0, 1, 4, 5 are the front (+l/2) side."""
    b = np.asarray(boxes_cam, dtype=np.float64).reshape(-1, 7)
    h, w, l, ry = b[:, 3], b[:, 4], b[:, 5], b[:, 6]
    sx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) / 2
    sy = np.array([0, 0, 0, 0, -1, -1, -1, -1])
    sz = np.array([1, -1, -1, 1, 1, -1, -1, 1]) / 2
    local = np.stack([l[:, None] * sx, h[:, None] * sy, w[:, None] * sz], axis=-1)  # (N, 8, 3)
    c, s = np.cos(ry), np.sin(ry)
    # Rotation about camera y: object x-axis -> (cos ry, 0, -sin ry).
    R = np.zeros((len(b), 3, 3))
    R[:, 0, 0], R[:, 0, 2] = c, s
    R[:, 1, 1] = 1
    R[:, 2, 0], R[:, 2, 2] = -s, c
    return local @ R.transpose(0, 2, 1) + b[:, None, :3]


def camera_boxes_to_2d(boxes_cam: np.ndarray, calib: Calib,
                       image_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE) -> np.ndarray:
    """Project 3D boxes and return (N, 4) [x1, y1, x2, y2] clipped to the image."""
    corners = camera_box_corners(boxes_cam)
    n = len(corners)
    if n == 0:
        return np.zeros((0, 4))
    uv, _ = calib.rect_to_img(corners.reshape(-1, 3))
    uv = uv.reshape(n, 8, 2)
    hgt, wid = image_shape
    x1 = np.clip(uv[..., 0].min(1), 0, wid - 1)
    y1 = np.clip(uv[..., 1].min(1), 0, hgt - 1)
    x2 = np.clip(uv[..., 0].max(1), 0, wid - 1)
    y2 = np.clip(uv[..., 1].max(1), 0, hgt - 1)
    return np.column_stack([x1, y1, x2, y2])


def observation_angle(boxes_cam: np.ndarray) -> np.ndarray:
    """KITTI alpha: ry minus the viewing-ray angle to the box."""
    b = np.asarray(boxes_cam).reshape(-1, 7)
    return wrap_angle(b[:, 6] - np.arctan2(b[:, 0], b[:, 2]))


def in_camera_fov(boxes_cam: np.ndarray, calib: Calib,
                  image_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE) -> np.ndarray:
    """Boolean mask: some part of the box is visible in the image.

    Uses the projected extent of the corners that are in front of the camera, not just the center:
    KITTI labels truncated objects whose center lies outside the image (~5% of training GT), and
    a center-only test would discard exactly those.
    """
    corners = camera_box_corners(boxes_cam)
    n = len(corners)
    if n == 0:
        return np.zeros(0, dtype=bool)
    uv, depth = calib.rect_to_img(corners.reshape(-1, 3))
    uv, front = uv.reshape(n, 8, 2), depth.reshape(n, 8) > 0
    hgt, wid = image_shape
    u = np.where(front, uv[..., 0], np.nan)
    v = np.where(front, uv[..., 1], np.nan)
    with warnings.catch_warnings():  # rows with no corner in front are all-NaN; masked below
        warnings.simplefilter("ignore", RuntimeWarning)
        overlaps = ((np.nanmax(u, 1) >= 0) & (np.nanmin(u, 1) < wid)
                    & (np.nanmax(v, 1) >= 0) & (np.nanmin(v, 1) < hgt))
    return front.any(1) & overlaps


def detections_to_kitti(dets: pd.DataFrame, calib: Calib,
                        image_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE) -> pd.DataFrame:
    """Add KITTI camera-frame columns (x, y, z, h, w, l, ry, alpha, 2D bbox) to LiDAR detections
    and drop boxes outside the camera FOV.

    Input: DataFrame from kitti.load_detections. LiDAR columns are kept with a `lidar_` prefix.
    """
    lidar = dets[["x", "y", "z", "dx", "dy", "dz", "heading"]].to_numpy()
    cam = lidar_boxes_to_camera(lidar, calib)
    out = dets.rename(columns={c: f"lidar_{c}" for c in ["x", "y", "z", "dx", "dy", "dz", "heading"]}).copy()
    out[["x", "y", "z", "h", "w", "l", "ry"]] = cam
    out["alpha"] = observation_angle(cam)
    out[["x1", "y1", "x2", "y2"]] = camera_boxes_to_2d(cam, calib, image_shape)
    return out[in_camera_fov(cam, calib, image_shape)].reset_index(drop=True)
