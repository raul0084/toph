"""AB3DMOT-style 3D multi-object tracker (Weng et al., 2020), run independently per class.

Per frame:  predict every track -> associate with detections -> update matched tracks ->
            start tentative tracks from unmatched detections -> delete tracks unseen for too long.

Why the birth/death rules work:
- min_hits: a detector fires false positives, but rarely on the same spot several frames in a row.
  Requiring a few consecutive matches before reporting a track filters most of them out.
- max_age: true objects are sometimes missed (occlusion, sparse points far away). Keeping an
  unmatched track alive for a few frames, coasting on its predicted motion, lets it re-acquire the
  object with the *same ID* instead of starting a new one (an ID switch).
- Classes are tracked separately: a car never matches a pedestrian, and each class gets its own
  association metric/threshold suited to its size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.boxes import camera_boxes_to_2d, observation_angle, DEFAULT_IMAGE_SHAPE
from src.data.kitti import Calib
from src.tracking.association import associate, similarity
from src.tracking.kalman import BoxKalmanFilter

RESULT_COLUMNS = [
    "frame", "track_id", "type", "truncated", "occluded", "alpha",
    "x1", "y1", "x2", "y2", "h", "w", "l", "x", "y", "z", "ry", "score",
]


@dataclass
class ClassConfig:
    metric: str = "iou"         # 'iou' or 'dist'
    threshold: float = 0.1      # IoU >= threshold, or -dist >= threshold (e.g. -2.0 for 2 m)
    min_hits: int = 3           # consecutive matches before a track is reported
    max_age: int = 2            # frames a track survives without a match
    score_threshold: float = 0.0  # detections below this are ignored


DEFAULT_CONFIGS = {
    "Car": ClassConfig(metric="iou", threshold=0.1),
    "Pedestrian": ClassConfig(metric="dist", threshold=-1.5),
    "Cyclist": ClassConfig(metric="dist", threshold=-2.0),
}


@dataclass
class Track:
    id: int
    kf: BoxKalmanFilter
    score: float
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    age: int = 0
    confirmed: bool = False  # set once hit_streak reaches min_hits; never revoked


@dataclass
class ClassTracker:
    """Tracks one object class. Boxes in/out are tracker order [x, y, z, ry, l, w, h], camera frame."""
    config: ClassConfig = field(default_factory=ClassConfig)
    id_counter: count = field(default_factory=lambda: count(1))
    tracks: list[Track] = field(default_factory=list)
    frame_count: int = 0

    def step(self, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, np.ndarray, float]]:
        """Advance one frame. Returns [(track_id, box, score)] for tracks reported this frame."""
        cfg = self.config
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        keep = scores >= cfg.score_threshold
        boxes, scores = boxes[keep], scores[keep]
        self.frame_count += 1

        preds = np.array([t.kf.predict() for t in self.tracks]).reshape(-1, 7)
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        matches, _, unmatched_d = associate(similarity(preds, boxes, cfg.metric), cfg.threshold)

        for ti, di in matches:
            t = self.tracks[ti]
            t.kf.update(boxes[di])
            t.score = scores[di]
            t.hits += 1
            t.hit_streak = t.hit_streak + 1 if t.time_since_update == 1 else 1
            t.time_since_update = 0
            t.confirmed |= t.hit_streak >= cfg.min_hits
        for t in self.tracks:
            if t.time_since_update > 0:
                t.hit_streak = 0

        for di in unmatched_d:
            self.tracks.append(Track(id=next(self.id_counter), kf=BoxKalmanFilter(boxes[di]),
                                     score=scores[di]))

        out = [
            (t.id, t.kf.box, t.score) for t in self.tracks
            if t.time_since_update == 0
            # Confirmed tracks stay reported after a dropout (same ID, no re-confirmation).
            # Early frames: report immediately, otherwise the start of every sequence is lost.
            and (t.confirmed or t.hit_streak >= cfg.min_hits or self.frame_count <= cfg.min_hits)
        ]
        self.tracks = [t for t in self.tracks if t.time_since_update <= cfg.max_age]
        return out


def track_sequence(dets: pd.DataFrame, calib: Calib, n_frames: int | None = None,
                   configs: dict[str, ClassConfig] | None = None,
                   image_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE) -> pd.DataFrame:
    """Run per-class trackers over a sequence of KITTI camera-frame detections.

    dets: DataFrame with columns frame, type, score, x, y, z, h, w, l, ry (e.g. from
          src.data.boxes.detections_to_kitti). Classes without a config are ignored.
    Returns a DataFrame in KITTI tracking result format (RESULT_COLUMNS). Track IDs are unique
    across classes.
    """
    configs = configs or DEFAULT_CONFIGS
    ids = count(1)  # shared so IDs are unique across classes
    trackers = {c: ClassTracker(config=cfg, id_counter=ids) for c, cfg in configs.items()}
    if n_frames is None:
        n_frames = int(dets["frame"].max()) + 1 if len(dets) else 0
    by_frame = {f: g for f, g in dets.groupby("frame")}

    rows = []
    for frame in range(n_frames):
        g = by_frame.get(frame)
        for cls, trk in trackers.items():
            d = g[g["type"] == cls] if g is not None else None
            if d is None or d.empty:
                boxes, scores = np.zeros((0, 7)), np.zeros(0)
            else:
                boxes = d[["x", "y", "z", "ry", "l", "w", "h"]].to_numpy()
                scores = d["score"].to_numpy()
            for tid, box, score in trk.step(boxes, scores):
                rows.append((frame, tid, cls, box, score))

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    boxes = np.array([r[3] for r in rows])  # tracker order
    cam = boxes[:, [0, 1, 2, 6, 5, 4, 3]]   # -> [x, y, z, h, w, l, ry]
    bb = camera_boxes_to_2d(cam, calib, image_shape)
    out = pd.DataFrame({
        "frame": [r[0] for r in rows], "track_id": [r[1] for r in rows], "type": [r[2] for r in rows],
        "truncated": -1, "occluded": -1, "alpha": observation_angle(cam),
        "x1": bb[:, 0], "y1": bb[:, 1], "x2": bb[:, 2], "y2": bb[:, 3],
        "h": cam[:, 3], "w": cam[:, 4], "l": cam[:, 5],
        "x": cam[:, 0], "y": cam[:, 1], "z": cam[:, 2], "ry": cam[:, 6],
        "score": [r[4] for r in rows],
    })
    return out[RESULT_COLUMNS]


def write_kitti_tracking(results: pd.DataFrame, path: str | Path) -> None:
    """Write results in the KITTI tracking submission format (one line per box, space separated)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fmt = {"alpha": 6, "x1": 2, "y1": 2, "x2": 2, "y2": 2, "h": 4, "w": 4, "l": 4,
           "x": 4, "y": 4, "z": 4, "ry": 4, "score": 4}
    lines = []
    for r in results.sort_values(["frame", "track_id"]).itertuples(index=False):
        d = r._asdict()
        parts = [str(int(d["frame"])), str(int(d["track_id"])), d["type"],
                 str(int(d["truncated"])), str(int(d["occluded"]))]
        parts += [f"{d[c]:.{p}f}" for c, p in fmt.items()]
        lines.append(" ".join(parts))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))
