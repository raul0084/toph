"""Sanity check the tracker on real KITTI motion without a detector.

Turns ground-truth labels into fake detections (Gaussian noise, ~10% random misses, occasional
heading flips, random false positives), runs the tracker, and reports how well GT identities are
preserved. This isolates the tracker: if IDs break here, it's the tracker's fault, not the detector's.

Not a benchmark metric (that's HOTA in Phase 3) - just "does it keep IDs on real trajectories?".

Usage: python scripts/track_gt_sanity.py [--seqs 0000 0001 ...] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.kitti import Calib, load_labels  # noqa: E402
from src.data.paths import image_shape, kitti_dir  # noqa: E402
from src.tracking.tracker import DEFAULT_CONFIGS, track_sequence  # noqa: E402

CLASSES = list(DEFAULT_CONFIGS)
BOX = ["x", "y", "z", "h", "w", "l", "ry"]
TYPICAL_DIMS = {"Car": (1.5, 1.6, 3.9), "Pedestrian": (1.75, 0.65, 0.85), "Cyclist": (1.75, 0.6, 1.75)}


def fake_detections(gt: pd.DataFrame, n_frames: int, rng: np.random.Generator,
                    pos_std=0.15, drop=0.10, flip=0.05, fp_rate=0.5) -> pd.DataFrame:
    d = gt[gt["type"].isin(CLASSES)].copy()
    d = d[rng.random(len(d)) > drop]
    n = len(d)
    d["x"] += rng.normal(0, pos_std, n)
    d["z"] += rng.normal(0, pos_std, n)
    d["y"] += rng.normal(0, pos_std / 3, n)
    for c in ["h", "w", "l"]:
        d[c] *= 1 + rng.normal(0, 0.03, n)
    d["ry"] += rng.normal(0, 0.05, n) + np.pi * (rng.random(n) < flip)
    d["score"] = rng.uniform(0.5, 1.0, n)

    fps = []
    for f in range(n_frames):
        for cls in CLASSES:
            for _ in range(rng.poisson(fp_rate / len(CLASSES))):
                h, w, l = TYPICAL_DIMS[cls]
                fps.append(dict(frame=f, type=cls, x=rng.uniform(-15, 15), y=1.7, z=rng.uniform(5, 50),
                                h=h, w=w, l=l, ry=rng.uniform(-np.pi, np.pi), score=rng.uniform(0.3, 0.7)))
    return pd.concat([d[["frame", "type", "score"] + BOX], pd.DataFrame(fps)], ignore_index=True)


def id_stats(gt: pd.DataFrame, res: pd.DataFrame, max_dist=1.0) -> dict:
    """Frame-by-frame Hungarian matching GT <-> tracker output (same class, center distance)."""
    gt = gt[gt["type"].isin(CLASSES)]
    assigned: dict[int, list[int]] = {}
    n_gt = n_matched = n_fp = 0
    res_by_frame = {f: g for f, g in res.groupby("frame")}
    for f, g in gt.groupby("frame"):
        r = res_by_frame.get(f, res.iloc[:0])
        n_gt += len(g)
        matched_r = set()
        for cls in CLASSES:
            gc, rc = g[g["type"] == cls], r[r["type"] == cls]
            if gc.empty or rc.empty:
                continue
            dist = np.hypot(gc["x"].to_numpy()[:, None] - rc["x"].to_numpy()[None],
                            gc["z"].to_numpy()[:, None] - rc["z"].to_numpy()[None])
            rows, cols = linear_sum_assignment(dist)
            for i, j in zip(rows, cols):
                if dist[i, j] < max_dist:
                    assigned.setdefault(int(gc["track_id"].iloc[i]), []).append(int(rc["track_id"].iloc[j]))
                    matched_r.add(rc.index[j])
                    n_matched += 1
        n_fp += len(r) - len(matched_r)
    for f, r in res_by_frame.items():
        if f not in set(gt["frame"]):
            n_fp += len(r)
    switches = sum(int(np.sum(np.diff(v) != 0)) for v in assigned.values())
    constant = np.mean([len(set(v)) == 1 for v in assigned.values()]) if assigned else float("nan")
    return dict(gt_objects=len(assigned), id_constant=constant, id_switches=switches,
                coverage=n_matched / max(n_gt, 1), fp_outputs=n_fp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", nargs="+", default=["0000", "0001", "0004", "0011", "0016", "0019"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = []
    for seq in args.seqs:
        rng = np.random.default_rng(args.seed)
        calib = Calib.from_file(kitti_dir("calib") / f"{seq}.txt")
        gt = load_labels(kitti_dir("label_02") / f"{seq}.txt")
        gt = gt[gt["type"] != "DontCare"]
        n_frames = int(gt["frame"].max()) + 1
        dets = fake_detections(gt, n_frames, rng)
        res = track_sequence(dets, calib, n_frames=n_frames, image_shape=image_shape(seq) or (375, 1242))
        stats = id_stats(gt, res)
        stats.update(seq=seq, frames=n_frames, fake_dets=len(dets))
        rows.append(stats)

    df = pd.DataFrame(rows)[["seq", "frames", "fake_dets", "gt_objects", "id_constant",
                             "id_switches", "coverage", "fp_outputs"]]
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tot_obj = df["gt_objects"].sum()
    print(f"\nALL: {tot_obj} GT objects, ID constant for {np.average(df.id_constant, weights=df.gt_objects):.1%}, "
          f"{df.id_switches.sum()} ID switches, coverage {np.average(df.coverage, weights=df.frames):.1%}")


if __name__ == "__main__":
    main()
