import numpy as np
import pandas as pd
import pytest

from src.data.kitti import Calib
from src.tracking.association import associate, center_distance, iou_3d
from src.tracking.kalman import BoxKalmanFilter, _align_heading
from src.tracking.tracker import ClassConfig, track_sequence, write_kitti_tracking

CALIB = Calib(
    P2=np.array([[700.0, 0, 620, 0], [0, 700, 187, 0], [0, 0, 1, 0]]),
    R_rect=np.eye(3),
    Tr_velo_cam=np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=float),
)
CAR = [4.0, 2.0, 1.5]  # l, w, h


def box(x, z, ry=0.0, y=1.0, l=4.0, w=2.0, h=1.5):
    return np.array([x, y, z, ry, l, w, h])


# ------------------------------------------------------------------ Kalman filter

def test_kf_learns_constant_velocity():
    rng = np.random.default_rng(0)
    v = np.array([0.5, 0.0, 1.2])  # m/frame
    kf = BoxKalmanFilter(box(0, 10))
    for t in range(1, 40):
        kf.predict()
        truth = box(0 + v[0] * t, 10 + v[2] * t)
        kf.update(truth + rng.normal(0, [0.1, 0.05, 0.1, 0.02, 0.05, 0.05, 0.05]))
    np.testing.assert_allclose(kf.velocity, v, atol=0.05)
    pred = kf.predict()
    np.testing.assert_allclose(pred[[0, 2]], [0.5 * 40, 10 + 1.2 * 40], atol=0.2)


def test_kf_smooths_noise():
    errs_meas, errs_kf = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        kf = BoxKalmanFilter(box(0, 20))
        for t in range(1, 60):
            kf.predict()
            truth = box(0, 20 + t)
            meas = truth + rng.normal(0, [0.2, 0, 0.2, 0, 0, 0, 0])
            kf.update(meas)
            if t > 10:
                errs_meas.append(abs(meas[2] - truth[2]))
                errs_kf.append(abs(kf.box[2] - truth[2]))
    assert np.mean(errs_kf) < 0.75 * np.mean(errs_meas)


def test_heading_flip_is_absorbed():
    assert _align_heading(np.pi, 0.05) == pytest.approx(0.0, abs=1e-9)
    assert _align_heading(0.1, 0.0) == pytest.approx(0.1)
    kf = BoxKalmanFilter(box(0, 10, ry=0.0))
    kf.predict()
    kf.update(box(0, 10, ry=np.pi))  # detector flipped front/back
    assert abs(kf.box[3]) < 0.1


# ------------------------------------------------------------------ IoU / association

def test_iou_identical_and_disjoint():
    a = box(0, 10)
    assert iou_3d([a], [a])[0, 0] == pytest.approx(1.0)
    assert iou_3d([a], [box(10, 10)])[0, 0] == 0.0
    assert iou_3d([a], [a + [0, 5, 0, 0, 0, 0, 0]])[0, 0] == 0.0  # stacked vertically, no overlap


@pytest.mark.parametrize("other", [
    box(2, 10),                  # shifted half a length along x (ry=0 -> length along x)
    box(0, 10, y=1.75),          # shifted down by half the height
    box(0, 10, ry=np.pi / 2),    # rotated 90 deg: 2x2 footprint overlap
])
def test_iou_known_partial_overlap(other):
    # Each case: intersection volume 6, each box 12 -> IoU = 6 / 18.
    assert iou_3d([box(0, 10)], [other])[0, 0] == pytest.approx(1 / 3, abs=1e-6)


def test_center_distance_uses_geometric_center():
    assert center_distance([box(0, 10)], [box(3, 14)])[0, 0] == pytest.approx(5.0)


def test_associate_hungarian_and_gating():
    sim = np.array([[0.9, 0.8], [0.85, 0.05]])
    # Greedy would give track0->det0 then track1 only has det1 (0.05). Hungarian maximizes total.
    m, ut, ud = associate(sim, threshold=0.1)
    assert sorted(map(tuple, m)) == [(0, 1), (1, 0)]
    m, ut, ud = associate(np.array([[0.05]]), threshold=0.1)
    assert len(m) == 0 and ut.tolist() == [0] and ud.tolist() == [0]


def test_associate_gates_before_solving():
    # Regression: tracks at x=0 and x=3, detections at x=0.5 and x=-10, 1.5 m gate.
    # Minimizing raw total distance prefers {A->-10 (10 m), B->0.5 (2.5 m)} = 12.5 over
    # {A->0.5, B->-10} = 13.5, then both pairs fail the gate and the good 0.5 m match is lost.
    tracks = np.array([box(0, 10), box(3, 10)])
    dets = np.array([box(0.5, 10), box(-10, 10)])
    from src.tracking.association import similarity
    m, ut, ud = associate(similarity(tracks, dets, "dist"), threshold=-1.5)
    assert m.tolist() == [[0, 0]]
    assert ut.tolist() == [1] and ud.tolist() == [1]


# ------------------------------------------------------------------ tracker

def _run(objects, n_frames, cls, config, noise, drop, seed=0):
    """objects: {gt_id: fn(frame) -> box}. drop: {gt_id: set(frames)}. Returns (results, gt table)."""
    rng = np.random.default_rng(seed)
    dets, gt = [], []
    for f in range(n_frames):
        for gid, fn in objects.items():
            b = fn(f)
            gt.append((f, gid, b))
            if f in drop.get(gid, ()):
                continue
            n = b + rng.normal(0, noise)
            dets.append(dict(frame=f, type=cls, score=0.9, x=n[0], y=n[1], z=n[2], ry=n[3],
                             l=n[4], w=n[5], h=n[6]))
    res = track_sequence(pd.DataFrame(dets), CALIB, n_frames=n_frames, configs={cls: config})
    return res, gt


def _ids_per_object(res, gt, max_dist=1.0):
    ids = {}
    for f, gid, b in gt:
        r = res[res.frame == f]
        if r.empty:
            continue
        d = np.hypot(r.x - b[0], r.z - b[2])
        if d.min() < max_dist:
            ids.setdefault(gid, []).append(int(r.track_id.iloc[int(np.argmin(d.to_numpy()))]))
    return ids


NOISE = [0.1, 0.03, 0.1, 0.03, 0.05, 0.03, 0.03]


def test_parallel_cars_with_dropout_keep_ids():
    objects = {
        1: lambda f: box(-1.6, 5 + 1.0 * f),   # two cars in adjacent lanes, 3.2 m apart
        2: lambda f: box(1.6, 8 + 1.2 * f),
    }
    res, gt = _run(objects, 40, "Car", ClassConfig(metric="iou", threshold=0.1, max_age=3),
                   NOISE, drop={1: {15, 16}})
    ids = _ids_per_object(res, gt)
    assert set(ids) == {1, 2}
    assert all(len(set(v)) == 1 for v in ids.values()), ids
    assert ids[1][0] != ids[2][0]
    # Car 1 is reported again right after the dropout (no re-confirmation delay).
    assert {17, 18} <= set(res[res.track_id == ids[1][0]].frame)


def test_crossing_pedestrians_keep_ids():
    # Two pedestrians walking toward each other, passing 1.2 m apart; one is missed for 2 frames
    # right before they pass.
    ped = dict(l=0.8, w=0.6, h=1.7)
    objects = {
        1: lambda f: box(-6 + 0.3 * f, 12.0, ry=0.0, **ped),
        2: lambda f: box(6 - 0.3 * f, 13.2, ry=np.pi, **ped),
    }
    res, gt = _run(objects, 40, "Pedestrian", ClassConfig(metric="dist", threshold=-1.0, max_age=3),
                   [0.05, 0.02, 0.05, 0.05, 0.02, 0.02, 0.02], drop={2: {18, 19}})
    ids = _ids_per_object(res, gt, max_dist=0.5)
    assert all(len(set(v)) == 1 for v in ids.values()), ids
    assert ids[1][0] != ids[2][0]


def test_false_positive_is_not_reported():
    objects = {1: lambda f: box(0, 10 + f)}
    rng = np.random.default_rng(0)
    res, _ = _run(objects, 20, "Car", ClassConfig(min_hits=3), NOISE, drop={})
    n_ids = res.track_id.nunique()
    # Inject a one-frame ghost far away: it must never be reported (needs 3 consecutive hits).
    dets = [dict(frame=f, type="Car", score=0.9, x=0, y=1, z=10 + f, ry=0, l=4, w=2, h=1.5) for f in range(20)]
    dets.append(dict(frame=10, type="Car", score=0.9, x=15, y=1, z=30, ry=0, l=4, w=2, h=1.5))
    res = track_sequence(pd.DataFrame(dets), CALIB, n_frames=20, configs={"Car": ClassConfig(min_hits=3)})
    assert n_ids == 1 and res.track_id.nunique() == 1
    assert not ((res.frame == 10) & (res.x > 10)).any()


def test_writer_format(tmp_path):
    dets = pd.DataFrame([dict(frame=f, type="Car", score=0.8, x=0.0, y=1.0, z=10.0 + f, ry=-1.57,
                              l=4, w=2, h=1.5) for f in range(5)])
    res = track_sequence(dets, CALIB, configs={"Car": ClassConfig()})
    p = tmp_path / "0000.txt"
    write_kitti_tracking(res, p)
    lines = p.read_text().splitlines()
    assert len(lines) == 5
    parts = lines[0].split()
    assert len(parts) == 18 and parts[2] == "Car" and parts[1] == "1"
