import numpy as np
import pytest

from src.data.boxes import (
    camera_box_corners,
    camera_boxes_to_2d,
    detections_to_kitti,
    in_camera_fov,
    lidar_boxes_to_camera,
    observation_angle,
)
from src.data.kitti import Calib, load_detections, load_labels

# Real values from KITTI Tracking training/calib/0000.txt (tracking format: no colons on R_rect/Tr_*).
CALIB_TRACKING = """\
P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
R_rect 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_cam 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_velo 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01
"""


@pytest.fixture
def real_calib(tmp_path):
    p = tmp_path / "0000.txt"
    p.write_text(CALIB_TRACKING)
    return Calib.from_file(p)


@pytest.fixture
def ideal_calib():
    """LiDAR axes mapped exactly onto camera axes, no offsets."""
    f, cx, cy = 700.0, 620.0, 187.0
    return Calib(
        P2=np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]]),
        R_rect=np.eye(3),
        Tr_velo_cam=np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=float),
    )


def lidar_corners(boxes):
    """Independent reference: corners of OpenPCDet LiDAR boxes (rotation about z)."""
    out = []
    for x, y, z, dx, dy, dz, th in boxes:
        c, s = np.cos(th), np.sin(th)
        pts = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    lx, ly = sx * dx / 2, sy * dy / 2
                    pts.append([x + c * lx - s * ly, y + s * lx + c * ly, z + sz * dz / 2])
        out.append(pts)
    return np.array(out)


def test_parse_tracking_calib(real_calib):
    assert real_calib.P2.shape == (3, 4)
    assert real_calib.R_rect[0, 0] == pytest.approx(0.9999239)
    assert real_calib.Tr_velo_cam[2, 3] == pytest.approx(-0.2717806)


def test_parse_object_calib(tmp_path):
    text = (CALIB_TRACKING.replace("R_rect ", "R0_rect: ")
            .replace("Tr_velo_cam ", "Tr_velo_to_cam: "))
    p = tmp_path / "obj.txt"
    p.write_text(text)
    assert Calib.from_file(p).R_rect[0, 1] == pytest.approx(9.837760e-03)


def test_lidar_to_rect_axes(ideal_calib):
    np.testing.assert_allclose(ideal_calib.lidar_to_rect([10, 2, 1]), [[-2, -1, 10]])


def test_box_conversion_bottom_center_and_heading(ideal_calib):
    # Car 10 m ahead, center 1 m up, facing forward (+x LiDAR == +z camera).
    cam = lidar_boxes_to_camera([[10, 0, 1, 4, 2, 1.5, 0]], ideal_calib)[0]
    x, y, z, h, w, l, ry = cam
    assert (x, z) == pytest.approx((0, 10))
    assert y == pytest.approx(-(1 - 0.75))  # bottom at LiDAR z=0.25 -> camera y=-0.25 (y is down)
    assert (h, w, l) == pytest.approx((1.5, 2, 4))
    assert ry == pytest.approx(-np.pi / 2)
    corners = camera_box_corners(cam)[0]
    assert corners[[0, 1, 4, 5], 2] == pytest.approx([12] * 4)  # front face points forward


@pytest.mark.parametrize("calib_name", ["ideal_calib", "real_calib"])
def test_corners_match_independent_lidar_corners(calib_name, request):
    calib = request.getfixturevalue(calib_name)
    rng = np.random.default_rng(0)
    boxes = np.column_stack([
        rng.uniform(5, 40, 20), rng.uniform(-10, 10, 20), rng.uniform(-2, 1, 20),
        rng.uniform(0.5, 5, 20), rng.uniform(0.5, 2.5, 20), rng.uniform(1, 2, 20),
        rng.uniform(-np.pi, np.pi, 20),
    ])
    ours = camera_box_corners(lidar_boxes_to_camera(boxes, calib))
    ref = np.stack([calib.lidar_to_rect(c) for c in lidar_corners(boxes)])
    # The real calib has a ~0.015 rad tilt between LiDAR z and camera y; KITTI camera boxes are
    # upright in the camera frame, so corners differ by up to ~0.015 * box size there.
    tol = 1e-9 if calib_name == "ideal_calib" else 0.1
    for a, b in zip(ours, ref):
        dist = np.linalg.norm(a[:, None] - b[None], axis=-1)
        assert dist.min(axis=1).max() < tol
        assert len(set(dist.argmin(axis=1))) == 8  # one-to-one


def test_projection_and_fov(ideal_calib):
    cam = lidar_boxes_to_camera([
        [10, 0, 0.0, 4, 2, 1.5, 0],     # straight ahead, centered at camera height
        [-10, 0, 0.75, 4, 2, 1.5, 0],   # behind
        [5, 30, 0.75, 4, 2, 1.5, 0],    # far left, outside image
    ], ideal_calib)
    assert in_camera_fov(cam, ideal_calib).tolist() == [True, False, False]
    x1, y1, x2, y2 = camera_boxes_to_2d(cam[:1], ideal_calib)[0]
    assert (x1 + x2) / 2 == pytest.approx(620, abs=1)
    assert x1 < 620 < x2 and y1 < 187 < y2


def test_fov_keeps_truncated_box(ideal_calib):
    # Center projects left of the image (u < 0), but the right side of the box is visible.
    # Image half-width 620 px at f=700 -> edge at x/z = -0.886. Center x/z = -1.0, near edge visible.
    cam = np.array([[-6.0, 1.0, 6.0, 1.5, 2.0, 4.0, 0.0]])  # [x, y, z, h, w, l, ry]
    uv, _ = ideal_calib.rect_to_img(cam[:, :3])
    assert uv[0, 0] < 0
    assert in_camera_fov(cam, ideal_calib).tolist() == [True]


def test_alpha_straight_ahead(ideal_calib):
    cam = lidar_boxes_to_camera([[10, 0, 0, 4, 2, 1.5, 0]], ideal_calib)
    assert observation_angle(cam)[0] == pytest.approx(-np.pi / 2)


def test_load_labels(tmp_path):
    p = tmp_path / "0000.txt"
    p.write_text(
        "0 -1 DontCare -1 -1 -10 219.31 188.49 245.50 218.56 -1000 -1000 -1000 -10 -1 -1 -1\n"
        "0 0 Van 0 0 -1.793451 296.744956 161.752147 455.226042 292.372804 2.000000 1.823255 4.433886 -4.552284 1.858523 13.410495 -2.115488\n"
    )
    df = load_labels(p)
    assert len(df) == 2 and df.loc[1, "type"] == "Van" and df.loc[1, "track_id"] == 0
    assert df.loc[1, "ry"] == pytest.approx(-2.115488)


def test_load_detections_roundtrip(tmp_path, ideal_calib):
    p = tmp_path / "0000.txt"
    p.write_text(
        "# frame class_id score x y z dx dy dz heading\n"
        "0 0 0.9500 10.0 0.0 0.75 4.0 2.0 1.5 0.0\n"
        "0 1 0.6000 -8.0 1.0 0.9 0.8 0.6 1.8 0.0\n"
        "3 2 0.4000 15.0 -1.0 0.8 1.8 0.6 1.7 1.0\n"
    )
    dets = load_detections(p, ["Car", "Pedestrian", "Cyclist"])
    assert dets["type"].tolist() == ["Car", "Pedestrian", "Cyclist"]
    kitti = detections_to_kitti(dets, ideal_calib)
    assert kitti["type"].tolist() == ["Car", "Cyclist"]  # pedestrian behind the car is dropped
    assert {"x1", "y2", "alpha", "ry", "lidar_heading"} <= set(kitti.columns)
