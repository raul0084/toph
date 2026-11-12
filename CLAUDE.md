# toph

LiDAR 3D multi-object tracking on KITTI Tracking, using OpenPCDet detections.

## Split between machines

- **GPU machine (a friend's PC, likely Windows → WSL2):** only runs detection. Entry points in
  `scripts/gpu/` (`windows.ps1` → `setup.sh` → `run_detection.py`). The operator is not
  technical: keep it a single command, idempotent/resumable, with friendly errors. Anything
  changed there must be pushed to `github.com/raul0084/toph` `main` to take effect, since the
  one-liner pulls from GitHub.
- **This Mac (no CUDA):** everything else — calibration/FOV filtering, tracking (`src/tracking`),
  evaluation (`src/eval`), visualization (`src/viz`), data loading (`src/data`). Never add code
  here that requires CUDA or importing `pcdet`.

## Detection output contract (GPU → Mac)

`results/<model>/<split>/<seq>.txt`, columns `frame class_id score x y z dx dy dz heading`,
boxes in the **LiDAR frame**, OpenPCDet convention. `class_id` indexes `meta.json["class_names"]`
(`Car, Pedestrian, Cyclist`). No FOV crop or score filtering is applied on the GPU side
(model's internal score threshold only). Change this contract in `run_detection.py` and here together.

## Tracking (`src/tracking`)

AB3DMOT-style, in the KITTI **camera frame**, per class. Tracker box order is
`[x, y, z, ry, l, w, h]` (bottom-center, y down) — note this differs from KITTI label order
`h, w, l, x, y, z, ry`; `track_sequence` converts at the boundary.
- `kalman.py` constant-velocity KF (dt = 1 frame), heading treated as ambiguous up to pi.
- `association.py` 3D IoU (shapely BEV × height) or center distance; Hungarian with gating applied
  *before* solving (out-of-gate pairs get a constant cost — see regression test).
- `tracker.py` `track_sequence(dets_df, calib, ...)` -> KITTI tracking result DataFrame;
  `write_kitti_tracking` writes submission files. Per-class `ClassConfig` in `DEFAULT_CONFIGS`.
- No ego-motion compensation yet (oxts is available) — main remaining source of ID switches on turns.
- `scripts/track_gt_sanity.py` runs the tracker on noisy GT-derived detections (no detector needed).

## Environment & tests

Local env is conda `holocene` (Python 3.11). Run `python -m pytest` from the repo root.
Imports are `from src.<pkg>.<module> import ...` (pytest `pythonpath = ["."]`).

## Layout

- `third_party/OpenPCDet` — git submodule, pinned; don't edit.
- `data/` — KITTI files as unzipped (gitignored): `data_tracking_calib/<split>/calib`,
  `data_tracking_oxts/<split>/oxts`, `data_tracking_image_2/<split>/image_02`,
  `training/label_02`. Always resolve via `src/data/paths.py` (`kitti_dir`, `image_shape`).
- `results/` — detection zips unpacked here, tracker outputs.
