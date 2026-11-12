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

## Layout

- `third_party/OpenPCDet` — git submodule, pinned; don't edit.
- `data/` — KITTI calib/labels/images locally (gitignored).
- `results/` — detection zips unpacked here, tracker outputs.
