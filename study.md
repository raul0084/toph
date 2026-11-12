# Study guide: LiDAR, 3D detection and tracking

Reading list for the toph project (LiDAR 3D multi-object tracking on KITTI).

## Main recommendation

**Andreas Geiger — "Self-Driving Cars" lectures** (University of Tübingen, free on YouTube).
Geiger co-created KITTI, so the course is built around the same sensors and dataset we use.
It covers sensors and calibration, 3D detection, tracking and odometry at the right depth.
If you only do one thing, do this.

## A path, in order

### 1. How LiDAR and the KITTI rig work
- **"Vision meets Robotics: The KITTI Dataset"** — Geiger et al., IJRR 2013.
  Describes the exact car: Velodyne HDL-64E (64 beams, 10 Hz), cameras, GPS/IMU, and how
  they're calibrated. After reading it, the `calib` and `oxts` files will make sense.

### 2. Kalman filters and state estimation (core of `src/tracking/kalman.py`)
- **"Kalman and Bayesian Filters in Python"** — Roger Labbe. Free interactive Jupyter book on
  GitHub; builds intuition from the ground up with runnable code. Best way to understand Phase 2.
- **"Probabilistic Robotics"** — Thrun, Burgard & Fox. The classic, rigorous textbook on Bayes
  filters, sensor models and SLAM. For later.

### 3. Deep learning on point clouds (the detector on the GPU machine)
Read roughly in order — each fixes a weakness of the previous one:
- **PointNet** (Qi et al., 2017) — feeding an unordered set of points into a neural net.
- **VoxelNet** (Zhou & Tuzel, 2018), then **SECOND** (Yan et al., 2018) — 3D voxel grid +
  sparse 3D convolutions (the `spconv` library).
- **PointPillars** (Lang et al., 2019) — vertical "pillars" for speed. Our fallback model.
- **PV-RCNN** (Shi et al., 2020) — the model we run: voxel first pass + point-level refinement.
- **CenterPoint** (Yin et al., 2021) — objects as center points; modern default, also tracks.

### 4. 3D multi-object tracking
- **AB3DMOT** — "3D Multi-Object Tracking: A Baseline and New Evaluation Metrics",
  Weng et al., IROS 2020. Our tracker is modeled on it; read alongside `src/tracking/`.
- **SimpleTrack** (Pang et al., 2021) — which tracker design choices actually matter.
  Good once the baseline runs.
- **HOTA** (Luiten et al., IJCV 2020) — the metric for Phase 3 evaluation.

### 5. Point cloud geometry, odometry and SLAM (beyond detection)
- **Cyrill Stachniss's lectures** (University of Bonn, YouTube) — ICP, point cloud
  registration, SLAM.
- **KISS-ICP** (Vizzo et al., 2023) — LiDAR odometry with short, readable code. Hands-on way to
  estimate ego-motion without GPS.
- **Open3D tutorials** — load and visualize point clouds yourself (try the KITTI `.bin` files).

## How to fit it alongside the project
1. Now: the KITTI paper and AB3DMOT, plus the first chapters of Labbe's book while working on
   the tracker. Start the Geiger lectures in parallel.
2. When detection results come back: the detector papers (section 3), to understand why
   PV-RCNN misses certain objects.
3. Phase 3: HOTA. Then SimpleTrack and KISS-ICP when improving the tracker (ego-motion).
