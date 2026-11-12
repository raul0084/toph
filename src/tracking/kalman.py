"""Constant-velocity Kalman filter for one 3D box (AB3DMOT-style).

State (10): [x, y, z, ry, l, w, h, vx, vy, vz] in the KITTI camera frame, velocity in meters/frame.
Measurement (7): [x, y, z, ry, l, w, h] from the detector.

Why it works:
- At 10 Hz, objects move almost linearly between frames, so "position + velocity" predicts the next
  box to within tens of centimeters. That prediction is what makes association reliable.
- The filter keeps an uncertainty (covariance P) next to the state. A new track starts with large
  velocity uncertainty, so its first few measurements quickly *teach* it the velocity. A mature
  track trusts its prediction more and smooths out detector jitter.
- The update blends prediction and measurement with the Kalman gain K = P H^T (H P H^T + R)^-1:
  if the prediction is uncertain (P large) relative to measurement noise R, K -> 1 (trust detection);
  otherwise K -> 0 (trust motion). This is the optimal linear estimate under Gaussian noise.
- Size (l, w, h) and heading are modeled as constant: rigid objects don't change size, and heading
  changes slowly enough that the measurement update tracks it.
"""
from __future__ import annotations

import numpy as np

from src.data.boxes import wrap_angle

STATE_DIM, MEAS_DIM = 10, 7


def _align_heading(measured: float, predicted: float) -> float:
    """Return the measured heading, flipped by pi if needed, so it is within pi/2 of the prediction.

    LiDAR detectors often get the front/back of a box wrong (a car's point cloud looks nearly the
    same from both ends). A raw pi flip would look like an enormous innovation and wreck the filter,
    so we treat heading as ambiguous up to pi when updating. AB3DMOT does the same.
    """
    diff = wrap_angle(measured - predicted)
    if abs(diff) > np.pi / 2:
        measured = measured + np.pi
    return float(predicted + wrap_angle(measured - predicted))


class BoxKalmanFilter:
    def __init__(self, box: np.ndarray, *, meas_noise: np.ndarray | None = None,
                 process_noise: np.ndarray | None = None, init_vel_var: float = 10.0):
        """box: [x, y, z, ry, l, w, h]. Noise arguments are diagonal variances."""
        self.x = np.zeros(STATE_DIM)
        self.x[:MEAS_DIM] = box
        self.x[3] = wrap_angle(self.x[3])

        self.F = np.eye(STATE_DIM)
        self.F[0, 7] = self.F[1, 8] = self.F[2, 9] = 1.0  # x += vx, ... (dt = 1 frame)
        self.H = np.eye(MEAS_DIM, STATE_DIM)

        # Defaults: detector noise ~0.1-0.2 m in position, ~0.1 rad heading, ~0.1 m size.
        self.R = np.diag(meas_noise if meas_noise is not None
                         else [0.04, 0.01, 0.04, 0.01, 0.01, 0.01, 0.01])
        # Process noise: how much reality deviates from constant velocity per frame.
        # Velocity variance 1e-3 (m/frame)^2 ~ std 0.03 m/frame per frame ~ 3 m/s^2 of unmodeled
        # acceleration (typical car braking/accelerating). Trade-off, measured on synthetic data with
        # 0.2 m detector noise: KF position error is ~62% of raw detections at constant velocity and
        # ~82% under 3 m/s^2 acceleration. Larger values follow maneuvers better but smooth less.
        self.Q = np.diag(process_noise if process_noise is not None
                         else [1e-3, 5e-4, 1e-3, 0.01, 1e-4, 1e-4, 1e-4, 1e-3, 2.5e-4, 1e-3])
        self.P = np.diag(np.r_[np.diag(self.R), [init_vel_var] * 3])

    @property
    def box(self) -> np.ndarray:
        """Current [x, y, z, ry, l, w, h]."""
        return self.x[:MEAS_DIM].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[7:].copy()

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.x[3] = wrap_angle(self.x[3])
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box

    def update(self, box: np.ndarray) -> None:
        z = np.asarray(box, dtype=np.float64).copy()
        z[3] = _align_heading(z[3], self.x[3])
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        self.x[3] = wrap_angle(self.x[3])
        I_KH = np.eye(STATE_DIM) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form: stays symmetric PSD
