"""KITTI-360 adapter.

Pose source: `data_poses/<drive>/poses.txt` (frame_id + 3x4 in IMU/pose frame).
LiDAR→IMU: calib_cam_to_pose @ inv(calib_cam_to_velo).
Point clouds: `data_3d_raw/2013_05_28_drive_<seq:04d>_sync/velodyne_points/data/<frame:010d>.bin`.
Frame IDs are non-continuous.
"""

from __future__ import annotations

import os

import numpy as np

from ..adapter_base import BenchmarkAdapter


_DRIVE_SEQS = {0, 2, 4, 5, 6, 9}


class KITTI360Adapter(BenchmarkAdapter):
    name = "KITTI-360"
    default_pair_mode = "intra_seq_lc"
    has_native_int_ids = True

    def __init__(
        self,
        data_root: str,
        sequences: list[int] | None = None,
        ego_motion_correction: bool = False,
    ):
        self.data_root = data_root
        self._sequences = sorted(sequences if sequences is not None else list(_DRIVE_SEQS))
        self._ego_motion_correction = ego_motion_correction

        self._lidar_to_pose = self._load_lidar_to_pose()
        self._poses: dict[int, dict[int, np.ndarray]] = {}
        self._frames: dict[int, np.ndarray] = {}
        for seq in self._sequences:
            self._poses[seq], self._frames[seq] = self._load_poses(seq)

    def _load_lidar_to_pose(self) -> np.ndarray:
        cam_to_pose = np.eye(4)
        path = os.path.join(self.data_root, "calibration", "calib_cam_to_pose.txt")
        with open(path) as f:
            line = f.readline().strip()
            vals = line.split()[1:]  # skip "image_00:"
            cam_to_pose[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)

        cam_to_velo = np.eye(4)
        path = os.path.join(self.data_root, "calibration", "calib_cam_to_velo.txt")
        with open(path) as f:
            line = f.readline().strip()
            vals = line.split()
            cam_to_velo[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)

        return cam_to_pose @ np.linalg.inv(cam_to_velo)

    def _load_poses(self, seq: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
        drive_str = f"2013_05_28_drive_{seq:04d}_sync"
        pose_file = os.path.join(self.data_root, "data_poses", drive_str, "poses.txt")
        raw = np.genfromtxt(pose_file).reshape(-1, 13)
        frame_ids = raw[:, 0].astype(np.int64)
        poses: dict[int, np.ndarray] = {}
        for i in range(len(frame_ids)):
            T = np.eye(4)
            T[:3, :4] = raw[i, 1:].reshape(3, 4)
            poses[int(frame_ids[i])] = T @ self._lidar_to_pose
        return poses, frame_ids

    def list_sequences(self) -> list[int]:
        return list(self._sequences)

    def list_frames(self, seq: int) -> np.ndarray:
        return self._frames[seq]

    def get_position(self, seq: int, frame_id: int) -> np.ndarray:
        return self._poses[seq][frame_id][:3, 3]

    def relative_pose(
        self, seq_a: int, frame_a: int, seq_b: int, frame_b: int
    ) -> np.ndarray:
        T_a = self._poses[seq_a][frame_a]
        T_b = self._poses[seq_b][frame_b]
        return np.linalg.inv(T_b) @ T_a

    def read_pc(self, seq: int, frame_id: int) -> np.ndarray:
        drive_str = f"2013_05_28_drive_{seq:04d}_sync"
        path = os.path.join(
            self.data_root, "data_3d_raw", drive_str,
            "velodyne_points", "data", f"{frame_id:010d}.bin",
        )
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
