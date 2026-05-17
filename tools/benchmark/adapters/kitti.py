"""KITTI adapter.

Pose source: SemanticKITTI's refined poses (`benchmarks/sem_kitti_poses/`),
which are numerically more accurate than the official KITTI release.
LiDAR pose = pose_in_cam @ Tr  (Tr from sequences/<seq>/calib.txt).
"""

from __future__ import annotations

import os

import numpy as np

from ..adapter_base import BenchmarkAdapter


class KittiAdapter(BenchmarkAdapter):
    name = "KITTI"
    default_pair_mode = "intra_seq_lc"
    has_native_int_ids = True

    def __init__(
        self,
        data_root: str,
        poses_root: str,
        sequences: list[int] | None = None,
    ):
        self.data_root = data_root
        self.poses_root = poses_root
        self._sequences = sequences if sequences is not None else list(range(11))

        self._tr: dict[int, np.ndarray] = {}
        self._poses: dict[int, dict[int, np.ndarray]] = {}
        self._frames: dict[int, np.ndarray] = {}
        for seq in self._sequences:
            self._tr[seq] = self._load_tr(seq)
            self._poses[seq], self._frames[seq] = self._load_poses(seq)

    def _load_tr(self, seq: int) -> np.ndarray:
        calib = os.path.join(self.data_root, "sequences", f"{seq:02d}", "calib.txt")
        with open(calib) as f:
            for line in f:
                if line.startswith("Tr:"):
                    vals = np.array(line.strip().split()[1:], dtype=np.float64)
                    Tr = np.eye(4)
                    Tr[:3, :] = vals.reshape(3, 4)
                    return Tr
        raise RuntimeError(f"`Tr:` not found in {calib}")

    def _load_poses(self, seq: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
        path = os.path.join(self.poses_root, f"{seq:02d}.txt")
        arr = np.genfromtxt(path).reshape(-1, 3, 4)
        poses: dict[int, np.ndarray] = {}
        for i in range(arr.shape[0]):
            T = np.eye(4)
            T[:3, :] = arr[i]
            poses[i] = T @ self._tr[seq]
        return poses, np.arange(arr.shape[0], dtype=np.int64)

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
        T = np.eye(4)
        T[:3, :3] = T_b[:3, :3].T @ T_a[:3, :3]
        T[:3, 3] = T_b[:3, :3].T @ (T_a[:3, 3] - T_b[:3, 3])
        return T

    def read_pc(self, seq: int, frame_id: int) -> np.ndarray:
        path = os.path.join(
            self.data_root, "sequences", f"{seq:02d}", "velodyne", f"{frame_id:06d}.bin"
        )
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
