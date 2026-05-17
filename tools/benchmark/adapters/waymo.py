"""Waymo adapter.

Expects the user to have pre-converted .tfrecord to KITTI-like format:
  <data_root>/
    <seq_int>/
      velodyne/<frame:06d>.bin   (float32, Nx4)
      poses.txt                  (Nx12, each row is 3x4 flattened)
      segment_id.txt             (one line: original segment string)

This avoids pulling in the heavy waymo-open-dataset-tf dependency at
benchmark generation time. A conversion script is documented in
docs/benchmark.md.

If the user has NOT pre-converted, they can install waymo-open-dataset and
use a future `tools/benchmark/convert_waymo.py` helper.
"""

from __future__ import annotations

import os

import numpy as np

from ..adapter_base import BenchmarkAdapter


class WaymoAdapter(BenchmarkAdapter):
    name = "waymo"
    default_pair_mode = "intra_seq_ode"
    has_native_int_ids = False
    optional_deps = ("waymo-open-dataset-tf-2-6-0",)

    def __init__(
        self,
        data_root: str,
        sequences: list[int] | None = None,
        lidar_select: str = "top",
    ):
        self.data_root = data_root
        self._lidar_select = lidar_select

        # Discover sequences: each subdirectory named as an int
        if sequences is not None:
            self._sequences = sorted(sequences)
        else:
            self._sequences = sorted(
                int(d) for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d)) and d.isdigit()
            )

        self._poses: dict[int, dict[int, np.ndarray]] = {}
        self._frames: dict[int, np.ndarray] = {}
        self._native_seq_ids: dict[int, str] = {}
        for seq in self._sequences:
            self._load_seq(seq)

    def _load_seq(self, seq: int) -> None:
        seq_dir = os.path.join(self.data_root, str(seq))
        pose_file = os.path.join(seq_dir, "poses.txt")
        raw = np.genfromtxt(pose_file).reshape(-1, 3, 4)
        n = raw.shape[0]
        self._frames[seq] = np.arange(n, dtype=np.int64)
        self._poses[seq] = {}
        for i in range(n):
            T = np.eye(4)
            T[:3, :] = raw[i]
            self._poses[seq][i] = T

        seg_file = os.path.join(seq_dir, "segment_id.txt")
        if os.path.exists(seg_file):
            with open(seg_file) as f:
                self._native_seq_ids[seq] = f.readline().strip()
        else:
            self._native_seq_ids[seq] = str(seq)

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
        path = os.path.join(
            self.data_root, str(seq), "velodyne", f"{frame_id:06d}.bin"
        )
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

    def native_seq_id(self, seq: int) -> str:
        return self._native_seq_ids.get(seq, str(seq))

    def native_frame_id(self, seq: int, frame_id: int) -> str:
        return str(frame_id)
