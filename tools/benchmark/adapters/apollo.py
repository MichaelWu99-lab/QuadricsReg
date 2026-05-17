"""Apollo-SouthBay adapter.

Pose source: `<session>/poses/gt_poses.txt` (idx, timestamp, x, y, z, qx, qy, qz, qw).
Poses are already in LiDAR frame (no extrinsic needed).
Point clouds: `<session>/pcds/<frame_id>.pcd`.
No semantic labels available.
"""

from __future__ import annotations

import os

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from ..adapter_base import BenchmarkAdapter


_SESSIONS = {
    20: "TestData/HighWay237/2018-10-12/",
    21: "TestData/SunnyvaleBigloop/2018-10-03/",
    22: "TestData/MathildaAVE/2018-10-12/",
    23: "TestData/SanJoseDowntown/2018-10-11/2/",
    24: "TestData/SanJoseDowntown/2018-10-11/1/",
    25: "TestData/BaylandsToSeafood/2018-10-12/",
    26: "TestData/ColumbiaPark/2018-10-11/",
}


class ApolloAdapter(BenchmarkAdapter):
    name = "Apollo-SouthBay"
    default_pair_mode = "intra_seq_lc"
    has_native_int_ids = True

    def __init__(
        self,
        data_root: str,
        sequences: list[int] | None = None,
    ):
        self.data_root = data_root
        self._sequences = sorted(sequences if sequences is not None else list(_SESSIONS.keys()))

        self._poses: dict[int, dict[int, np.ndarray]] = {}
        self._frames: dict[int, np.ndarray] = {}
        for seq in self._sequences:
            self._poses[seq], self._frames[seq] = self._load_poses(seq)

    def _load_poses(self, seq: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
        session_path = _SESSIONS[seq]
        pose_file = os.path.join(self.data_root, session_path, "poses", "gt_poses.txt")
        raw = np.genfromtxt(pose_file)
        frame_ids = raw[:, 0].astype(np.int64)
        poses: dict[int, np.ndarray] = {}
        for i in range(len(frame_ids)):
            T = np.eye(4)
            T[:3, :3] = R.from_quat(raw[i, 5:9]).as_matrix()
            T[:3, 3] = raw[i, 2:5]
            poses[int(frame_ids[i])] = T
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
        session_path = _SESSIONS[seq]
        pcd_path = os.path.join(self.data_root, session_path, "pcds", f"{frame_id}.pcd")
        pcd = o3d.io.read_point_cloud(pcd_path)
        return np.asarray(pcd.points, dtype=np.float32)
