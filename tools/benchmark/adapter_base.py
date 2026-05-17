"""Adapter base class for benchmark generation.

Each dataset only implements the data access hooks; the sampling /
pair-finding / ICP-refine / file IO live in `pipeline.py`.

The hooks intentionally take primitive ints for `seq` and `frame_id`.
For datasets whose native IDs are strings (Waymo segment, nuScenes
scene/sample tokens), the adapter maintains a bidirectional table and
exposes it via `native_seq_id` / `native_frame_id`. The pipeline then
writes a sidecar `_index.json` next to the benchmark `.txt`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BenchmarkAdapter(ABC):
    name: str = ""
    default_pair_mode: str = "intra_seq_lc"
    has_native_int_ids: bool = True
    optional_deps: tuple[str, ...] = ()

    @abstractmethod
    def list_sequences(self) -> list[int]:
        ...

    @abstractmethod
    def list_frames(self, seq: int) -> np.ndarray:
        ...

    @abstractmethod
    def get_position(self, seq: int, frame_id: int) -> np.ndarray:
        ...

    @abstractmethod
    def relative_pose(
        self, seq_a: int, frame_a: int, seq_b: int, frame_b: int
    ) -> np.ndarray:
        ...

    @abstractmethod
    def read_pc(self, seq: int, frame_id: int) -> np.ndarray:
        ...

    def native_seq_id(self, seq: int) -> str:
        return str(seq)

    def native_frame_id(self, seq: int, frame_id: int) -> str:
        return str(frame_id)
