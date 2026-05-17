"""Benchmark sampling / refining / writing pipeline.

Currently implements:
  - intra_seq_lc: same-sequence loop-closure pairs (KITTI / KITTI-360 /
    Apollo style). KDTree neighbor query within a translation range, with
    skip_frames and minimum spatial spacing between adjacent kept pairs.

Pair modes intra_seq_ode and cross_seq_lc are stubs for now; they get
filled in when the Waymo / nuScenes adapters land.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from tqdm import tqdm

from .adapter_base import BenchmarkAdapter


@dataclass
class SamplingCfg:
    distance_levels: dict[str, tuple[float, float]]
    sequences: list[int] | None = None
    sample_interval_m: float = 2.0
    skip_frames: int = 250
    icp_refine: bool = True
    icp_voxel: float = 0.05
    icp_max_corr_dist: float = 0.2
    icp_max_iter: int = 500
    pair_mode: str = "intra_seq_lc"
    frame_gap: tuple[int, int] = (5, 20)


def _icp_refine(
    src_xyz: np.ndarray, tgt_xyz: np.ndarray, T_init: np.ndarray, cfg: SamplingCfg
) -> np.ndarray:
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_xyz[:, :3].astype(np.float64))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_xyz[:, :3].astype(np.float64))
    src = src.voxel_down_sample(cfg.icp_voxel)
    tgt = tgt.voxel_down_sample(cfg.icp_voxel)
    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        cfg.icp_max_corr_dist,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iter),
    )
    return np.asarray(reg.transformation)


def _sample_intra_seq_lc(
    adapter: BenchmarkAdapter, seq: int, lo: float, hi: float, cfg: SamplingCfg
) -> list[tuple[int, int, int]]:
    """Return list of (seq, frame_i, frame_j) for one sequence, one level."""
    frames = np.asarray(adapter.list_frames(seq))
    if len(frames) == 0:
        return []
    positions = np.stack([adapter.get_position(seq, int(f)) for f in frames], axis=0)
    tree = KDTree(positions)

    pairs: list[tuple[int, int, int]] = []
    last_pos_i = None
    last_pos_j = None
    for idx_i in tqdm(range(len(frames)), desc=f"{adapter.name} seq {seq}"):
        neighbor_idxs = tree.query_radius(positions[idx_i : idx_i + 1], r=hi)[0]
        for idx_j in neighbor_idxs:
            idx_j = int(idx_j)
            if idx_j <= idx_i or idx_j - idx_i <= cfg.skip_frames:
                continue
            dist = float(np.linalg.norm(positions[idx_i] - positions[idx_j]))
            if dist <= lo or dist >= hi:
                continue
            if last_pos_i is not None:
                if (
                    np.linalg.norm(positions[idx_i] - last_pos_i) < cfg.sample_interval_m
                    or np.linalg.norm(positions[idx_j] - last_pos_j) < cfg.sample_interval_m
                ):
                    continue
            pairs.append((seq, int(frames[idx_i]), int(frames[idx_j])))
            last_pos_i = positions[idx_i]
            last_pos_j = positions[idx_j]
    return pairs


def _refine_and_pack(
    adapter: BenchmarkAdapter,
    raw_pairs: list[tuple[int, int, int]],
    cfg: SamplingCfg,
) -> list[list]:
    """For each (seq, i, j), compute (refined) 4x4 GT and flatten into a row."""
    rows: list[list] = []
    for seq, i, j in tqdm(raw_pairs, desc="refine"):
        T = adapter.relative_pose(seq, i, seq, j)
        if cfg.icp_refine:
            try:
                src = adapter.read_pc(seq, i)
                tgt = adapter.read_pc(seq, j)
                T = _icp_refine(src, tgt, T, cfg)
            except Exception as exc:
                print(f"[warn] ICP refine failed for seq={seq} {i}->{j}: {exc}")
        rows.append([seq, i, seq, j, *T.flatten().tolist()])
    return rows


def _write_txt(rows: list[list], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = "seq i seq_db j " + " ".join(f"mot{k}" for k in range(1, 17))
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            seq, i, seq_db, j, *mot = row
            f.write(f"{int(seq)} {int(i)} {int(seq_db)} {int(j)}")
            for v in mot:
                f.write(f" {float(v):f}")
            f.write("\n")


def _write_sidecar(adapter: BenchmarkAdapter, rows: list[list], path: str) -> None:
    if adapter.has_native_int_ids:
        return
    seq_map: dict[str, str] = {}
    frame_map: dict[str, dict[str, str]] = {}
    for row in rows:
        seq, i, seq_db, j = (int(x) for x in row[:4])
        for s in (seq, seq_db):
            seq_map.setdefault(str(s), adapter.native_seq_id(s))
            frame_map.setdefault(str(s), {})
        frame_map[str(seq)][str(i)] = adapter.native_frame_id(seq, i)
        frame_map[str(seq_db)][str(j)] = adapter.native_frame_id(seq_db, j)
    payload = {
        "dataset": adapter.name,
        "seq_id_to_native": seq_map,
        "frame_id_to_native": frame_map,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _sample_intra_seq_ode(
    adapter: BenchmarkAdapter, seq: int, lo: float, hi: float, cfg: SamplingCfg
) -> list[tuple[int, int, int]]:
    """Odometry-style pairs: same seq, frame gap in [gap_lo, gap_hi], dist in [lo, hi].

    Suitable for short segments (Waymo/nuScenes ~20s) where loop closures don't exist.
    """
    frames = np.asarray(adapter.list_frames(seq))
    if len(frames) == 0:
        return []
    positions = np.stack([adapter.get_position(seq, int(f)) for f in frames], axis=0)
    gap_lo, gap_hi = cfg.frame_gap

    pairs: list[tuple[int, int, int]] = []
    last_pos_i = None
    for idx_i in range(len(frames)):
        for idx_j in range(idx_i + gap_lo, min(idx_i + gap_hi + 1, len(frames))):
            dist = float(np.linalg.norm(positions[idx_i] - positions[idx_j]))
            if dist <= lo or dist >= hi:
                continue
            if last_pos_i is not None:
                if np.linalg.norm(positions[idx_i] - last_pos_i) < cfg.sample_interval_m:
                    continue
            pairs.append((seq, int(frames[idx_i]), int(frames[idx_j])))
            last_pos_i = positions[idx_i]
            break  # one pair per source frame
    return pairs


def build_benchmark(adapter: BenchmarkAdapter, cfg: SamplingCfg, save_dir: str) -> None:
    if cfg.pair_mode not in ("intra_seq_lc", "intra_seq_ode"):
        raise NotImplementedError(
            f"pair_mode={cfg.pair_mode} not implemented yet; "
            "cross_seq_lc will land later."
        )

    sequences = cfg.sequences if cfg.sequences is not None else adapter.list_sequences()

    for level_name, (lo, hi) in cfg.distance_levels.items():
        print(f"=== level {level_name}: dist ∈ ({lo}, {hi}) ===")
        raw: list[tuple[int, int, int]] = []
        for seq in sequences:
            if cfg.pair_mode == "intra_seq_lc":
                raw.extend(_sample_intra_seq_lc(adapter, int(seq), float(lo), float(hi), cfg))
            else:
                raw.extend(_sample_intra_seq_ode(adapter, int(seq), float(lo), float(hi), cfg))
        print(f"  raw pairs: {len(raw)}")
        rows = _refine_and_pack(adapter, raw, cfg)

        out_txt = os.path.join(save_dir, f"test_{level_name}.txt")
        _write_txt(rows, out_txt)
        print(f"  wrote {len(rows)} rows -> {out_txt}")

        if not adapter.has_native_int_ids:
            _write_sidecar(adapter, rows, os.path.join(save_dir, "_index.json"))
