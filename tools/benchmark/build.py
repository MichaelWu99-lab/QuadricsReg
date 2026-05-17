"""Benchmark builder CLI.

Usage:
    python -m tools.benchmark.build --dataset kitti --config configs/data_set/KITTI.yaml

Adapters are registered as `module:ClassName` strings and lazily imported,
so missing optional SDKs (waymo / nuscenes) don't break unrelated runs.
"""

from __future__ import annotations

import argparse
import importlib
import os

import yaml

from .pipeline import SamplingCfg, build_benchmark


ADAPTERS: dict[str, str] = {
    "kitti": "tools.benchmark.adapters.kitti:KittiAdapter",
    "kitti360": "tools.benchmark.adapters.kitti360:KITTI360Adapter",
    "apollo": "tools.benchmark.adapters.apollo:ApolloAdapter",
    "waymo": "tools.benchmark.adapters.waymo:WaymoAdapter",
    "nuscenes": "tools.benchmark.adapters.nuscenes:NuScenesAdapter",
}


def _load_adapter(name: str, cfg_dataset: dict, bench_cfg: dict, repo_root: str):
    if name not in ADAPTERS:
        raise SystemExit(
            f"unknown dataset '{name}'. registered: {list(ADAPTERS)}"
        )
    module_path, cls_name = ADAPTERS[name].split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    if name == "kitti":
        poses_root = cfg_dataset.get("poses_root") or os.path.join(
            repo_root, "benchmarks", "sem_kitti_poses"
        )
        sequences = bench_cfg.get("sequences")
        return cls(
            data_root=cfg_dataset["data_root"],
            poses_root=poses_root,
            sequences=sequences,
        )
    if name == "kitti360":
        sequences = bench_cfg.get("sequences")
        ego_corr = bench_cfg.get("ego_motion_correction", False)
        return cls(
            data_root=cfg_dataset["data_root"],
            sequences=sequences,
            ego_motion_correction=ego_corr,
        )
    if name == "apollo":
        sequences = bench_cfg.get("sequences")
        return cls(
            data_root=cfg_dataset["data_root"],
            sequences=sequences,
        )
    if name == "waymo":
        sequences = bench_cfg.get("sequences")
        lidar_select = bench_cfg.get("lidar_select", "top")
        return cls(
            data_root=cfg_dataset["data_root"],
            sequences=sequences,
            lidar_select=lidar_select,
        )
    if name == "nuscenes":
        sequences = bench_cfg.get("sequences")
        return cls(
            data_root=cfg_dataset["data_root"],
            sequences=sequences,
        )
    raise SystemExit(f"loader for '{name}' not wired yet")


def _sampling_cfg_from_yaml(bench_cfg: dict) -> SamplingCfg:
    levels_raw = bench_cfg.get("distance_levels", {"0_10": [0, 10]})
    levels = {k: (float(v[0]), float(v[1])) for k, v in levels_raw.items()}
    frame_gap_raw = bench_cfg.get("frame_gap", [5, 20])
    return SamplingCfg(
        distance_levels=levels,
        sequences=bench_cfg.get("sequences"),
        sample_interval_m=float(bench_cfg.get("sample_interval_m", 2.0)),
        skip_frames=int(bench_cfg.get("skip_frames", 250)),
        icp_refine=bool(bench_cfg.get("icp_refine", True)),
        icp_voxel=float(bench_cfg.get("icp_voxel", 0.05)),
        icp_max_corr_dist=float(bench_cfg.get("icp_max_corr_dist", 0.2)),
        icp_max_iter=int(bench_cfg.get("icp_max_iter", 500)),
        pair_mode=bench_cfg.get("pair_mode", "intra_seq_lc"),
        frame_gap=(int(frame_gap_raw[0]), int(frame_gap_raw[1])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(ADAPTERS))
    parser.add_argument("--config", required=True, help="path to configs/data_set/<NAME>.yaml")
    parser.add_argument("--out", default=None, help="override output dir; default benchmarks/<NAME>_lc")
    parser.add_argument("--levels", default=None, help="comma-separated subset, e.g. 0_10,10_20")
    parser.add_argument("--no-icp", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(args.config) as f:
        full = yaml.safe_load(f)
    cfg_dataset = full.get("DATA_SET", {})
    bench_cfg = full.get("benchmark", {})

    adapter = _load_adapter(args.dataset, cfg_dataset, bench_cfg, repo_root)
    cfg = _sampling_cfg_from_yaml(bench_cfg)
    if args.no_icp:
        cfg.icp_refine = False
    if args.levels:
        wanted = set(args.levels.split(","))
        cfg.distance_levels = {k: v for k, v in cfg.distance_levels.items() if k in wanted}

    name = cfg_dataset.get("name", args.dataset.upper())
    save_dir = args.out or os.path.join(repo_root, "benchmarks", f"{name}_lc")
    os.makedirs(save_dir, exist_ok=True)

    build_benchmark(adapter, cfg, save_dir)


if __name__ == "__main__":
    main()
