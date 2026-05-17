"""Convert nuScenes dataset to KITTI-like layout.

Output layout (expected by NuScenesAdapter):
  <output_root>/
    <seq_int>/
      velodyne/<frame:06d>.bin   (float32, Nx5: x,y,z,intensity,ring)
      poses.txt                  (Nx12, each row is 3x4 flattened, LiDAR world pose)
      scene_token.txt            (one line: original scene token)
      sample_tokens.txt          (N lines: one sample token per frame)

Usage:
    pip install nuscenes-devkit
    python tools/benchmark/convert_nuscenes.py \
        --dataroot /path/to/nuscenes/v1.0-trainval \
        --output_dir /path/to/nuscenes_kitti_format/ \
        [--version v1.0-trainval] \
        [--max_scenes 100]

Notes:
    - Extracts LIDAR_TOP samples only.
    - LiDAR world pose = ego_pose @ calibrated_sensor (sensor-to-ego).
    - Each scene becomes one integer-named subdirectory.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
except ImportError:
    raise ImportError(
        "This script requires nuscenes-devkit.\n"
        "Install with: pip install nuscenes-devkit"
    )


def _get_lidar_world_pose(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """Compute 4x4 LiDAR-to-world pose for a given sample."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)

    # sensor-to-ego
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    T_ego_sensor = np.eye(4)
    T_ego_sensor[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    T_ego_sensor[:3, 3] = np.array(cs["translation"])

    # ego-to-world
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    T_world_ego = np.eye(4)
    T_world_ego[:3, :3] = Quaternion(ep["rotation"]).rotation_matrix
    T_world_ego[:3, 3] = np.array(ep["translation"])

    return T_world_ego @ T_ego_sensor


def _get_lidar_path(nusc: NuScenes, sample_token: str) -> str:
    """Get the file path for LIDAR_TOP of a sample."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)
    return os.path.join(nusc.dataroot, sd["filename"])


def convert_scene(nusc: NuScenes, scene_token: str, output_dir: str, seq_id: int) -> None:
    """Convert one scene to KITTI-like layout."""
    scene = nusc.get("scene", scene_token)
    seq_dir = os.path.join(output_dir, str(seq_id))
    velo_dir = os.path.join(seq_dir, "velodyne")
    os.makedirs(velo_dir, exist_ok=True)

    # Iterate through all samples in the scene
    sample_token_cur = scene["first_sample_token"]
    poses = []
    sample_tokens = []
    frame_idx = 0

    while sample_token_cur:
        sample = nusc.get("sample", sample_token_cur)

        # Read point cloud
        lidar_path = _get_lidar_path(nusc, sample_token_cur)
        pc = LidarPointCloud.from_file(lidar_path)
        # pc.points is (4, N): x, y, z, intensity
        # nuScenes .pcd.bin is actually (N, 5): x, y, z, intensity, ring
        # Read raw to preserve ring channel
        raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

        # Save as bin
        bin_path = os.path.join(velo_dir, f"{frame_idx:06d}.bin")
        raw.tofile(bin_path)

        # Compute LiDAR world pose
        pose = _get_lidar_world_pose(nusc, sample_token_cur)
        poses.append(pose[:3, :].flatten())
        sample_tokens.append(sample_token_cur)

        # Next sample
        sample_token_cur = sample["next"] if sample["next"] else None
        frame_idx += 1

    # Write poses.txt
    poses_arr = np.array(poses)
    np.savetxt(os.path.join(seq_dir, "poses.txt"), poses_arr, fmt="%.10f")

    # Write scene_token.txt
    with open(os.path.join(seq_dir, "scene_token.txt"), "w") as f:
        f.write(scene_token)

    # Write sample_tokens.txt
    with open(os.path.join(seq_dir, "sample_tokens.txt"), "w") as f:
        for t in sample_tokens:
            f.write(t + "\n")

    print(f"  seq {seq_id}: {frame_idx} frames, scene={scene['name']}")


def main():
    parser = argparse.ArgumentParser(description="Convert nuScenes to KITTI-like layout")
    parser.add_argument("--dataroot", required=True, help="nuScenes dataroot (e.g. /data/nuscenes/v1.0-trainval)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version string")
    parser.add_argument("--max_scenes", type=int, default=None, help="Max number of scenes to convert")
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} from {args.dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    scenes = nusc.scene
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    print(f"Converting {len(scenes)} scenes ...")
    os.makedirs(args.output_dir, exist_ok=True)

    for seq_id, scene in enumerate(scenes):
        print(f"[{seq_id + 1}/{len(scenes)}] {scene['name']}")
        convert_scene(nusc, scene["token"], args.output_dir, seq_id)

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
