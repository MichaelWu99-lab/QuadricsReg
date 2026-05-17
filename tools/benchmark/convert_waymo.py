"""Convert Waymo .tfrecord files to KITTI-like layout.

Output layout (expected by WaymoAdapter):
  <output_root>/
    <seq_int>/
      velodyne/<frame:06d>.bin   (float32, Nx4: x,y,z,intensity)
      poses.txt                  (Nx12, each row is 3x4 flattened, vehicle world pose)
      segment_id.txt             (one line: original segment string)

Usage:
    pip install waymo-open-dataset-tf-2-6-0  # or compatible version
    python tools/benchmark/convert_waymo.py \
        --input_dir /path/to/waymo/training/ \
        --output_dir /path/to/waymo_kitti_format/ \
        [--max_segments 100] \
        [--lidar top]

Notes:
    - By default only extracts the TOP LiDAR (most points, 64-beam).
    - Each .tfrecord segment becomes one integer-named subdirectory.
    - Requires `waymo-open-dataset-tf-2-6-0` and TensorFlow.
"""

from __future__ import annotations

import argparse
import os
import glob

import numpy as np

try:
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils
except ImportError:
    raise ImportError(
        "This script requires waymo-open-dataset and tensorflow.\n"
        "Install with: pip install waymo-open-dataset-tf-2-6-0"
    )


LIDAR_NAMES = {
    "top": open_dataset.LaserName.TOP,
    "front": open_dataset.LaserName.FRONT,
    "side_left": open_dataset.LaserName.SIDE_LEFT,
    "side_right": open_dataset.LaserName.SIDE_RIGHT,
    "rear": open_dataset.LaserName.REAR,
}


def _extract_points_top(frame) -> np.ndarray:
    """Extract TOP LiDAR points as (N, 4) float32 [x, y, z, intensity]."""
    range_images, camera_projections, _, range_image_top_pose = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose,
        keep_polar_features=False,
    )
    # points is a list per LiDAR; index 0 = TOP
    pts = points[0]  # (N, 3)
    # intensity from range_images
    intensities = range_images[open_dataset.LaserName.TOP][0].numpy()
    # range_image shape: (H, W, 4) — channel 1 is intensity
    intensity_flat = intensities[:, :, 1].flatten()
    # Only keep points that have valid range (same mask as points[0])
    valid_mask = range_images[open_dataset.LaserName.TOP][0].numpy()[:, :, 0].flatten() > 0
    intensity_valid = intensity_flat[valid_mask].reshape(-1, 1)
    xyzi = np.hstack([pts, intensity_valid]).astype(np.float32)
    return xyzi


def _extract_points_simple(frame, lidar_name: str) -> np.ndarray:
    """Fallback: extract points using range image conversion for any LiDAR."""
    range_images, camera_projections, _, range_image_top_pose = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose,
        keep_polar_features=False,
    )
    lidar_idx = list(LIDAR_NAMES.values()).index(LIDAR_NAMES[lidar_name])
    pts = points[lidar_idx]
    intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
    return np.hstack([pts, intensity]).astype(np.float32)


def _get_vehicle_pose(frame) -> np.ndarray:
    """Get 4x4 vehicle-to-world pose from frame."""
    pose = np.array(frame.pose.transform).reshape(4, 4)
    return pose


def convert_segment(tfrecord_path: str, output_dir: str, seq_id: int, lidar: str) -> None:
    """Convert one .tfrecord segment to KITTI-like layout."""
    seq_dir = os.path.join(output_dir, str(seq_id))
    velo_dir = os.path.join(seq_dir, "velodyne")
    os.makedirs(velo_dir, exist_ok=True)

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    poses = []
    segment_name = None

    for frame_idx, raw_record in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(raw_record.numpy()))

        if segment_name is None:
            segment_name = frame.context.name

        # Extract points
        if lidar == "top":
            try:
                xyzi = _extract_points_top(frame)
            except Exception:
                xyzi = _extract_points_simple(frame, "top")
        else:
            xyzi = _extract_points_simple(frame, lidar)

        # Save point cloud
        bin_path = os.path.join(velo_dir, f"{frame_idx:06d}.bin")
        xyzi.tofile(bin_path)

        # Collect pose
        pose = _get_vehicle_pose(frame)
        poses.append(pose[:3, :].flatten())

    # Write poses.txt
    poses_arr = np.array(poses)
    np.savetxt(os.path.join(seq_dir, "poses.txt"), poses_arr, fmt="%.10f")

    # Write segment_id.txt
    with open(os.path.join(seq_dir, "segment_id.txt"), "w") as f:
        f.write(segment_name or os.path.basename(tfrecord_path))

    print(f"  seq {seq_id}: {frame_idx + 1} frames, segment={segment_name}")


def main():
    parser = argparse.ArgumentParser(description="Convert Waymo tfrecords to KITTI-like layout")
    parser.add_argument("--input_dir", required=True, help="Directory containing .tfrecord files")
    parser.add_argument("--output_dir", required=True, help="Output directory for KITTI-like layout")
    parser.add_argument("--max_segments", type=int, default=None, help="Max number of segments to convert")
    parser.add_argument("--lidar", default="top", choices=list(LIDAR_NAMES.keys()))
    args = parser.parse_args()

    tfrecords = sorted(glob.glob(os.path.join(args.input_dir, "*.tfrecord")))
    if not tfrecords:
        tfrecords = sorted(glob.glob(os.path.join(args.input_dir, "**/*.tfrecord"), recursive=True))
    if not tfrecords:
        raise SystemExit(f"No .tfrecord files found in {args.input_dir}")

    if args.max_segments:
        tfrecords = tfrecords[: args.max_segments]

    print(f"Found {len(tfrecords)} segments, converting with lidar={args.lidar}")
    os.makedirs(args.output_dir, exist_ok=True)

    for seq_id, path in enumerate(tfrecords):
        print(f"[{seq_id + 1}/{len(tfrecords)}] {os.path.basename(path)}")
        convert_segment(path, args.output_dir, seq_id, args.lidar)

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
