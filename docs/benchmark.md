# 📊 Benchmark Generation & Evaluation

This document explains how to generate benchmark test pairs from official datasets and run evaluation with QuadricsReg.

## Supported Datasets

| Dataset | Pair Mode | Semantics |
|---|---|---|
| [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) | `intra_seq_lc` | Yes |
| [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/) | `intra_seq_lc` | Yes |
| [Apollo-SouthBay](https://developer.apollo.auto/southbay.html) | `intra_seq_lc` | No |
| [Waymo](https://waymo.com/open/) | `intra_seq_ode` | Yes |
| [nuScenes](https://www.nuscenes.org/) | `intra_seq_ode` | Yes |

## Quick Start

```bash
# 1. Edit the dataset config — fill in your data_root
vim configs/data_set/KITTI.yaml

# 2. Generate benchmark pairs
python -m tools.benchmark.build --dataset kitti --config configs/data_set/KITTI.yaml

# 3. Run evaluation
python eval.py --DATA_SET_name KITTI --mode 0_10
```

## Dataset Preparation

### KITTI

Download from [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

Expected layout:
```
<data_root>/
└── sequences/
    ├── 00/
    │   ├── calib.txt
    │   └── velodyne/
    │       ├── 000000.bin
    │       └── ...
    ├── 01/
    └── ...
```


### KITTI-360

Download from [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/).

Expected layout:
```
<data_root>/
├── calibration/
│   ├── calib_cam_to_pose.txt
│   └── calib_cam_to_velo.txt
├── data_3d_raw/
│   └── 2013_05_28_drive_XXXX_sync/
│       └── velodyne_points/data/*.bin
└── data_poses/
    └── 2013_05_28_drive_XXXX_sync/
        └── poses.txt
```

### Apollo-SouthBay

Download from [Apollo-SouthBay](https://developer.apollo.auto/southbay.html).

Expected layout:
```
<data_root>/
└── TestData/
    ├── HighWay237/2018-10-12/
    │   ├── pcds/*.pcd
    │   └── poses/gt_poses.txt
    ├── SunnyvaleBigloop/...
    └── ...
```

### Waymo

The adapter expects a pre-converted KITTI-like layout to avoid the heavy `waymo-open-dataset` dependency:

```
<data_root>/
├── 0/
│   ├── velodyne/000000.bin
│   ├── poses.txt           # Nx12 (3x4 flattened per row)
│   └── segment_id.txt      # original segment string
├── 1/
└── ...
```

Convert from official `.tfrecord` files:

```bash
pip install waymo-open-dataset-tf-2-6-0
python tools/benchmark/convert_waymo.py \
    --input_dir /path/to/waymo/training/ \
    --output_dir /path/to/waymo_kitti_format/ \
    --lidar top
```

### nuScenes

Similarly expects a pre-converted layout:

```
<data_root>/
├── 0/
│   ├── velodyne/000000.bin  # float32, Nx5 (x,y,z,intensity,ring)
│   ├── poses.txt            # Nx12 (3x4 flattened per row, LiDAR world pose)
│   ├── scene_token.txt      # original scene token
│   └── sample_tokens.txt    # N lines, one sample token per frame
├── 1/
└── ...
```

Convert from official nuScenes format:

```bash
pip install nuscenes-devkit
python tools/benchmark/convert_nuscenes.py \
    --dataroot /path/to/nuscenes/v1.0-trainval \
    --output_dir /path/to/nuscenes_kitti_format/ \
    --version v1.0-trainval
```

## Benchmark Generation

### Command

```bash
python -m tools.benchmark.build \
    --dataset <name> \
    --config configs/data_set/<NAME>.yaml \
    [--levels 0_10,10_20] \
    [--no-icp] \
    [--out <output_dir>]
```

Options:
- `--dataset`: one of `kitti`, `kitti360`, `apollo`, `waymo`, `nuscenes`
- `--config`: path to the dataset yaml (contains `data_root` and `benchmark:` section)
- `--levels`: comma-separated subset of distance levels to generate (default: all)
- `--no-icp`: skip ICP refinement of GT poses (faster, less accurate)
- `--out`: override output directory (default: `benchmarks/<NAME>_lc/`)

### Configuration

All generation parameters live in the `benchmark:` section of `configs/data_set/<NAME>.yaml`:

```yaml
benchmark:
  enabled: true
  sequences: [0, 2, 5, 6, 8]       # which sequences to use
  distance_levels:
    "0_10":  [0, 10]                # translation range in meters
    "10_20": [10, 20]
    "20_30": [20, 30]
  sample_interval_m: 2.0            # min spacing between kept pairs
  skip_frames: 250                  # min frame gap (LC mode)
  frame_gap: [5, 20]                # frame gap range (ODE mode)
  pair_mode: 'intra_seq_lc'         # or 'intra_seq_ode'
  icp_refine: true
  icp_voxel: 0.05
  icp_max_corr_dist: 0.2
  icp_max_iter: 500
```

### Output Format

Each generated file is a space-separated text file:

```
seq i seq_db j mot1 mot2 mot3 mot4 mot5 mot6 mot7 mot8 mot9 mot10 mot11 mot12 mot13 mot14 mot15 mot16
0 0 0 4438 0.789617 -0.613308 ... 1.000000
```

- `seq`, `i`: source sequence and frame ID
- `seq_db`, `j`: target sequence and frame ID
- `mot1..mot16`: 4x4 ground truth transformation matrix (row-major)

For Waymo/nuScenes, an additional `_index.json` sidecar maps integer IDs back to native tokens.

## Evaluation

### Metrics

Registration is considered **successful** if:
- Rotation error < 5 degrees
- Translation error < 2 meters

Additional metrics logged per pair:
- `rot_error` (degrees)
- `trans_error` (meters)
- `time_total`, `time_reg`, and per-stage timings
- `correspondence_num`, `correspondence_inlier_ratio`
