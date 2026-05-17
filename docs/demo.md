# 🚀 Demo

Single-pair point cloud registration. Input two `.ply` or `.pcd` files, output the estimated 6-DoF transformation and the merged registered point cloud.

## Quick start

```bash
python demo.py
```

By default this runs registration on a VLP-64 same-sensor pair under `demo/pcd/`.

## Output

The demo writes two files to `demo/result/`:

| File | Content |
|---|---|
| `T_optimized.txt` | 4 × 4 estimated transformation, source → target |
| `pcd_combined.ply` | Source (transformed, yellow) + target (blue), merged for visualization |

Open the `.ply` file in [CloudCompare](https://www.cloudcompare.org/) or [MeshLab](https://www.meshlab.net/) to inspect alignment quality.

## Switching the input pair

The `demo/pcd/` directory ships with several pre-aligned demo pairs:

**Same-sensor pairs** (`.ply`):

```bash
python demo.py \
    --pcd_source_path demo/pcd/avia_source.ply \
    --pcd_target_path demo/pcd/avia_target.ply \
    --extraction_config_source configs/extraction/elements_extractor_avia.yaml \
    --extraction_config_target configs/extraction/elements_extractor_avia.yaml
```

Available same-sensor pairs: `avia`, `mid360`, `vlp16`, `vlp64`.

**Cross-sensor pairs** (`.pcd`):

```bash
# Livox Mid-360 ↔ Velodyne VLP-16
python demo.py \
    --pcd_source_path demo/pcd/cross_mid360_vlp_source_1.pcd \
    --pcd_target_path demo/pcd/cross_mid360_vlp_target_1.pcd \
    --extraction_config_source configs/extraction/elements_extractor_mid360.yaml \
    --extraction_config_target configs/extraction/elements_extractor_vlp16.yaml
```

Available cross-sensor pairs: `cross_avia_vlp_*_{1,2}`, `cross_mid360_vlp_*_1`.

## Available sensor configs

`configs/extraction/` contains pre-tuned extraction parameters for:

- `avia` — Livox Avia (non-repetitive scanning, narrow FoV)
- `mid360` — Livox Mid-360 (semi-dome FoV)
- `vlp16` — Velodyne VLP-16 (16-line spinning)
- `vlp64` — Velodyne HDL-64E

## Optional arguments

| Argument | Default | Purpose |
|---|---|---|
| `--voxel_size` | `0.0` | Voxel down-sample size before processing (0 = disabled) |
| `--if_centerize` | `False` | Subtract point cloud mean before processing (helps with large coordinates, e.g. UTM) |
| `--config_quadricsReg` | `configs/quadricsReg_demo.yaml` | Algorithm hyper-parameters (fitting / matching / estimation) |

## Troubleshooting

**`ImportError: ... GLIBCXX_3.4.29 not found`**: Make sure the `LD_PRELOAD` step from [Installation §2](install.md#2-linux-only-fix-libstdc-runtime-version) is in place.

**`ImportError: elements_extractor_bindings`**: The C++ build hasn't been run, or the resulting `.so` files aren't in the repo root. Run `bash build.sh`.

**Empty correspondences / wrong result**: Try `--voxel_size 0.1` to down-sample dense input clouds, or pass a sensor-specific extraction config that matches your data.
