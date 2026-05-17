# 📦 Installation

QuadricsReg requires Python 3.8 - 3.10 and a Linux system (tested on Ubuntu 20.04 / 22.04).

## 1. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate quadricsReg
```

## 2. (Linux only) Fix `libstdc++` runtime version

Some recent Python packages (scipy, open3d, gtsam) compiled with newer GCC require `GLIBCXX_3.4.29+`, which is missing on Ubuntu 20.04's system `libstdc++`. The conda environment already provides a newer `libstdc++` via `libstdcxx-ng`, so we just need to make sure Python loads it first.

Run once after creating the environment:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/libstdcxx.sh <<'EOF'
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
EOF

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/deactivate.d/libstdcxx.sh <<'EOF'
unset LD_PRELOAD
unset LD_LIBRARY_PATH
EOF
```

Re-activate the environment for the change to take effect:

```bash
conda deactivate
conda activate quadricsReg
```

## 3. Install system dependencies

The C++ extension modules require:

```bash
sudo apt install -y \
    libpcl-dev libeigen3-dev libboost-all-dev \
    libyaml-cpp-dev libigraph-dev libtbb-dev
```

## 4. Build the C++ bindings

```bash
bash build.sh
```

This compiles `ElementsExtractor` and `MaxcliqueSlover` and copies the resulting `.so` files to the repository root.

## 5. (Optional) Install as a Python package

```bash
pip install -e .
```

After this you can `import quadricsreg` from any working directory.

## Verification

Run the demo to confirm everything is wired correctly:

```bash
python demo.py
```

You should see timing logs and two output files printed at the end:

```
Estimated transformation saved to: .../demo/result/T_optimized.txt
Registered point cloud saved to:   .../demo/result/pcd_combined.ply
```
