#!/bin/bash
set -e

REPO_ROOT=$(dirname "$(realpath "$0")")
CONDA_PREFIX=$(python -c "import sys; print(sys.prefix)")
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYBIND11_DIR="$CONDA_PREFIX/lib/python${PY_VER}/site-packages/pybind11/share/cmake/pybind11"

echo "Python prefix: $CONDA_PREFIX"
echo "Python version: $PY_VER"
echo "pybind11 cmake dir: $PYBIND11_DIR"
echo ""

# 1. ElementsExtractor
echo "Building ElementsExtractor..."
cd "$REPO_ROOT/Thirdparty/ElementsExtractor"
mkdir -p build && cd build
cmake .. -Dpybind11_DIR="$PYBIND11_DIR"
make -j$(nproc)
cp elements_extractor_bindings*.so "$REPO_ROOT/"
find . -name "*.so" ! -name "elements_extractor_bindings*" -exec cp {} "$REPO_ROOT/" \;
echo "ElementsExtractor done."
echo ""

# 2. MaxcliqueSlover
echo "Building MaxcliqueSlover..."
cd "$REPO_ROOT/Thirdparty/MaxcliqueSlover"
mkdir -p build && cd build
cmake .. -Dpybind11_DIR="$PYBIND11_DIR"
make -j$(nproc)
cp maxclique_solver_bindings*.so "$REPO_ROOT/"
find . -name "*.so" ! -name "maxclique_solver_bindings*" -exec cp {} "$REPO_ROOT/" \;
echo "MaxcliqueSlover done."
echo ""

# 3. Set up LD_LIBRARY_PATH in conda activate hook
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
ACTIVATE_SCRIPT="$ACTIVATE_DIR/libstdcxx.sh"
mkdir -p "$ACTIVATE_DIR"

# Check if LD_LIBRARY_PATH line already exists
if ! grep -q "LD_LIBRARY_PATH.*$REPO_ROOT" "$ACTIVATE_SCRIPT" 2>/dev/null; then
    echo "export LD_LIBRARY_PATH=\"$REPO_ROOT:\$LD_LIBRARY_PATH\"" >> "$ACTIVATE_SCRIPT"
    echo "Added LD_LIBRARY_PATH=$REPO_ROOT to conda activate hook."
fi

echo ""
echo "Build complete."
echo "Re-activate your conda environment for changes to take effect:"
echo "  conda deactivate && conda activate $(basename $CONDA_PREFIX)"
