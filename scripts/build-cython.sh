#!/usr/bin/env bash
# Build the (optional) Cython extension for regression testing.
#
# TensorState's CPU primitives now live in the Rust extension at
# `TensorState._TensorState_rs` (built automatically via `uv sync`).
# The original Cython extension at `TensorState._TensorState` is kept
# in-tree but is no longer the default; this script builds it on demand
# so the regression test in `benches/cython_vs_rust.py` can compare them.
#
# After running this script, set TENSORSTATE_USE_CYTHON=1 to make
# `import TensorState` resolve the Cython extension instead of Rust.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f "src/TensorState/_TensorState.pyx" ]; then
    echo "Cython source not found at src/TensorState/_TensorState.pyx"
    exit 1
fi

echo "Building Cython extension..."
uv run --with cython --with numpy --with setuptools python - <<'PY'
import os
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

os.environ.setdefault("CFLAGS", "-march=haswell -O3")
os.environ.setdefault("CXXFLAGS", "-march=haswell -O3")

setup(
    name="tensorstate-cython-ext",
    ext_modules=cythonize(
        [Extension(
            "TensorState._TensorState",
            sources=["src/TensorState/_TensorState.pyx"],
            language="c++",
            include_dirs=[numpy.get_include()],
        )],
        compiler_directives={"language_level": 3, "embedsignature": True},
    ),
    script_args=["build_ext", "--inplace", "--build-lib", "src", "--build-temp", "build/cython"],
)
PY

echo ""
echo "Done. Cython extension at:"
find src/TensorState -name "_TensorState.cpython-*.so" 2>/dev/null
echo ""
echo "To use the Cython path instead of Rust:"
echo "  export TENSORSTATE_USE_CYTHON=1"
