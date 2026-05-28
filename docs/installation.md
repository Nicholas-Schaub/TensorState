# Installation

## Introduction

TensorState uses an accelerated Rust extension on CPU and a Triton kernel on
GPU to capture and bit-pack neural layer state information. This can create
some issues when trying to install on architectures that do not include
prepackaged wheels. Please read the appropriate section carefully to make
sure installation of the package is successful.

As of v0.5, GPU acceleration is provided by a Triton kernel that ships with
PyTorch — no separate CUDA toolkit, no CuPy. If PyTorch is built with CUDA
support and a CUDA-capable GPU is available, TensorState's GPU path is
automatically used when input tensors live on `cuda`.

Most dependencies should be installed when using `pip`, however some may not
be installed.

## Simple installation

Precompiled wheels exist for Windows, Linux, and macOS for Python 3.13.
PyTorch >= 2.10 must already be installed.

```bash
pip install TensorState
```

GPU acceleration is automatic when PyTorch is built with CUDA and inputs
live on a CUDA device. No additional install steps are required.

## Troubleshooting

For Linux, there are manylinux wheels that should support most distributions
(`pip install TensorState`). In some cases pip may try to compile from
source (e.g., Alpine Linux). The source build uses `maturin` to compile the
Rust extension, so you need a Rust toolchain installed first (see
[rustup.rs](https://rustup.rs/)).

## Install from source

```bash
git clone https://github.com/Nicholas-Schaub/TensorState
cd TensorState
```

You need a Rust toolchain installed (install via
[rustup.rs](https://rustup.rs/)). The build is driven by `maturin` and does
not require a separate C++ compiler.

The recommended development workflow uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
uv run pytest
```

Or with pip directly:

```bash
pip install .
```

## Other information

The CPU extension picks a SIMD path at runtime: AVX2 and BMI2 on x86_64,
or NEON on aarch64. Older or unsupported architectures fall back to a
scalar path. If the extension fails to import for any reason, a slower
torch-backed CPU path is still available. If you hit a platform-specific
build issue, please open an issue on
[GitHub](https://github.com/Nicholas-Schaub/TensorState/issues).
