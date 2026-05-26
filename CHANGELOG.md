# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/)
version semantics.

## [Unreleased]

## [0.5.0.dev1] — modernization release (work in progress)

The first development release after a complete modernization pass. The
public Python API is preserved where reasonable; the build system,
extension implementation, and runtime dependencies are new.

### Added

- **Rust CPU extension** replacing the Cython AVX/BMI2 bit-packing
  routines. Built via `maturin` + PyO3; ships pre-compiled wheels for
  Linux / macOS / Windows. Output is byte-identical to the Cython baseline.
- **Triton GPU kernel** replacing the CuPy ElementwiseKernel for the
  bit-pack hot path. Dispatched automatically when the input tensor is
  on CUDA; matches CuPy throughput at the chunk sizes typical of CNN
  early layers.
- **Apoptosis primitives** (`Dependency.py`) for structural pruning:
  per-node `merge_outputs` / `merge_inputs` / `destroy_outputs` /
  `destroy_inputs` on Linear, Conv, BatchNorm, GroupNorm, LayerNorm; a
  candidate-generation layer (`zero_info_groups`,
  `correlated_weight_groups`); and a `GroupGraph.apoptose` orchestrator
  that walks the module graph and applies merge-then-destroy with
  linearity preservation (mean for producing layer, sum for consuming
  layer).
- **mkdocs-material documentation** with mkdocstrings-driven API pages
  (replacing the old Sphinx/RST setup).
- **Modern code-quality tooling**: ruff (lint + format), ty (type
  checking), typos, mdformat, pre-commit hygiene hooks. Lint runs in CI
  on every push and PR.
- **CI workflows**: test matrix (Python 3.13/3.14 × Linux/macOS/Windows),
  abi3 wheel builds on tag, OIDC-published PyPI release.
- **`[tool.bumpversion]` config** in `pyproject.toml` replacing the
  legacy `.bumpversion.cfg`.

### Changed

- **PyTorch 2.x is the only supported framework.** TensorFlow support
  was removed. `requires-python = ">=3.13,<3.15"`.
- **Build backend is `maturin`.** The `pyproject.toml` declares
  `build-backend = "maturin"`; the build no longer uses `setup.py`
  except as the optional helper for the regression Cython build.
- **PyPI distribution name is `tensorstate`** (lowercase). The import
  path stays `import TensorState` (CamelCase) for back-compat.
- **`uv` is the recommended environment manager.** `uv sync --group dev`
  installs everything needed to develop and build.
- **Version scheme follows PEP 440.** Dev releases are `X.Y.Z.devN`
  (period before `dev`), not the legacy `X.Y.Z-devN`.

### Removed

- **CuPy dependency.** The GPU path is Triton + torch native operations;
  CuPy is no longer required for any CUDA functionality.
- **Cython from the build chain.** The Cython extension is preserved
  in-tree for regression benchmarking only; it is not built by `uv sync`
  and not loaded at runtime.
- **Legacy CI workflows.** `build_manylinux_wheels.yml`,
  `build_macosx_windows_wheels.yml`, the composite action, and the conda
  build-environment yml files are gone, replaced by the modern test +
  wheels + release workflows.
- **`MERGE_REPORT_*.md`** and other narrative artifacts that did not
  belong in the library repository (decision documents now live in
  Plane comments per the project policy).

### Fixed

- **`BatchNormNode.destroy_outputs` double-applies on interior nodes.**
  The `GroupGraph.destroy` chain walker called both `destroy_outputs`
  and `destroy_inputs` on interior nodes; for BN/GN/LN where these are
  aliased to the same method via class-level assignment, the second call
  double-decremented `num_features`. The walker now detects the alias
  and calls once for single-dim nodes.
- **Rust `_decompress_tensor` 1-D vs 2-D signature mismatch.** The Rust
  port took a 1-D `PyReadonlyArray1<u8>` where Cython took a 2-D
  `unsigned char [:,:]`. `States.decompress_states()` passes the 2-D
  compressed array directly, causing a misleading PyO3 type error. The
  Rust signature now matches Cython.

[Unreleased]: https://github.com/Nicholas-Schaub/TensorState/compare/v0.5.0...HEAD
[0.5.0.dev1]: https://github.com/Nicholas-Schaub/TensorState/releases/tag/v0.5.0.dev1
