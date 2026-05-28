# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/)
version semantics.

## [Unreleased]

## [0.5.0] — modernization release

Stable release. The bulk of the modernization (Rust CPU extension,
Triton GPU kernel, apoptosis primitives, mkdocs / ruff / ty / uv tooling)
landed in [0.5.0.dev1]; the items below are what changed between dev1
and the stable release.

### Added

- **DuckDB-backed state store** replacing the zarr `_raw_states` array.
  In-memory by default with a configurable `memory_limit` (DuckDB
  auto-spills past the cap); optional on-disk database via the existing
  `disk_path` argument. An Arrow-batched staging buffer subsumes the
  bounded-window / multi-batch aggregation that used to be a separate
  concern. `counts()` and `state_ids()` now derive from a single cached
  `SELECT s, COUNT(*) FROM states GROUP BY s`, retiring the
  `_lex_sort` + bin-edge analysis path. The Rust `_lex_sort` is kept in
  the crate as a reference oracle for validating the DuckDB path.
- **`probe.to_arrow()`** returning the captured microstates as a
  `pyarrow.Table` for notebook / REPL inspection.
- **`ts.attach` / `ts.match` predicate attach API** (PEP 8, composable).
  `ts.match(types=, name=, predicate=)` returns a callable matcher that
  composes via `|`, `&`, and `~`; `ts.attach(model, where, when="after", storage_path=, memory_device=, memory_limit=, raise_on_capture_error=)`
  drives the same machinery as the legacy `build_efficiency_model`.
- **Inspection API**: `ts.layers(model)`, `ts.layer(model, name)`,
  `ts.entropy(model)`, `ts.efficiency(model)`. `entropy()` dispatches on
  the first argument so the legacy `entropy(counts, alpha)` call still
  works.
- **`StateCaptureHook` is now an `nn.Module` probe** owned by a top-level
  `_tensorstate_probes` `ModuleDict`. Probes travel with `.to()` /
  `.cuda()` via non-persistent buffers; observational state stays out
  of `state_dict()`.
- **AttentionNode** for `nn.MultiheadAttention` head-level apoptosis
  (destroy + identical-head merge only; distinct-head merge is non-linear
  in the projection weights and intentionally not implemented).
  `ModuleGraph` now hooks MHA as a single node and skips its `out_proj`
  child so the whole module surfaces as one `ModuleData`.
- **`raise_on_capture_error`** flag on `StateCaptureHook`. Capture-thread
  failures are logged at the time they happen via a `Future` done
  callback and re-raised on read with a chained traceback. Default stays
  non-disruptive so one bad batch doesn't abort a long training run.
- **Conv / depthwise / transposed forward-invariant tests** for the
  existing channel-merge surgery (the surgery was already correct; it
  was untested under real `(N,C,H,W)` input).
- **Hook lifecycle tests** covering attach idempotency, detach,
  `reset_states`, `capture_on` toggle, and that observational state is
  not carried by `state_dict()`.
- **`mkdocs build --strict` re-enabled** in CI; every public function
  has parameter type annotations and `Raises:` blocks are griffe-clean.

### Changed

- **Module files renamed to PEP 8 snake_case**: `States.py`→`states.py`,
  `Layers.py`→`layers.py`, `Dependency.py`→`dependency.py`,
  `TensorState.py`→`core.py`. The package directory `TensorState/`
  stays CamelCase since it is the public import name. `git mv`
  preserves history; intra-package imports, logger names, and docs
  references all updated.
- **`requires-python` tightened to `>=3.13,<3.14`.** Python 3.14 is
  blocked upstream until `networkx` ships 3.14 support (its
  `configs.py` slotted-dataclass fails to import on 3.14, which breaks
  `torch.compile`'s functorch import chain).
- **`build_efficiency_model` is a thin shim over `ts.attach`**, kept
  for back-compat. `model.efficiency_layers` is a deprecated alias
  (`_DeprecatedProbeList`) that warns on iteration and resolves to the
  live probes. Both will be removed in a future release; new code should
  use `ts.attach` and `ts.layers(model)`.
- **`StateCaptureHook._capture` is decorated `torch._dynamo.disable`** so
  that a `torch.compile`'d model cleanly graph-breaks around the hook
  and the eager hook still captures. Without this, dynamo traced into
  the host-side store path and silently captured zero states.
- **GPU buffer is independent of the (former zarr) chunk size**, sized
  from `gpu_buffer_size` (MB) with a sensible floor.
- **Default `memory_device`** resolves to `"gpu"` when CUDA is
  available, `"cpu"` otherwise.

### Removed

- **`zarr` and `numcodecs`** runtime dependencies (and their sub-deps
  `asciitree`, `deprecated`, `fasteners`, `wrapt`).

### Fixed

- **Capture under `torch.compile` silently dropped to zero states.** The
  fix above (`torch._dynamo.disable`) restores capture. Verified with a
  regression test that asserts a non-zero captured count on a compiled
  LeNet5.
- **`compress_states(bool_)` dtype mismatch** — the extension's pi8
  path expects `uint8`. Bool is one byte so a zero-copy `.view(np.uint8)`
  on the bool array fixes it.
- **`ModuleGraph._grad_trace` was a `ClassVar` dict** accumulating
  `grad_fn -> module` entries across every graph in the session. Reset
  per-instance now, fixing the cross-test pollution that made the
  AlexNet apoptosis test intermittently fail in full-suite runs.
- **`disk_path` test fixture leaked** the `states_master` directory
  between parametrized cases. Switched to `tmp_path_factory.mktemp` so
  each case gets a unique, auto-cleaned dir.

## [0.5.0.dev1] — modernization release (work in progress)

The first development release after a complete modernization pass. The
public Python API is preserved where reasonable; the build system,
extension implementation, and runtime dependencies are new.

### Added

- **Rust CPU extension** replacing the Cython AVX/BMI2 bit-packing
  routines. Built via `maturin` + PyO3; ships pre-compiled wheels for
  Linux / macOS / Windows. Output is byte-identical to the Cython baseline.
- **Hand-tuned SIMD compress paths with runtime dispatch.** AVX2 +
  BMI2 implementations on x86_64 and NEON on aarch64, behind a
  `OnceLock`-cached function-pointer dispatch. AVX2 is preferred when
  available; falls back through BMI2 (u8 path only) to scalar.
  Production `_compress_tensor_ps` / `_compress_tensor_pi8` inherit
  the speedup transparently — measurements show ~3–7× faster than
  Cython AVX on the f32 path and ~10–13× faster than Cython BMI2 on
  the u8 path. Cargo parity tests verify byte-equivalence with the
  scalar reference. `_build_info()` reports the runtime-selected
  path (e.g., `"x86_64 avx2/avx2"`).
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
- **`tensorstate.testing` fixture package** — download-free, deterministic
  data and models for tests, examples, and benchmarks: `tiny_dataset`,
  `tiny_text_dataset`, `tiny_loader`, `random_states`, `degenerate_states`,
  `seed_all`, and a `small_model` registry (`lenet5`, `mlp`,
  `groupnorm_conv`, `tiny_transformer`). Re-exported as pytest fixtures
  via a `pytest11` entry point so downstream repos get them for free.
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

[0.5.0]: https://github.com/Nicholas-Schaub/TensorState/releases/tag/v0.5.0
[0.5.0.dev1]: https://github.com/Nicholas-Schaub/TensorState/releases/tag/v0.5.0.dev1
[unreleased]: https://github.com/Nicholas-Schaub/TensorState/compare/v0.5.0...HEAD
