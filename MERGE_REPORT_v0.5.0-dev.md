# Merge Report: integration/v0.5.0-dev

Plane issue: **AIQ-2** — Merge unmerged branches into a working integration branch

Date: 2026-05-22
Author: Claude (under Nick's direction)

## Summary

Created branch `integration/v0.5.0-dev` from `origin/master` (v0.4.0,
`2dc4e01`) and merged three unmerged feature branches in order:

1. `origin/dev` (GPU state caching)
1. `origin/feat/zero_info` (zero-information neuron detector)
1. `origin/feat/dependency` (dependency-graph-based pruning infrastructure)

**All three merges completed cleanly with zero conflicts.** Git's `ort`
strategy auto-merged `TensorState.py` for the dev+zero_info combination
since their textual changes did not overlap.

## Merge order rationale

| Order | Branch            | Why this position                                                                                               |
| ----- | ----------------- | --------------------------------------------------------------------------------------------------------------- |
| 1     | `dev`             | Foundational — modifies core infrastructure (`Layers.py`, `TensorState.py`). Other branches likely build on it. |
| 2     | `feat/zero_info`  | Small purely-additive change (~50 lines, single new function). Easy to verify in isolation.                     |
| 3     | `feat/dependency` | Largest addition (~840 lines added, new external deps). Goes last so the smaller merges are already settled.    |

## Per-merge details

### Merge 1 — `origin/dev`

**Commit on integration branch:** `2da1ae3`

**Files touched:**

- `.pre-commit-config.yaml`: trivial (`files: src` → `files: src/`)
- `examples/PT_MobileNetV2_Tune.py`: updated to use new `memory_device="gpu"` parameter; uses ThreadPoolExecutor for entropy aggregation; reworked AccuracyCallback hooks for proper capture-on/capture-off semantics
- `src/TensorState/Layers.py`: +132 lines. Adds `memory_device` parameter (default "cpu"), `_state_cache` GPU buffer, `_collect_cache` method that batches GPU-side states before transferring to main memory. New behavior: `state_count` and `states` property getters call `_collect_cache` before returning
- `src/TensorState/TensorState.py`: +30 lines. Propagates `memory_device` parameter through `build_efficiency_model`, `_pt_efficiency_model`, `_tf_efficiency_model`. Comments out `logger.setLevel(logging.WARNING)` (probably a debugging artifact left in by accident)

**Conflicts:** None.

**Note for follow-up:** the `# logger.setLevel(logging.WARNING)` change in TensorState.py looks like a debugging artifact. Recommend restoring it (uncommenting) during AIQ-3.

### Merge 2 — `origin/feat/zero_info`

**Commit on integration branch:** `0c1fa4c`

**Files touched:**

- `src/TensorState/TensorState.py`: +51 lines. New `zero_info()` function (lines 20-66 in the integrated file). Detects three categories: always-off neurons, always-on neurons, and groups of perfectly-synchronized neurons. Also adjusts the `network_efficiency()` docstring to mention PyTorch/Lightning support.

**Conflicts:** None. Git auto-merged `TensorState.py` because dev's changes (logger line + efficiency builder signatures) did not overlap with feat/zero_info's changes (new function + docstring).

**Note:** This is the upstream original of `zero_info_neurons` in `neurofilament/apoptosis.py:31`. The implementations are essentially identical. Once we migrate apoptosis into tensorstate (AIQ paper module), the neurofilament copy can be deleted.

### Merge 3 — `origin/feat/dependency`

**Commit on integration branch:** `bb3824b`

**Files touched:**

- `pyproject.toml`: adds `grandalf = "^0.8"` and `pydantic = "^1.10.7"` to runtime deps
- `poetry.lock`: regenerated for the new deps
- `src/TensorState/__init__.py`: exports `ElementNode`, `ModuleGraph`, `OpNode` from `Dependency`
- `src/TensorState/models/LeNet.py`: adds module-level logger (no behavioral change)
- `src/TensorState/Dependency.py`: 686 lines (new file). Graph-based dependency tracking inspired by [torch-pruning](https://github.com/VainF/Torch-Pruning). Class hierarchy:
  - `GradientData` / `ModuleData` (pydantic models)
  - `ElementNode(Vertex)` — base graph node
  - `OpNode(ElementNode)` — operation node
  - Specialized nodes: `BatchNormNode`, `AdaptivePoolNode`, `PermuteNode`, `ReshapeNode`, `ConvNode`, `ConvGroupNode`, `LinearNode`
  - `Dependency(Edge)` — graph edge
  - `GroupGraph(graph_core)`, `ModuleGraph(Graph)`
  - `linked_neurons()` method that returns groups of neurons that must be updated together — the apoptosis-relevant primitive
- `tests/conftest.py`: adds `id` parameter to a couple of pytest.param test cases (cosmetic)
- `tests/test_dependency.py`: 84 lines (new file). Test for `linked_neurons()` on a simple Conv→Pool→Flatten→Linear sequence, plus a sketch of using the graph for apoptosis-style pruning.

**Conflicts:** None. `feat/dependency` touched a disjoint set of files.

**Implication for design doc.** The design doc at
`papers/entropy_regularization/design_doc.pdf` §7 (Implementation outline)
proposed building dependency-graph infrastructure for apoptosis from scratch
(e.g., a "sample_buffer" method, ring-buffer mode, etc.). Much of this
infrastructure already exists in `Dependency.py`. The design doc should be
revised in light of this finding before any new code is written — the right
approach is to *extend* the existing graph, not duplicate it.

## Verification performed

- All three merges completed via `git merge --no-ff` with no manual conflict resolution.
- All four branch heads confirmed as ancestors of the integration branch (`git merge-base --is-ancestor`).
- All Python files in `src/`, `tests/`, `examples/` parsed cleanly via `ast.parse`.
- Key features confirmed present:
  - `zero_info()` at `src/TensorState/TensorState.py:20`
  - `memory_device`, `_state_cache`, `_collect_cache` at `src/TensorState/Layers.py`
  - `Dependency.py` (21,976 bytes) and `test_dependency.py` (1,824 bytes) present

## Verification NOT performed (deferred to AIQ-3)

- **Test suite execution.** Running `pytest` requires a Python environment with `torch`, `torchvision`, `grandalf`, `pydantic`, `cython`, `zarr`, etc. The current pyproject.toml pins `torch = "1.13.1"` and `python = "^3.8.1"` — both of which AIQ-3 will modernize. Running tests on the unmodernized environment is wasted effort; running them on the modernized environment is the natural verification gate.
- **Cython extension build.** Same dependency reason. AIQ-3 will rebuild on Linux against modern numpy/Cython.
- **Functional smoke test.** Same.

## Status

- Integration branch `integration/v0.5.0-dev` exists locally at `/polus1/schaubnj/ngrf/tensorstate/`.
- Not pushed to origin yet — pushing should wait until AIQ-3 confirms tests pass and the modernization is complete, at which point we know the integration branch is actually shippable.
- The branch has not yet been version-bumped to 0.5.0-dev1 — version bump goes with the AIQ-3 modernization.

## Next steps (AIQ-3)

- Modernize torch from 1.13.1 → latest 2.x.
- Drop TensorFlow code paths.
- Bump Python minimum to ^3.10 or ^3.11.
- Bump `cython` from `3.0.0a11` to current stable.
- Confirm Linux Cython build.
- Run test suite, confirm tests pass.
- Restore `logger.setLevel(logging.WARNING)` (currently commented out from the dev merge).
- After all of the above: bump to `0.5.0-dev1`, push integration branch, open PR.
