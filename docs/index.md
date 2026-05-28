# TensorState

TensorState is a toolbox designed to analyze the way neural networks process
information.

PyTorch (including Lightning) is supported. As of v0.5, TensorFlow support
has been removed. Complex networks may prove problematic for some of the
network functions (such as automatically building an efficiency model).

For comments, suggestions, and bug reports, open an issue on
[GitHub](https://github.com/Nicholas-Schaub/TensorState/issues).

## Where to start

- **[Installation](installation.md)** — set up TensorState with PyTorch.
- **[State Space](state-space.md)** — what "neural layer state space" means
  and why it matters.
- **[Tutorials](tutorials/index.md)** — worked examples, including a full
  PyTorch LeNet-5 walkthrough.
- **[API Reference](reference/index.md)** — auto-generated docs for the
  public API.

## What's new in 0.5

- TensorFlow / Keras support removed; the library is PyTorch-only.
- CuPy dependency removed; the GPU bit-packing path now uses a Triton
  kernel that ships with PyTorch.
- The CPU extension is now Rust (PyO3 + maturin) instead of Cython.
- State storage moved from zarr to DuckDB. Counts and entropy now run
  through a SQL `GROUP BY` over the captured microstates.
- New `ts.attach` / `ts.match` API for selecting which layers to probe.
  The old `build_efficiency_model` still works and uses the same
  machinery underneath.
- Modernized to torch 2.10, Python 3.13, uv-managed development.
