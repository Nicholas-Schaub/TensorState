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
- Modernized to torch 2.10, Python 3.13/14, uv-managed development.
