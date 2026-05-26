# tensorstate

Neural network state-space analysis. Capture and analyze the firing-state
distribution of neural network layers — the "microstates" defined as the
binary firing patterns produced by each layer on each input — and use them
to measure information-theoretic properties (Shannon entropy, layer
efficiency) and to drive structural pruning ("apoptosis") of redundant
neurons.

Implements and extends the work in
[Assessing Intelligence in Artificial Neural Networks](https://arxiv.org/abs/2006.02909)
(Schaub & Hotaling, 2020).

## Installation

```bash
pip install tensorstate
```

Requires Python ≥ 3.13. Prebuilt wheels ship for Linux (x86_64), macOS
(x86_64 and arm64), and Windows (x86_64). Other platforms install from
sdist, which builds the Rust extension via `maturin` automatically (a Rust
toolchain is required for the source install).

## Quick start

```python
import torch, torchvision
import TensorState as ts

model = torchvision.models.mobilenet_v2(num_classes=10)
ts.build_efficiency_model(model, attach_to=["Conv2dNormActivation"])

# Run the model. Microstates are captured per attached layer.
for x, _ in data_loader:
    model(x)

# Inspect per-layer firing entropy.
for layer in model.efficiency_layers:
    print(layer.name, layer.entropy())
```

The capture path uses Triton for bit-packing on CUDA and the Rust
extension for bit-packing on CPU, so the hook overhead is small.

## Developing

The project uses [uv](https://docs.astral.sh/uv/) for environment
management, [maturin](https://www.maturin.rs/) for the Rust extension,
[ruff](https://docs.astral.sh/ruff/) for lint + format, and
[pre-commit](https://pre-commit.com/) to run them on every commit.

```bash
# Install dev dependencies (ruff, ty, pre-commit, pytest, mkdocs, maturin, ...)
uv sync --group dev

# Build the Rust extension in place
uv run maturin develop --release

# Install the git hook so every commit is auto-checked
uv run pre-commit install

# Run the test suite
uv run pytest

# Run all pre-commit hooks against the whole tree
uv run pre-commit run --all-files
```

The legacy Cython extension is preserved in-tree for regression
benchmarking. Build it manually with `bash scripts/build-cython.sh` —
it is not part of CI and not loaded at runtime.

## Documentation

https://nicholas-schaub.github.io/TensorState/

## License

MIT — see `LICENSE`.
