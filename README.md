# TensorState (v0.4.0) - Neural Network Efficiency Tools

TensorState is a toolbox to capture neural network information to better
understand how information flows through the network. The core of the toolbox is
the ability to capture and analyze neural layer state space, which logs unique
firing states of neural network layers. This repository implements and extends
the work by Schaub and Hotaling in their paper,
[Assessing Intelligence in Artificial Neural Networks](https://arxiv.org/abs/2006.02909).

## Installation

Precompiled wheels exist for Windows, Linux, and MacOS for Python 3.6-3.8. No
special installation instructions are required in most cases:

`pip install pip --upgrade`

`pip install TensorState`

If the wheels don't download or you run into an error, try installing the
pre-requisites for compiling before installing with `pip`.

`pip install pip --upgrade`

`pip install numpy==1.19.2 Cython==3.0a1`

`pip install TensorState`

## Documentation

https://tensorstate.readthedocs.io/en/latest/

## Developing

The project uses [uv](https://docs.astral.sh/uv/) for environment management,
[ruff](https://docs.astral.sh/ruff/) for lint + format,
[ty](https://docs.astral.sh/ty/) for type checking, and
[pre-commit](https://pre-commit.com/) to run them on every commit.

```bash
# install dev dependencies (ruff, ty, pre-commit, pytest, mkdocs, ...)
uv sync --group dev

# install the git hook so every commit is auto-checked
uv run pre-commit install

# run all hooks against the whole tree (CI-equivalent)
uv run pre-commit run --all-files
```

Individual tools are available via `uv run`:

```bash
uv run ruff check .         # lint
uv run ruff format .        # format
uv run ty check src         # type check
uv run pytest               # tests
```
