# TensorState - Neural Network Efficiency Tools

TensorState is a toolbox to capture neural network information to better understand how information flows through the network. The core of the toolbox is the ability to capture and analyze neural layer state space, which logs unique firing states of neural network layers. This repository implements and extends the work by Schaub and Hotaling in their paper, [Assessing Intelligence in Artificial Neural Networks](https://arxiv.org/abs/2006.02909).

## Installation

If running on Windows, TensorState can be easily installed using pip for Python version 3.6 and 3.7.

`pip install TensorState`

If running on Linux or on Windows with Python 3.8, make sure to install numpy and Cython 3 before using pip.

`pip install numpy Cython==3.0a1`

`pip install TensorState`

## Documentation

Coming soon!

## Important Information

Currently, the toolbox only works for Tensorflow. However, there are plans to implement PyTorch integration.

The current release is an alpha version, meaning there is no gauruntee that breaking changes will not be made in the near future.