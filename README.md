# TensorState - Neural Network Efficiency Tools

TensorState is a toolbox to capture neural network information to better understand how information flows through the network. The core of the toolbox is the ability to capture and analyze neural layer state space, which logs unique firing states of neural network layers. This repository implements and extends the work by Schaub and Hotaling in their paper, [Assessing Intelligence in Artificial Neural Networks](https://arxiv.org/abs/2006.02909).

## Installation

If running on Windows or Linux, TensorState can be easily installed using pip for Python version 3.5-3.8. However, to install the precompiled wheels, make sure `pip>=19.3`.

`pip install pip --upgrade`

`pip install TensorState`

If the wheels don't download or you run into an error, try installing the pre-requisites for compiling before installing with `pip`.

`pip install numpy Cython==3.0a1`

`pip install TensorState`

## Documentation

https://tensorstate.readthedocs.io/en/latest/

## Important Information

Currently, the toolbox only works for Tensorflow. However, there are plans to implement PyTorch integration.

The current release is an alpha version, meaning there is no gauruntee that breaking changes will not be made in the near future.