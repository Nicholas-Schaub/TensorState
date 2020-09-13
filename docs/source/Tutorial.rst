========
Tutorial
========

Coming Soon!

.. ------------
.. Introduction
.. ------------

.. The core ideas behind this package were originally described in our paper,
.. `Assessing Intelligence in Artificial Neural Networks <https://arxiv.org/abs/2006.02909>`_.

.. The fundamental concept behind the work is that the power of a neural network
.. is found in the combination of firing and non-firing neurons rather than individual
.. neurons. For a given input, a state is the combination of firing and none-firing
.. neurons. To make computation easier and more storage friendly, neuron activation
.. is quantized into on/off states based on whether the neurons value is ``>0`` (on) or
.. ``<=0`` (off).

.. If information is stored in the combination of firing neurons, rather than
.. into discrete neurons, then the information storage space of neural networks
.. is considerably larger than conventionally supposed. For example, a neuron 
.. layer with 64 neurons may be considered to have 64 unique features by convential
.. thinking. However, if information is stored in the state space, then the number
.. of features may be up to 2^64. This conception of how information is stored
.. in the network may explain why neural networks are capable of `memorizing labels
.. on large data sets <https://arxiv.org/abs/1611.03530>`_.

.. This package simplifies the capture of neural layers states, and provides some
.. utility functions to assist in analyzing the state space of neural layers.

.. -----------------------------------------------
.. Attach StateCapture Layers to an Existing Model
.. -----------------------------------------------

.. '''''
.. Setup
.. '''''

.. This example will create a LeNet-5 model to classify MNIST digits. It requires
.. the Tensorflow Datasets package to easily access the MNIST dataset:

.. ``pip install tensorflow-datasets``

.. ''''''''''''''''''''
.. Create a LeNet Model
.. ''''''''''''''''''''

.. .. code-block:: python

..     import tensorflow as tf
..     import tensorflow.keras as keras

