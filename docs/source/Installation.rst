============
Installation
============

------------
Introduction
------------

TensorState uses accelerated Cython code to capture neural layer state
information. This can create some issues when trying to install on architectures
do not include prepackaged wheels. Please read the appropriate section carefully
to make sure installation of the package is successful.

As of TensorState v0.3.0, many operations can be performed on Nvidia GPUs if
CuPy is installed. This can lead to significant improvement in performance
because in addition to the data being processed in parallel on the GPU, less
data is transferred over the bus since data is compressed before sending to main
memory.

Most dependencies should be installed when using ``pip``, however some may not
be installed.

-------------------
Simple Installation
-------------------

Precompiled wheels exist for Windows 10, Linux, and MacOS for Python versions 
3.6 to 3.9. No special dependencies are required, but it is assumed that either
TensorFlow >= 2.2 and PyTorch >= 1.6 has already been installed.

``pip install TensorState``

For Cuda acceleration of TensorState operations,
`please read the CuPy documentation on how to install for your version of Cuda. <https://docs.cupy.dev/en/stable/install.html>`_.

---------------
Troubleshooting
---------------

For Linux, there are manylinux wheels that should support most versions of
Linux (``pip install TensorState``). In some cases it may try to compile from
source (e.g. Alpine linux). When compiling, it is necessary to install ``numpy``
and ``Cython`` prior to installation.

``pip install numpy==1.19.2 Cython==3.0a1``

``pip install TensorState``

-------------------
Install From Source
-------------------

If you want to install from source, clone the repo and change directories.

``git clone https://github.com/Nicholas-Schaub/tensorstate``

``cd tensorstate``

You must have a C++ compiler installed. For Windows, mingw will likely not work
but also has not been tested. Microsoft Visual Studio 2015  or later is needed.
For Linux, gcc must be installed.

Once compilers are installed, get the requirements.

``pip install -r requirements.txt``

Finally, install using either:

``python setup.py install``

or

``pip install .``

Since ``TensorState`` is designed to work with both PyTorch and Tensorflow,
neither of these packages are required for installation, but you will need to
install both to run all of the examples. See the PyTorch installation
instructions and tensorflow installation instructions to install each package.

-----------------
Other Information
-----------------

The compile code uses compiler intrinsics found in most CPUs created in 2015 or
later. As long as the CPU is haswell or later, there shouldn't be any issues.

Currently, there is no fallback for working on platforms that do not have a C++
compiler or are working on platforms other than x86 architectures such as ARM.
If there is interest, please open an issue on
`Github <https://github.com/TensorState/issues>`_.
