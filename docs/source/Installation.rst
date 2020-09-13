============
Installation
============

------------
Introduction
------------

TensorState uses accelerated Cython code to capture neural layer state information.
This can create some issues when trying to install on architectures do not include
prepackaged wheels. Please read the appropriate section carefully to make sure
installation of the package is successful.

Most dependencies should be installed when using ``pip``, however some may not be
installed. Please contact me if there are any installation errors so I can fix it.

-------------------------
Windows and Python >= 3.6
-------------------------

Precompiled wheels exist for Windows 10 and all versions of Python up to version 3.8.
No special dependencies are required.

``pip install TensorState``

-----
Linux
-----

For Linux, there is an effort to create manylinux wheels, but for now TensorState
will be compiled from source when using pip. This only requires installing ``numpy``
and ``Cython`` prior to installing TensorState using ``pip``.

``pip install numpy==1.19.2 Cython==3.0a1``

``pip install TensorState``

---------------
Troubleshooting
---------------

Currently, there is no fallback for working on platforms that do not have a C++
compiler or are working on platforms other than x86 architectures such as ARM. If
there is interest, please contact me.
