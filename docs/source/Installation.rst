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

Most dependencies should be installed when using ``pip``, however some may not
be installed. Please contact me if there are any installation errors so I can
fix it.

-------------------------
Windows and Python >= 3.6
-------------------------

Precompiled wheels exist for Windows 10 and versions of Python from 3.6 to 3.8.
No special dependencies are required.

``pip install TensorState``

-----
Linux
-----

For Linux, there are manylinux wheels that should support most versions of
Linux (``pip install TensorState``). In some cases it may try to compile from
source (e.g. Alpine linux). When compiling, it is necessary to install ``numpy``
and ``Cython`` prior to installation.

``pip install numpy==1.19.2 Cython==3.0a1``

``pip install TensorState``

-------
Mac OSX
-------

There are currently no OSX wheels, so compilation from source is necessary
following the instructions in the Linux section.

---------------
Troubleshooting
---------------

Currently, there is no fallback for working on platforms that do not have a C++
compiler or are working on platforms other than x86 architectures such as ARM.
If there is interest, please `contact me <nicholas.j.schaub@gmail.com>`_.
