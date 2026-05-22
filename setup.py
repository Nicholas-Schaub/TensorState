"""Build the Cython extension for TensorState.

Project metadata lives in pyproject.toml (PEP 621). This file exists only
because setuptools needs a setup.py (or equivalent build hook) to compile
the Cython extension via `cythonize`.
"""

import os

import numpy
from Cython.Build import cythonize
from setuptools import setup


os.environ.setdefault("CFLAGS", "-march=haswell -O3")
os.environ.setdefault("CXXFLAGS", "-march=haswell -O3")

setup(
    ext_modules=cythonize(
        ["src/TensorState/_TensorState.pyx"],
        compiler_directives={"language_level": 3, "embedsignature": True},
    ),
    include_dirs=[numpy.get_include()],
)
