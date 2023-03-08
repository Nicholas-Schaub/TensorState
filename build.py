import os

import numpy
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext


def build(setup_kwargs):
    print("Building tensorstate")
    compiler_directives = {"language_level": 3, "embedsignature": True}

    os.environ["CFLAGS"] = "-march=haswell -O3"
    os.environ["CXXFLAGS"] = "-march=haswell -O3"

    setup_kwargs.update(
        {
            "name": "TensorState",
            "package": ["TensorState"],
            "ext_modules": cythonize(
                ["src/TensorState/_TensorState.pyx"],
                compiler_directives=compiler_directives,
            ),
            "cmdclass": {"build_ext": build_ext},
            "include_dirs": [numpy.get_include(), "."],
        }
    )
