import setuptools
import numpy, os
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

os.environ['CFLAGS'] = '-march=native -O3'
os.environ['CXXFLAGS'] = '-march=native -O3'

with open("VERSION",'r') as fh:
    version = fh.read()
    
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TensorState",
    version=version,
    author="Nick Schaub",
    author_email="nick.schaub@nih.gov",
    description="Tools for analyzing neural network architecture.",
    url="https://tensorstate.readthedocs.io/en/latest/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'cython>=3.0a1',
        'numpy>=1.19.1',
        'tensorflow>=2.1.0',
        'zarr>=2.4.0',
        'numcodecs>=0.6.4'
    ],
    ext_modules=cythonize("./TensorState/_TensorState.pyx",compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()]
)