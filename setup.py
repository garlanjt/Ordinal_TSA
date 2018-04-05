from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('ordinal_TSA', ['ordinal_TSA.pyx'], include_dirs = [np.get_include()]),
    ]

setup(
    ext_modules = cythonize(extensions)
)