from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('time_series_construction', ['time_series_construction.pyx'], include_dirs = [np.get_include()]),
    ]

setup(
    ext_modules = cythonize(extensions)
)