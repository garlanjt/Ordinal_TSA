from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
#Extension(include_dirs=[numpy.get_include()])
#setup(
#    ext_modules=cythonize("entropy.pyx"),
#    include_dirs=[numpy.get_include()]
#
#)

#setup(
#    ext_modules=[
#        Extension("wpe_helpers", ["wpe_helpers.pyx"],
#                  include_dirs=[numpy.get_include()]),
#    ],
#)



extensions = [
    Extension('ordinal_TSA', ['ordinal_TSA.pyx'], include_dirs = [np.get_include()]),
    ]

setup(
    ext_modules = cythonize(extensions)
)