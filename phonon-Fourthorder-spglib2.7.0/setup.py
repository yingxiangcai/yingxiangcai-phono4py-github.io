#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy
from setuptools import setup, Extension

# Add the location of the "spglib/spglib.h" to this list if necessary.
# Example: INCLUDE_DIRS=["/home/user/local/include"]
# Updated for spglib-2.0.0
INCLUDE_DIRS=["/opt/spglib-2.7.0/include"]
# Add the location of the spglib shared library to this list if necessary.
# Example: LIBRARY_DIRS=["/home/user/local/lib"]
# Updated for spglib-2.0.0
LIBRARY_DIRS=["/opt/spglib-2.7.0/lib64"]

# Set USE_CYTHON to True if you want include the cythonization in your build process.
USE_CYTHON=False

ext=".pyx" if USE_CYTHON else ".c"

extensions=[Extension("Fourthorder_core",
                      ["Fourthorder_core"+ext],
                      include_dirs=[numpy.get_include()]+INCLUDE_DIRS,
                      library_dirs=LIBRARY_DIRS,
                      runtime_library_dirs=LIBRARY_DIRS,
                      libraries=["symspg"])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions=cythonize(extensions)

setup(
    name="Fourthorder",
    ext_modules=extensions
)
