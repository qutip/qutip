from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np
import os

exts = ['cy_grape']


def configuration(parent_package='', top_path=None):
    # compiles files during installation
    from numpy.distutils.misc_util import Configuration
    config = Configuration('control', parent_package, top_path)

    if os.environ['QUTIP_RELEASE'] == 'TRUE':
        for ext in exts:
            config.add_extension(
                ext, sources=[ext + ".c"],
                include_dirs=[np.get_include()],
                extra_compile_args=[
                    '-w -ffast-math -O3 -march=native -mfpmath=sse'],
                extra_link_args=[])

    else:
        for ext in exts:
            config.add_extension(
                ext, sources=[ext + ".pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=[
                    '-w -ffast-math -O3 -march=native -mfpmath=sse'],
                extra_link_args=[])

        config.ext_modules = cythonize(config.ext_modules)

    return config
