from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np
import os

exts = ['spmatfuncs', 'stochastic', 'sparse_utils', 'graph_utils', 'interpolate']

_compiler_flags = ['-w', '-ffast-math', '-O3', '-march=native', '-funroll-loops']

def configuration(parent_package='', top_path=None):
    # compiles files during installation
    from numpy.distutils.misc_util import Configuration
    config = Configuration('cy', parent_package, top_path)

    for ext in exts:
        if ext == 'spmatfuncs':
            src = [ext + ".pyx", os.path.join('src','zspmv.c')]
        else:
            src = [ext + ".pyx"]
        config.add_extension(
            ext, 
            sources=src,
            include_dirs=[np.get_include(), 'src'],
            extra_compile_args=_compiler_flags,
            extra_link_args=[],
            depends = [os.path.join('src', 'zspmv.h')])

    config.ext_modules = cythonize(config.ext_modules)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
