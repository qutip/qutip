from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np
import os

exts = ['cy_grape']

_compiler_flags = ['-w', '-ffast-math', '-O3', '-mtune=native']

def configuration(parent_package='', top_path=None):
    # compiles files during installation
    from numpy.distutils.misc_util import Configuration
    config = Configuration('control', parent_package, top_path)

    for ext in exts:
        config.add_extension(
            ext, sources=[ext + ".pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=_compiler_flags,
            extra_link_args=[])

    config.ext_modules = cythonize(config.ext_modules)

    return config


if __name__ == '__main__':
    # builds c-file from pyx for distribution
    setup(
        cmdclass={'build_ext': build_ext},
        include_dirs=[np.get_include()],
        ext_modules=[Extension(
            ext, [ext + ".pyx"],
            extra_compile_args=_compiler_flags,
            extra_link_args=[]) for ext in exts])
