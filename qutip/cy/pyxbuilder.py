import os
import sys
from setuptools import Extension, setup
import importlib
import numpy as np

try:
    from Cython.Build import cythonize
    import Cython.Build.Inline as Inline
except ImportError:
    cythonize = None
    Inline = None


__all__ = ["importpyx"]


def importpyx(file, func):
    """
    Compile f"{file}.pyx" import "func" from it and return that object.

    Inspired cython#3145.
    """
    extra_link_args = []
    if sys.platform == 'win32' and os.environ.get('MSYSTEM') is None:
        extra_compile_args = ['/w', '/O1']
    else:
        extra_compile_args = ['-w', '-O1']
    if sys.platform == 'darwin':
        extra_opt = '-mmacosx-version-min=10.9'
        extra_compile_args.append('-mmacosx-version-min=10.9')
        extra_link_args.append('-mmacosx-version-min=10.9')

    ext = Extension(
        file,
        sources=[file+".pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )

    build_extension = Inline._get_build_extension()
    build_extension.extensions = cythonize(ext)
    build_extension.build_temp = "."
    build_extension.build_lib  = "."
    build_extension.run()

    module = importlib.import_module(file)

    return getattr(module, func)
