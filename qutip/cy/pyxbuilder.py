import sys
from setuptools import Extension, setup
from Cython.Build import cythonize
import importlib
import numpy as np

__all__ = ["importpyx"]

def importpyx(file, func):
    extra_link_args = []
    if sys.platform == 'win32' and os.environ.get('MSYSTEM') is None:
        extra_compile_args = ['/w', '/O1']
    else:
        extra_compile_args = ['-w', '-O1']
    if sys.platform == 'darwin':
        extra_opt = '-mmacosx-version-min=10.9'
        extra_compile_args.append('-mmacosx-version-min=10.9')
        extra_link_args.append('-mmacosx-version-min=10.9')

    sys.argv = ["setup.py", "build_ext", "--inplace"]

    ext = Extension(
        file,
        sources=[file+".pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )
    setup(ext_modules=cythonize(ext))

    module = importlib.import_module(file)

    return getattr(module, func)
