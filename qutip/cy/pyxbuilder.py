import sys
#import os
#import warnings
from setuptools import Extension, setup
from Cython.Build import cythonize
import importlib
import numpy as np

__all__ = ["importpyx"]
'''
with warnings.catch_warnings():
    # TODO: pyximport loads the imp module, which is deprecated in favour of
    # importlib and slated for removal in Python 3.12.  Our intent is to
    # remove all usage of pyximport in QuTiP 5.0 with new Coefficient objects,
    # before Python 3.12 is released.
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="pyximport",
    )
    import pyximport

try:
    old_get_distutils_extension = pyximport.pyximport.get_distutils_extension

    def new_get_distutils_extension(modname, pyxfilename, language_level=None):
        # Remove -Wstrict-prototypes from cflags; we build in C++ mode, where
        # the flag is invalid, but for some reason distutils still has it as a
        # default, and tries to append CFLAGS to the compile even in C++ mode.
        import distutils.sysconfig
        cfg_vars = distutils.sysconfig.get_config_vars()
        if "CFLAGS" in cfg_vars:
            cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace(
                "-Wstrict-prototypes", ""
            )
        extension_mod, setup_args =\
            old_get_distutils_extension(modname, pyxfilename, language_level)
        extension_mod.language = 'c++'
        # If on Win and Python version >= 3.5 and not in MSYS2
        # (i.e. Visual studio compile)
        if sys.platform == 'win32' and os.environ.get('MSYSTEM') is None:
            extension_mod.extra_compile_args = ['/w', '/O1']
        else:
            extension_mod.extra_compile_args = ['-w', '-O1']
        if sys.platform == 'darwin':
            extra_opt = '-mmacosx-version-min=10.9'
            extension_mod.extra_compile_args.append(extra_opt)
            extension_mod.extra_link_args.append(extra_opt)
        return extension_mod, setup_args

    pyximport.pyximport.get_distutils_extension = new_get_distutils_extension

except AttributeError:
    pass
'''

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

'''
def install():
    """Install the pyximport interface."""
    return pyximport.install(
        setup_args={
            'include_dirs': [np.get_include()],
            'script_args': [("language", "c++")],
        }
    )
'''
