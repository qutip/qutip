from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


def configuration(parent_package='', top_path=None):
    # compiles files during installation
    from numpy.distutils.misc_util import Configuration
    config = Configuration('cyQ', parent_package, top_path)
    config.add_extension('spmatfuncs',
                         sources=["spmatfuncs.c"],
                         include_dirs=[np.get_include()],
                         extra_compile_args=['-w -ffast-math -O3'],
                         extra_link_args=[])
    return config


if __name__ == '__main__':
    # builds c-file from pyx for distribution
    setup(
        cmdclass={'build_ext': build_ext},
        include_dirs=[np.get_include()],
        ext_modules=[Extension("spmatfuncs", ["spmatfuncs.pyx"],
                               extra_compile_args=['-w -ffast-math -O3'],
                               extra_link_args=[])]
    )
