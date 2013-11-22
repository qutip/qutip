from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np
import os

def configuration(parent_package='', top_path=None):
    # compiles files during installation
    from numpy.distutils.misc_util import Configuration
    config = Configuration('cyQ', parent_package, top_path)

    exts = ['spmatfuncs', 'stochastic', 'sparse_utils', 'graph_utils']

    if os.environ['QUTIP_RELEASE'] == 'TRUE':
        for ext in exts:
            config.add_extension(ext,
                                 sources=[ext + ".c"],
                                 include_dirs=[np.get_include()],
                                 extra_compile_args=['-w -ffast-math -O3'],
                                 extra_link_args=[])

    else:
        for ext in exts:
            config.add_extension(ext,
                                 sources=[ext + ".pyx"],
                                 include_dirs=[np.get_include()],
                                 extra_compile_args=['-w -ffast-math -O3'],
                                 extra_link_args=[])
        config.ext_modules = cythonize(config.ext_modules)
   
    return config


if __name__ == '__main__':
    # builds c-file from pyx for distribution
    setup(
        cmdclass={'build_ext': build_ext},
        include_dirs=[np.get_include()],
        ext_modules=[Extension("spmatfuncs", ["spmatfuncs.pyx"],
                               extra_compile_args=['-w -ffast-math -O3'],
                               extra_link_args=[]),
                     Extension("stochastic", ["stochastic.pyx"],
                               extra_compile_args=['-w -ffast-math -O3'],
                               extra_link_args=[]),
                     Extension("sparse_utils", ["sparse_utils.pyx"],
                               extra_compile_args=['-w -ffast-math -O3'],
                               extra_link_args=[]),
                     Extension("graph_utils", ["graph_utils.pyx"],
                               extra_compile_args=['-w -ffast-math -O3'],
                               extra_link_args=[])]
    )
