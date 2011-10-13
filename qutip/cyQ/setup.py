from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np 

setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [Extension("ode_rhs", ["ode_rhs.pyx"],extra_compile_args=['-ffast-math','-fopenmp'],extra_link_args=['-fopenmp']),
                   Extension("matrix", ["matrix.pyx"],extra_compile_args=['-ffast-math','-fopenmp'],extra_link_args=['-fopenmp'])]
)
