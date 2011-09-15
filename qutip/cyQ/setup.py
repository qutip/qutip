from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np 

setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [Extension("spmv", ["spmv.pyx"],extra_compile_args=['-ffast-math','-fopenmp'],extra_link_args=['-fopenmp'])]
)