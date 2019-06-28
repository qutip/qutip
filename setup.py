#!/usr/bin/env python
"""QuTiP: The Quantum Toolbox in Python

QuTiP is open-source software for simulating the dynamics of closed and open
quantum systems. The QuTiP library depends on the excellent Numpy, Scipy, and
Cython numerical packages. In addition, graphical output is provided by
Matplotlib.  QuTiP aims to provide user-friendly and efficient numerical
simulations of a wide variety of quantum mechanical problems, including those
with Hamiltonians and/or collapse operators with arbitrary time-dependence,
commonly found in a wide range of physics applications. QuTiP is freely
available for use and/or modification on all common platforms. Being free of
any licensing fees, QuTiP is ideal for exploring quantum mechanics in research
as well as in the classroom.
"""

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
"""

# import statements
import os
import sys
# The following is required to get unit tests up and running.
# If the user doesn't have, then that's OK, we'll just skip unit tests.
try:
    from setuptools import setup, Extension
    TEST_SUITE = 'nose.collector'
    TESTS_REQUIRE = ['nose']
    EXTRA_KWARGS = {
        'test_suite': TEST_SUITE,
        'tests_require': TESTS_REQUIRE
    }
except:
    from distutils.core import setup
    from distutils.extension import Extension
    EXTRA_KWARGS = {}

try:
    import numpy as np
except:
    np = None

from Cython.Build import cythonize
from Cython.Distutils import build_ext

# all information about QuTiP goes here
MAJOR = 4
MINOR = 4
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
REQUIRES = ['numpy (>=1.12)', 'scipy (>=1.0)', 'cython (>=0.21)']
EXTRAS_REQUIRE = {'graphics':['matplotlib(>=1.2.1)']}
INSTALL_REQUIRES = ['numpy>=1.12', 'scipy>=1.0', 'cython>=0.21']
PACKAGES = ['qutip', 'qutip/ui', 'qutip/cy', 'qutip/cy/src',
            'qutip/qip', 'qutip/qip/models',
            'qutip/qip/algorithms', 'qutip/control', 'qutip/nonmarkov',
            'qutip/_mkl', 'qutip/tests', 'qutip/legacy',
            'qutip/cy/openmp', 'qutip/cy/openmp/src']
PACKAGE_DATA = {
    'qutip': ['configspec.ini'],
    'qutip/tests': ['*.ini'],
    'qutip/cy': ['*.pxi', '*.pxd', '*.pyx'],
    'qutip/cy/src': ['*.cpp', '*.hpp'],
    'qutip/control': ['*.pyx'],
    'qutip/cy/openmp': ['*.pxd', '*.pyx'],
    'qutip/cy/openmp/src': ['*.cpp', '*.hpp']
}
# If we're missing numpy, exclude import directories until we can
# figure them out properly.
INCLUDE_DIRS = [np.get_include()] if np is not None else []
NAME = "qutip"
AUTHOR = ("Alexander Pitchford, Paul D. Nation, Robert J. Johansson, "
          "Chris Granade, Arne Grimsmo, Nathan Shammah, Shahnawaz Ahmed, "
          "Neill Lambert, Eric Giguere")
AUTHOR_EMAIL = ("alex.pitchford@gmail.com, nonhermitian@gmail.com, "
                "jrjohansson@gmail.com, cgranade@cgranade.com, "
                "arne.grimsmo@gmail.com, nathan.shammah@gmail.com, "
                "shahnawaz.ahmed95@gmail.com, nwlambert@gmail.com, "
                "eric.giguere@usherbrooke.ca")
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "quantum physics dynamics"
URL = "http://qutip.org"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]


def git_short_hash():
    try:
        git_str = "+" + os.popen('git log -1 --format="%h"').read().strip()
    except:
        git_str = ""
    else:
        if git_str == '+': #fixes setuptools PEP issues with versioning
            git_str = ''
    return git_str

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'+str(MICRO)+git_short_hash()

# NumPy's distutils reads in versions differently than
# our fallback. To make sure that versions are added to
# egg-info correctly, we need to add FULLVERSION to
# EXTRA_KWARGS if NumPy wasn't imported correctly.
if np is None:
    EXTRA_KWARGS['version'] = FULLVERSION


def write_version_py(filename='qutip/version.py'):
    cnt = """\
# THIS FILE IS GENERATED FROM QUTIP SETUP.PY
short_version = '%(version)s'
version = '%(fullversion)s'
release = %(isrelease)s
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'fullversion':
                FULLVERSION, 'isrelease': str(ISRELEASED)})
    finally:
        a.close()

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0, local_path)
sys.path.insert(0, os.path.join(local_path, 'qutip'))  # to retrive _version

# always rewrite _version
if os.path.exists('qutip/version.py'):
    os.remove('qutip/version.py')

write_version_py()

# Add Cython extensions here
cy_exts = ['spmatfuncs', 'stochastic', 'sparse_utils', 'graph_utils', 'interpolate',
           'spmath', 'heom', 'math', 'spconvert', 'ptrace', 'checks', 'brtools', 'mcsolve',
           'brtools_checks', 'br_tensor', 'inter', 'cqobjevo', 'cqobjevo_factor', 'piqs']

# Extra link args
_link_flags = []

# If on Win and Python version >= 3.5 and not in MSYS2 (i.e. Visual studio compile)
if (sys.platform == 'win32' and int(str(sys.version_info[0])+str(sys.version_info[1])) >= 35
    and os.environ.get('MSYSTEM') is None):
    _compiler_flags = ['/w', '/Ox']
# Everything else
else:
    _compiler_flags = ['-w', '-O3', '-funroll-loops']
    if sys.platform == 'darwin':
        # These are needed for compiling on OSX 10.14+
        _compiler_flags.append('-mmacosx-version-min=10.9')
        _link_flags.append('-mmacosx-version-min=10.9')



EXT_MODULES =[]
# Add Cython files from qutip/cy
for ext in cy_exts:
    _mod = Extension('qutip.cy.'+ext,
            sources = ['qutip/cy/'+ext+'.pyx', 'qutip/cy/src/zspmv.cpp'],
            include_dirs = [np.get_include()],
            extra_compile_args=_compiler_flags,
            extra_link_args=_link_flags,
            language='c++')
    EXT_MODULES.append(_mod)

# Add Cython files from qutip/control
_mod = Extension('qutip.control.cy_grape',
            sources = ['qutip/control/cy_grape.pyx'],
            include_dirs = [np.get_include()],
            extra_compile_args=_compiler_flags,
            extra_link_args=_link_flags,
            language='c++')
EXT_MODULES.append(_mod)


# Add optional ext modules here
if "--with-openmp" in sys.argv:
    sys.argv.remove("--with-openmp")
    if (sys.platform == 'win32'
            and int(str(sys.version_info[0])+str(sys.version_info[1])) >= 35):
        omp_flags = ['/openmp']
        omp_args = []
    else:
        omp_flags = ['-fopenmp']
        omp_args = omp_flags
    _mod = Extension('qutip.cy.openmp.parfuncs',
            sources = ['qutip/cy/openmp/parfuncs.pyx',
                       'qutip/cy/openmp/src/zspmv_openmp.cpp'],
            include_dirs = [np.get_include()],
            extra_compile_args=_compiler_flags+omp_flags,
            extra_link_args=omp_args+_link_flags,
            language='c++')
    EXT_MODULES.append(_mod)
    # Add benchmark pyx
    _mod = Extension('qutip.cy.openmp.benchmark',
            sources = ['qutip/cy/openmp/benchmark.pyx'],
            include_dirs = [np.get_include()],
            extra_compile_args=_compiler_flags,
            extra_link_args=_link_flags,
            language='c++')
    EXT_MODULES.append(_mod)

    # Add brtools_omp
    _mod = Extension('qutip.cy.openmp.br_omp',
            sources = ['qutip/cy/openmp/br_omp.pyx'],
            include_dirs = [np.get_include()],
            extra_compile_args=_compiler_flags,
            extra_link_args=_link_flags,
            language='c++')
    EXT_MODULES.append(_mod)

    # Add omp_sparse_utils
    _mod = Extension('qutip.cy.openmp.omp_sparse_utils',
            sources = ['qutip/cy/openmp/omp_sparse_utils.pyx'],
            include_dirs = [np.get_include()],
            extra_compile_args=_compiler_flags+omp_flags,
            extra_link_args=omp_args+_link_flags,
            language='c++')
    EXT_MODULES.append(_mod)

    # Add cqobjevo_omp
    _mod = Extension('qutip.cy.openmp.cqobjevo_omp',
            sources = ['qutip/cy/openmp/cqobjevo_omp.pyx'],
            include_dirs = [np.get_include()],
            extra_compile_args=_compiler_flags+omp_flags,
            extra_link_args=omp_args,
            language='c++')
    EXT_MODULES.append(_mod)


# Remove -Wstrict-prototypes from cflags
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")


# Setup commands go here
setup(
    name = NAME,
    version = FULLVERSION,
    packages = PACKAGES,
    include_package_data=True,
    include_dirs = INCLUDE_DIRS,
    # headers = HEADERS,
    ext_modules = cythonize(EXT_MODULES),
    cmdclass = {'build_ext': build_ext},
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    license = LICENSE,
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    keywords = KEYWORDS,
    url = URL,
    classifiers = CLASSIFIERS,
    platforms = PLATFORMS,
    requires = REQUIRES,
    extras_require = EXTRAS_REQUIRE,
    package_data = PACKAGE_DATA,
    zip_safe = False,
    install_requires=INSTALL_REQUIRES,
    **EXTRA_KWARGS
)
_cite = """\
==============================================================================
Installation complete
Please cite QuTiP in your publication.
==============================================================================
For your convenience a bibtex reference can be easily generated using
`qutip.cite()`"""
print(_cite)
