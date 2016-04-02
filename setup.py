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
Development Status :: 4 - Beta
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
import numpy as np

# The following is required to get unit tests up and running.
# If the user doesn't have, then that's OK, we'll just skip unit tests.
try:
    import setuptools
    TEST_SUITE = 'nose.collector'
    TESTS_REQUIRE = ['nose']
    TESTING_KWARGS = {
        'test_suite': TEST_SUITE,
        'tests_require': TESTS_REQUIRE
    }
except:
    TESTING_KWARGS = {}

from numpy.distutils.core import setup

# all information about QuTiP goes here
MAJOR = 3
MINOR = 2
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
REQUIRES = ['numpy (>=1.6)', 'scipy (>=0.11)', 'cython (>=0.15)',
            'matplotlib (>=1.1)']
PACKAGES = ['qutip', 'qutip/ui', 'qutip/cy', 'qutip/qip', 'qutip/qip/models',
            'qutip/qip/algorithms', 'qutip/control', 'qutip/nonmarkov', 
            'qutip/mkl', 'qutip/tests']
PACKAGE_DATA = {
    'qutip': ['configspec.ini'],
    'qutip/tests': ['bucky.npy', 'bucky_perm.npy'],
    'qutip/cy': ['*.pxi', '*.pxd', '*.pyx'],
    'qutip/control': ['*.pyx']
}
INCLUDE_DIRS = [np.get_include()]
EXT_MODULES = []
NAME = "qutip"
AUTHOR = "Paul D. Nation, Robert J. Johansson"
AUTHOR_EMAIL = "pnation@korea.ac.kr, robert@riken.jp"
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "quantum physics dynamics"
URL = "http://qutip.org"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]


def write_f2py_f2cmap():
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, '.f2py_f2cmap'), 'w') as f:
        f.write("dict(real=dict(sp='float', dp='double', wp='double'), " +
                "complex=dict(sp='complex_float', dp='complex_double', " +
                "wp='complex_double'))")


def git_short_hash():
    try:
        return "-" + os.popen('git log -1 --format="%h"').read().strip()
    except:
        return ""

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev' + git_short_hash()


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

# check for fortran option
if "--with-f90mc" in sys.argv:
    with_f90mc = True
    sys.argv.remove("--with-f90mc")
    write_f2py_f2cmap()
else:
    with_f90mc = False

if not with_f90mc:
    os.environ['FORTRAN_LIBS'] = 'FALSE'
    print("Installing without the fortran mcsolver.")
else:
    os.environ['FORTRAN_LIBS'] = 'TRUE'

# using numpy distutils to simplify install of data directory for testing
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('qutip')
    config.get_version('qutip/version.py')  # sets config.version
    config.add_data_dir('qutip/tests')

    return config


# Setup commands go here
setup(
    name = NAME,
    packages = PACKAGES,
    include_dirs = INCLUDE_DIRS,
    ext_modules = EXT_MODULES,
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
    package_data = PACKAGE_DATA,
    configuration = configuration,
    zip_safe = False,
    **TESTING_KWARGS
)
