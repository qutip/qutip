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
Development Status :: 2 - Pre-Alpha
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
import re
import subprocess
import sys
# The following is required to get unit tests up and running.
# If the user doesn't have, then that's OK, we'll just skip unit tests.
try:
    from setuptools import setup, Extension
    EXTRA_KWARGS = {
        'tests_require': ['pytest']
    }
except:
    from distutils.core import setup
    from distutils.extension import Extension
    EXTRA_KWARGS = {}

try:
    import numpy as np
except ImportError as e:
    raise ImportError("numpy is required at installation") from e

from Cython.Build import cythonize
from Cython.Distutils import build_ext

# all information about QuTiP goes here
REQUIRES = ['numpy (>=1.12)', 'scipy (>=1.0)', 'cython (>=0.29.20)']
EXTRAS_REQUIRE = {'graphics': ['matplotlib(>=1.2.1)']}
INSTALL_REQUIRES = ['numpy>=1.12', 'scipy>=1.0', 'cython>=0.29.20']
PACKAGES = ['qutip', 'qutip/ui', 'qutip/qip', 'qutip/qip/device',
            'qutip/qip/operations', 'qutip/qip/compiler',
            'qutip/qip/algorithms', 'qutip/control',
            'qutip/solve', 'qutip/solve/nonmarkov',
            'qutip/_mkl', 'qutip/tests', 'qutip/tests/core',
            'qutip/tests/core/data', 'qutip/tests/solve',
            'qutip/core', 'qutip/core/cy',
            'qutip/core/data/', 'qutip/core/cy/openmp']
PACKAGE_DATA = {
    'qutip': ['configspec.ini'],
    'qutip/tests': ['*.ini'],
    'qutip/core/data': ['*.pxd', '*.pyx'],
    'qutip/core/cy': ['*.pxd', '*.pyx'],
    'qutip/core/cy/src': ['*.hpp', '*.cpp'],
    'qutip/core/cy/openmp': ['*.pxd', '*.pyx'],
    'qutip/core/cy/openmp/src': ['*.hpp', '*.cpp'],
    'qutip/solve': ['*.pxd', '*.pyx'],
    'qutip/solve/nonmarkov': ['*.pxd', '*.pyx'],
    'qutip/tests/qasm_files': ['*.qasm'],
    'qutip/control': ['*.pyx'],
}
# If we're missing numpy, exclude import directories until we can
# figure them out properly.
INCLUDE_DIRS = [np.get_include()] if np is not None else []
NAME = "qutip"
AUTHOR = ("Alexander Pitchford, Paul D. Nation, Robert J. Johansson, "
          "Chris Granade, Arne Grimsmo, Nathan Shammah, Shahnawaz Ahmed, "
          "Neill Lambert, Eric Giguere, Boxi Li, Jake Lishman")
AUTHOR_EMAIL = ("alex.pitchford@gmail.com, nonhermitian@gmail.com, "
                "jrjohansson@gmail.com, cgranade@cgranade.com, "
                "arne.grimsmo@gmail.com, nathan.shammah@gmail.com, "
                "shahnawaz.ahmed95@gmail.com, nwlambert@gmail.com, "
                "eric.giguere@usherbrooke.ca, etamin1201@gmail.com, "
                "jake@binhbar.com")
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "quantum physics dynamics"
URL = "http://qutip.org"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]

_ROOTDIR = os.path.dirname(os.path.abspath(__file__))

# Read from the VERSION file.  This should be a single line file containing
# valid Python package public identifier (see PEP 440), for example
#   4.5.2rc2
#   5.0.0
#   5.1.1a1
# We do that here rather than in setup.cfg so we can apply the local versioning
# number as well (or omit it if we've been passed '--release').
with open(os.path.join(_ROOTDIR, 'VERSION'), "r") as _version_file:
    version = short_version = _version_file.read().strip()
_VERSION_RE = r'\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?'
if re.fullmatch(_VERSION_RE, version, re.A) is None:
    raise ValueError("invalid version: " + version)

release = '--release' in sys.argv or bool(os.environ.get('CI_QUTIP_RELEASE'))
if '--release' in sys.argv:
    sys.argv.remove('--release')
if not release:
    version += "+"
    try:
        _git_out = subprocess.run(
            ('git', 'rev-parse', '--verify', '--short=7', 'HEAD'),
            check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _git_hash = _git_out.stdout.decode(sys.stdout.encoding).strip()
        version += _git_hash or "nogit"
    except subprocess.CalledProcessError:
        version += "nogit"


# Always overwrite qutip/version.py with the current version information.
_version_py_filename = os.path.join(_ROOTDIR, 'qutip', 'version.py')
_version_py_content = f"""\
# THIS FILE IS GENERATED FROM QUTIP SETUP.PY
short_version = '{short_version}'
version = '{version}'
release = {release}
"""
with open(_version_py_filename, 'w') as _version_py_file:
    print(_version_py_content, file=_version_py_file)

# Cython extensions to be compiled.  The key is the relative package name, the
# value is a list of the Cython modules in that package.
cy_exts = {
    'core.data': [
        'add',
        'adjoint',
        'base',
        'convert',
        'csr',
        'dense',
        'dispatch',
        'expect',
        'inner',
        'kron',
        'matmul',
        'mul',
        'norm',
        'permute',
        'pow',
        'project',
        'properties',
        'ptrace',
        'reshape',
        'tidyup',
        'trace',
    ],
    'core.cy': [
        'coefficient',
        'cqobjevo',
        'inter',
        'interpolate',
        'math',
    ],
    'control': [
        'cy_grape',
    ],
    'solve': [
        '_brtensor',
        '_brtools',
        '_brtools_checks',
        '_mcsolve',
        '_piqs',
        '_steadystate',
        '_stochastic',
    ],
    'solve.nonmarkov': [
        '_heom',
    ],
}

# Extra link args
_link_flags = []

# If on Win and Python version >= 3.5 and not in MSYS2
# (i.e. Visual studio compile)
if (
    sys.platform == 'win32'
    and int(str(sys.version_info[0])+str(sys.version_info[1])) >= 35
    and os.environ.get('MSYSTEM') is None
):
    _compiler_flags = ['/w', '/Ox']
# Everything else
else:
    _compiler_flags = ['-w', '-O3', '-funroll-loops']
    if sys.platform == 'darwin':
        # These are needed for compiling on OSX 10.14+
        _compiler_flags.append('-mmacosx-version-min=10.9')
        _link_flags.append('-mmacosx-version-min=10.9')

EXT_MODULES = []
_include = [
    np.get_include(),
]

# Add Cython files from qutip
for package, files in cy_exts.items():
    for file in files:
        _module = 'qutip' + ('.' + package if package else '') + '.' + file
        _file = os.path.join('qutip', *package.split("."), file + '.pyx')
        _sources = [_file, 'qutip/core/data/src/matmul_csr_vector.cpp']
        EXT_MODULES.append(Extension(_module,
                                     sources=_sources,
                                     include_dirs=_include,
                                     extra_compile_args=_compiler_flags,
                                     extra_link_args=_link_flags,
                                     language='c++'))


# Remove -Wstrict-prototypes from cflags
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")


# TODO: reinstate proper OpenMP handling.
if '--with-openmp' in sys.argv:
    sys.argv.remove('--with-openmp')


# Setup commands go here
setup(
    name=NAME,
    version=version,
    packages=PACKAGES,
    include_package_data=True,
    include_dirs=INCLUDE_DIRS,
    ext_modules=cythonize(EXT_MODULES),
    cmdclass={'build_ext': build_ext},
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    keywords=KEYWORDS,
    url=URL,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    requires=REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    package_data=PACKAGE_DATA,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    **EXTRA_KWARGS,
)

print("""\
==============================================================================
Installation complete
Please cite QuTiP in your publication.
==============================================================================
For your convenience a bibtex reference can be easily generated using
`qutip.cite()`\
""")
