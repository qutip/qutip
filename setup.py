#!/usr/bin/env python

import glob
import os
import re
import subprocess
import sys
import sysconfig
import warnings

# Required third-party imports, must be specified in pyproject.toml.
from setuptools import setup, Extension
from distutils import sysconfig
import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext

_ROOTDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOTDIR)

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


# Linker arguments
_link_flags = []

if (
    sys.platform == 'win32'
    and sys.version_info[:2] >= (3, 5)
    and os.environ.get('MSYSTEM') is None
):
    # Visual Studio
    _compiler_flags = ['/w', '/Ox']
else:
    # Everything else
    _compiler_flags = ['-w', '-O3', '-funroll-loops']
if sys.platform == 'darwin':
    # These are needed for compiling on OSX 10.14+
    _compiler_flags.append('-mmacosx-version-min=10.9')
    _link_flags.append('-mmacosx-version-min=10.9')

ext_modules = []
_include = [
    np.get_include(),
]

# Add Cython files from qutip
_matmul_csr_vector = os.path.join(
    # Don't include _ROOTDIR; this must be a relative path from setup.py in
    # order for it to be included in distribution SOURCES.txt etc.
    'qutip', 'core', 'data', 'src', 'matmul_csr_vector.cpp',
)
for pyx_file in glob.glob('qutip/**/*.pyx', recursive=True):
    if 'compiled_coeff' in pyx_file or 'qtcoeff_' in pyx_file:
        # In development (at least for QuTiP ~4.5 and ~5.0) sometimes the
        # Cythonised time-dependent coefficients would get dropped in the qutip
        # directory if you weren't careful - this is just trying to minimise
        # the occasional developer error.
        warnings.warn(
            "skipping generated time-dependent coefficient: "
            + pyx_file
        )
        continue
    # We have to be a little verbose about splitting the path because Windows
    # intermittently uses '\' or '/' in paths.
    pyx_module_path = []
    _head = pyx_file[:-4]
    while _head:
        _head, _tail = os.path.split(_head)
        pyx_module_path.append(_tail)
    pyx_module = '.'.join(reversed(pyx_module_path))
    pyx_sources = [pyx_file, _matmul_csr_vector]
    ext_modules.append(Extension(pyx_module,
                                 sources=pyx_sources,
                                 include_dirs=_include,
                                 extra_compile_args=_compiler_flags,
                                 extra_link_args=_link_flags,
                                 language='c++'))

# Remove -Wstrict-prototypes from CFLAGS; the flag is not valid for C++
# compiles, but CFLAGS gets appended to the call anyway.
cfg_vars = sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

# TODO: reinstate proper OpenMP handling.
if '--with-openmp' in sys.argv:
    sys.argv.remove('--with-openmp')

# Most of the kwargs to setup are defined in setup.cfg; the only ones we keep
# here are ones that we have done some compile-time processing on.
setup(
    version=version,
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext},
)

print("""\
==============================================================================
Installation complete
Please cite QuTiP in your publication.
==============================================================================
For your convenience a bibtex reference can be easily generated using
`qutip.cite()`""")
