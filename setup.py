#!/usr/bin/env python
"""QuTiP: The Quantum Toolbox in Python

QuTiP is open-source software for simulating the dynamics of 
open quantum systems.  The QuTiP library depends on the 
excellent Numpy and Scipy numerical packages. In addition, 
graphical output is provided by Matplotlib.  QuTiP aims
to provide user-friendly and efficient numerical simulations
of a wide variety of Hamiltonian's, including those with 
arbitrary time-dependence, commonly found in a wide range of 
physics applications. QuTiP is freely available for use and/or 
modification on all major platforms. Being free of any licensing 
fees, QuTiP is ideal for exploring quantum mechanics and 
dynamics in the classroom.
"""

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

MAJOR               = 1
MINOR               = 2
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

import os
import sys
import shutil
import re
import subprocess
import warnings
from distutils.core import setup,Extension
import numpy as np 

def svn_version():
    entries_path = 'qutip/.svn/entries'
    try: 
        entries = open(entries_path, 'r').read()
        if re.match('(\d+)', entries):
            rev_match = re.search('\d+\s+dir\s+(\d+)', entries)
            if rev_match:
                rev = rev_match.groups()[0]
            return str(rev)
        else:
            return ""
    except:
        return ""

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    # If in git or something, bypass the svn rev
    if os.path.exists('.svn'):
        FULLVERSION += svn_version()
 

def write_version_py(filename='qutip/_version.py'):
        cnt = """\
# THIS FILE IS GENERATED FROM QUTIP SETUP.PY
short_version='%(version)s'
version='%(fullversion)s'
release=%(isrelease)s
    """
        a = open(filename, 'w')
        try:
            a.write(cnt % {'version': VERSION,'fullversion': FULLVERSION, 'isrelease': str(ISRELEASED)})
        finally:
            a.close()

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0,local_path)
sys.path.insert(0,os.path.join(local_path,'qutip')) # to retrive _version
#always rewrite _version
if os.path.exists('qutip/_version.py'): os.remove('qutip/_version.py')
write_version_py()

setup(
    name = "QuTiP",
    version =FULLVERSION,
    packages = ['qutip','qutip/gui','qutip/examples','qutip/cyQ'],
    include_dirs = [np.get_include()],
    ext_modules=[Extension("qutip.cyQ.ode_rhs", ["qutip/cyQ/ode_rhs.c"],extra_compile_args=['-ffast-math'],extra_link_args=[]), 
                    Extension('qutip.cyQ.cy_mc_funcs', ['qutip/cyQ/cy_mc_funcs.c'],extra_compile_args=['-ffast-math'])],
    author = "Paul D. Nation, Robert J. Johansson",
    author_email = "pnation@riken.jp, robert@riken.jp",
    license = "GPL3",
    description = DOCLINES[0],
    long_description = "\n".join(DOCLINES[2:]),
    keywords = "quantum physics dynamics",
    url = "http://code.google.com/p/qutip/",
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ["Linux", "Mac OSX", "Unix", "Windows"],
    depends=['scipy','matplotlib'],
    package_data = {'qutip/gui': ['*.png']},
    include_package_data=True 
    )
