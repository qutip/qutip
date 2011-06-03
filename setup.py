#!/usr/bin/env python
"""QuTiP: The Quantum Toolbox in Python

QuTiP is open-source software for simulating the dynamcis of 
open quantum systems.  The QuTiP library depends on the 
excellent Numpy and Scipy numerical packages. In addition, 
graphical output is provided by Matplotlib.  QuTiP aims
to provide user-friendly and efficient numerical simulations
of a wide vairety of Hamiltonian's, including those with 
arbitrary time-dependent, commonly found in a wide range of 
physics applications. QuTiP is freely avaliable for use and/or 
modification on all major platforms. Being free of any licensing 
fees, QuTiP is ideal for exploring quantum mechanics and 
dynamics in the classroom.
"""

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: GNU GPL3
Programming Language :: Python
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

MAJOR               = 0
MINOR               = 1
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

import os
import sys
import shutil
import re
import subprocess
import warnings
from setuptools import setup, find_packages

def svn_version():
    from numpy.compat import asstr
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    try:
        out = subprocess.Popen(['svn', 'info'], stdout=subprocess.PIPE,
                env=env).communicate()[0]
    except OSError:
        warnings.warn(" --- Could not run svn info --- ")
        return ""
    r = re.compile('Revision: ([0-9]+)')
    svnver = None
    for line in asstr(out).split('\n'):
        m = r.match(line)
        if m:
            svnver = m.group(1)
    if not svnver:
        raise ValueError("Error while parsing svn version ?")
    return svnver


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
    packages = find_packages(),
    author = "Paul D. Nation, Robert J. Johansson",
    author_email = "pnation@riken.jp, robert@riken.jp",
    license = "GPL3",
    description = DOCLINES[0],
    long_description = "\n".join(DOCLINES[2:]),
    keywords = "quantum physics dynamics",
    url = "http://code.google.com/p/qutip/",
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ["Linux", "Mac OS-X", "Unix", "Windows"],
    install_requires=['numpy','scipy','matplotlib'],
    include_package_data=True 
    )
