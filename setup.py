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
modification on all Unix based platforms. Being free of any
licensing fees, QuTiP is ideal for exploring quantum mechanics
and dynamics in the classroom.
"""

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Topic :: Scientific/Engineering
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Windows
"""

# import statements
import os
import sys
import shutil
import fnmatch
import re
import subprocess
import warnings
from distutils.core import Extension, Command
from unittest import TextTestRunner, TestLoader
from glob import glob
from os.path import splitext, basename, join as pjoin
from os import walk
import numpy as np
from numpy.distutils.core import setup

# all information about QuTiP goes here-------
MAJOR = 2
MINOR = 2
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
REQUIRES = ['numpy (>=1.6)', 'scipy (>=0.9)', 'matplotlib (>=1.1)']
PACKAGES = ['qutip', 'qutip/gui', 'qutip/examples', 'qutip/cyQ']
PACKAGE_DATA = {'qutip/gui': ['logo.png', 'icon.png']}
INCLUDE_DIRS = [np.get_include()]
EXT_MODULES = [Extension(
    "qutip.cyQ.spmatfuncs", ["qutip/cyQ/spmatfuncs.c"],
    extra_compile_args=['-ffast-math -O3'], extra_link_args=[])]
NAME = "QuTiP"
AUTHOR = "Paul D. Nation, Robert J. Johansson"
AUTHOR_EMAIL = "pnation@korea.ac.kr, robert@riken.jp"
LICENSE = "GPL3"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "quantum physics dynamics"
URL = "http://code.google.com/p/qutip/"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]


def git_short_hash():
    try:
        return "-" + os.popen('git log -1 --format="%h"').read().strip()
    except:
        return ""

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    FULLVERSION += git_short_hash()


def write_version_py(filename='qutip/_version.py'):
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
if os.path.exists('qutip/_version.py'):
    os.remove('qutip/_version.py')
write_version_py()


#--------- test command for running unittests-------------#

class TestCommand(Command):
    user_options = []

    def initialize_options(self):
        self._dir = os.getcwd() + "/test/"

    def finalize_options(self):
        pass

    def run(self):
        '''
        Finds all the tests modules in tests/, and runs them.
        '''
        testfiles = []
        for t in glob(pjoin(self._dir, 'unittests', 'test_*.py')):
            if not t.endswith('__init__.py'):
                testfiles.append('.'.join(
                    ['test.unittests', splitext(basename(t))[0]])
                )
        tests = TestLoader().loadTestsFromNames(testfiles)
        t = TextTestRunner(verbosity=1)
        t.run(tests)

#--------- devtest command for running unittests-------------#


class TestHereCommand(Command):
    user_options = []
    sys.path.append(os.getcwd())

    def initialize_options(self):
        self._dir = os.getcwd() + "/test/"

    def finalize_options(self):
        pass

    def run(self):
        '''
        Finds all the tests modules in tests/, and runs them.
        '''
        testfiles = []
        for t in glob(pjoin(self._dir, 'unittests', 'test_*.py')):
            if not t.endswith('__init__.py'):
                testfiles.append('.'.join(
                    ['test.unittests', splitext(basename(t))[0]])
                )
        tests = TestLoader().loadTestsFromNames(testfiles)
        t = TextTestRunner(verbosity=1)
        t.run(tests)


#------ clean command for removing .pyc files -----------------#

class CleanCommand(Command):
    user_options = [("all", "a", "All")]

    def initialize_options(self):
        self._clean_me = []
        self.all = None
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.pyc'):
                    self._clean_me.append(pjoin(root, f))

    def finalize_options(self):
        pass

    def run(self):
        pyc_rm = 0
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except:
                pyc_rm += 1
        if pyc_rm > 0:
            print("Could not remove " + str(pyc_rm) + " pyc files.")
        else:
            print("Removed all pyc files.")

# remove needless error warnings for released version.
if ISRELEASED:
    os.environ['CFLAGS'] = '-w'


# using numpy distutils to simplify install of data directory for testing
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('qutip')
    config.get_version('qutip/_version.py')  # sets config.version
    config.add_data_dir('qutip/tests')

    return config

#--------- Setup commands go here ----------------#
setup(
    name=NAME,
    packages=PACKAGES,
    include_dirs=INCLUDE_DIRS,
    ext_modules=EXT_MODULES,
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
    package_data=PACKAGE_DATA,
    cmdclass={'test': TestCommand, 'devtest': TestHereCommand,
              'clean': CleanCommand},
    configuration=configuration
)


matches = []

try:
    walk = os.walk(os.getcwd() + '/build')
except:
    pass
else:
    for root, dirnames, filenames in walk:
        for filename in fnmatch.filter(filenames, 'spmatfuncs.so'):
            matches.append(os.path.join(root, filename))
    for files in matches:
        if 'spmatfuncs.so' in files:
            shutil.copyfile(files, os.getcwd() + '/qutip/cyQ/spmatfuncs.so')
