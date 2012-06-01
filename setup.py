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
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

MAJOR               = 2
MINOR               = 0
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

import os
import sys
import shutil,fnmatch
import re
import subprocess
import warnings
from distutils.core import setup,Extension,Command
from unittest import TextTestRunner, TestLoader
from glob import glob
from os.path import splitext, basename, join as pjoin
from os import walk
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


#--------- test command for running unittests-------------#

class TestCommand(Command):
    user_options = [ ]

    def initialize_options(self):        
        self._dir = os.getcwd()+"/test/"

    def finalize_options(self):
        pass

    def run(self):
        '''
        Finds all the tests modules in tests/, and runs them.
        '''
        testfiles = [ ]
        for t in glob(pjoin(self._dir, 'unittests', 'test_*.py')):
            if not t.endswith('__init__.py'):
                testfiles.append('.'.join(
                    ['test.unittests', splitext(basename(t))[0]])
                )
        tests = TestLoader().loadTestsFromNames(testfiles)
        t = TextTestRunner(verbosity = 1)
        t.run(tests)

#--------- devtest command for running unittests-------------#
class TestHereCommand(Command):
    user_options = [ ]
    sys.path.append(os.getcwd())
    def initialize_options(self):
        self._dir = os.getcwd()+"/test/"

    def finalize_options(self):
        pass

    def run(self):
        '''
        Finds all the tests modules in tests/, and runs them.
        '''
        testfiles = [ ]
        for t in glob(pjoin(self._dir, 'unittests', 'test_*.py')):
            if not t.endswith('__init__.py'):
                testfiles.append('.'.join(
                    ['test.unittests', splitext(basename(t))[0]])
                )
        tests = TestLoader().loadTestsFromNames(testfiles)
        t = TextTestRunner(verbosity = 1)
        t.run(tests)


#------ clean command for removing .pyc files -----------------#

class CleanCommand(Command):
    user_options = [ ]

    def initialize_options(self):
        self._clean_me = [ ]
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.pyc'):
                    self._clean_me.append(pjoin(root, f))

    def finalize_options(self):
        pass

    def run(self):
        pyc_rm=0
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except:
                pyc_rm+=1
        if pyc_rm>0:
            print("Could not remove "+str(pyc_rm)+" pyc files.")
        else:
            print("Removed all pyc files.")

#------ clean command for removing .svn directories ------------#

class CleanSVNCommand(Command):
    user_options = [ ]

    def initialize_options(self):
        self._clean_me = [ ]
        for root, dirs, files in os.walk('.'):
            for d in dirs:
                if d.endswith('.svn'):
                    self._clean_me.append(pjoin(root, d))

    def finalize_options(self):
        pass

    def run(self):
        svn_rm=0
        for clean_me in self._clean_me:
            try:
                shutil.rmtree(clean_me)
            except:
                svn_rm+=1
        if svn_rm>0:
            print("Could not remove "+str(svn_rm)+" svn directories.")
        else:
            print("Removed all SVN directories.")


#remove needless error warnings for released version.
if ISRELEASED:
    os.environ['CFLAGS'] = '-w'

#--------- Setup commands go here ----------------#
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
    platforms = ["Linux", "Mac OSX", "Unix"],
    requires=['numpy (>=1.6)','scipy (>=0.9)','matplotlib (>=1.1)'],
    package_data={'qutip/gui': ['logo.png']},
    cmdclass = { 'test': TestCommand,'devtest': TestHereCommand, 'clean': CleanCommand, 'svnclean': CleanSVNCommand }
    )


matches = []

try:
    walk=os.walk(os.getcwd()+'/build')
except:
    pass
else:
    for root, dirnames, filenames in walk:
      for filename in fnmatch.filter(filenames, 'cy_mc_funcs.so'):
          matches.append(os.path.join(root, filename))
      for filename in fnmatch.filter(filenames, 'ode_rhs.so'):
          matches.append(os.path.join(root, filename))
    for files in matches:
        if 'cy_mc_funcs.so' in files:
            shutil.copyfile(files,os.getcwd()+'/qutip/cyQ/cy_mc_funcs.so')
        elif 'ode_rhs.so' in files:
            shutil.copyfile(files,os.getcwd()+'/qutip/cyQ/ode_rhs.so')





