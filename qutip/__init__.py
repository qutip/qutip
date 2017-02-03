# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
from __future__ import division, print_function, absolute_import
import os
# Fix the multiprocessing issue with NumPy compiled against OPENBLAS
if 'OPENBLAS_MAIN_FREE' not in os.environ:
    os.environ['OPENBLAS_MAIN_FREE'] = '1'
# automatically set number of threads used by MKL and openblas to 1
# prevents errors when running things in parallel.  Should be set
# by user directly in a script or notebook if >1 is needed.
# Must be set BEFORE importing NumPy
if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '1'

if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import warnings

import qutip.settings
import qutip.version
from qutip.version import version as __version__
from qutip.utilities import _version2int

# -----------------------------------------------------------------------------
# Check if we're in IPython.
try:
    __IPYTHON__
    qutip.settings.ipython = True
except:
    qutip.settings.ipython = False

# -----------------------------------------------------------------------------
# Check for minimum requirements of dependencies, give the user a warning
# if the requirements aren't fulfilled
#

numpy_requirement = "1.6.0"
try:
    import numpy
    if _version2int(numpy.__version__) < _version2int(numpy_requirement):
        print("QuTiP warning: old version of numpy detected " +
              ("(%s), requiring %s." %
               (numpy.__version__, numpy_requirement)))
except:
    warnings.warn("numpy not found.")

scipy_requirement = "0.11.0"
try:
    import scipy
    if _version2int(scipy.__version__) < _version2int(scipy_requirement):
        print("QuTiP warning: old version of scipy detected " +
              ("(%s), requiring %s." %
               (scipy.__version__, scipy_requirement)))
except:
    warnings.warn("scipy not found.")

# -----------------------------------------------------------------------------
# check to see if running from install directory for released versions.
#
top_path = os.path.dirname(os.path.dirname(__file__))
try:
    setup_file = open(top_path + '/setup.py', 'r')
except:
    pass
else:
    if ('QuTiP' in setup_file.readlines()[1][3:]) and qutip.version.release:
        print("You are in the installation directory. " +
              "Change directories before running QuTiP.")
    setup_file.close()

del top_path

# -----------------------------------------------------------------------------
# setup the cython environment
#
_cython_requirement = "0.15.0"
try:
    import Cython
    if _version2int(Cython.__version__) < _version2int(_cython_requirement):
        print("QuTiP warning: old version of cython detected " +
              ("(%s), requiring %s." %
               (Cython.__version__, _cython_requirement)))

except Exception as e:
    print("QuTiP warning: Cython setup failed: " + str(e))
else:
    del Cython

# -----------------------------------------------------------------------------
# Load user configuration if present: override defaults.
#
try:
    if os.name == "nt":
        qutip_rc_file = os.path.join(
            os.getenv('APPDATA'), 'qutip', "qutiprc"
        )
    else:
        qutip_rc_file = os.path.join(
            # This should possibly be changed to ~/.config/qutiprc,
            # to follow XDG specs. Also, OS X uses a different naming
            # convention as well.
            os.environ['HOME'], ".qutiprc"
        )
    qutip.settings.load_rc_file(qutip_rc_file)

except KeyError as e:
    qutip.settings._logger.warning(
        "The $HOME environment variable is not defind. No custom RC file loaded.")

except Exception as e:
    try:
        qutip.settings._logger.warning("Error loading RC file.", exc_info=1)
    except:
        pass

# -----------------------------------------------------------------------------
# cpu/process configuration
#
import multiprocessing

# Check if environ flag for qutip processes is set
if 'QUTIP_NUM_PROCESSES' in os.environ:
    qutip.settings.num_cpus = int(os.environ['QUTIP_NUM_PROCESSES'])
else:
    os.environ['QUTIP_NUM_PROCESSES'] = str(qutip.settings.num_cpus)

if qutip.settings.num_cpus == 0:
    # if num_cpu is 0 set it to the available number of cores
    import qutip.hardware_info
    info =  qutip.hardware_info.hardware_info()
    if 'cpus' in info:
        qutip.settings.num_cpus = info['cpus']
    else:
        qutip.settings.num_cpus = multiprocessing.cpu_count()


# Find MKL library if it exists
import qutip._mkl



# -----------------------------------------------------------------------------
# Load configuration from environment variables: override defaults and
# configuration file.
#

# check for fortran mcsolver files
try:
    from qutip.fortran import mcsolve_f90
except:
    qutip.settings.fortran = False
else:
    qutip.settings.fortran = True

# -----------------------------------------------------------------------------
# Check that import modules are compatible with requested configuration
#


# Check for Matplotlib
try:
    import matplotlib
except:
    warnings.warn("matplotlib not found: Graphics will not work.")
else:
    del matplotlib

# -----------------------------------------------------------------------------
# Load modules
#

# core
from qutip.qobj import *
from qutip.states import *
from qutip.operators import *
from qutip.expect import *
from qutip.tensor import *
from qutip.superoperator import *
from qutip.superop_reps import *
from qutip.subsystem_apply import *
from qutip.graph import *

# graphics
from qutip.bloch import *
from qutip.visualization import *
from qutip.orbital import *
from qutip.bloch3d import *
from qutip.matplotlib_utilities import *

# library functions
from qutip.tomography import *
from qutip.wigner import *
from qutip.random_objects import *
from qutip.simdiag import *
from qutip.entropy import *
from qutip.metrics import *
from qutip.partial_transpose import *
from qutip.permute import *
from qutip.continuous_variables import *
from qutip.distributions import *
from qutip.three_level_atom import *

# evolution
from qutip.solver import *
from qutip.rhs_generate import *
from qutip.mesolve import *
from qutip.sesolve import *
from qutip.mcsolve import *
from qutip.stochastic import *
from qutip.essolve import *
from qutip.eseries import *
from qutip.propagator import *
from qutip.floquet import *
from qutip.bloch_redfield import *
from qutip.steadystate import *
from qutip.correlation import *
from qutip.countstat import *
from qutip.rcsolve import *
from qutip.nonmarkov import *
from qutip.interpolate import *

# quantum information
from qutip.qip import *

# utilities
from qutip.parallel import *
from qutip.utilities import *
from qutip.fileio import *
from qutip.about import *

# Setup pyximport 
import pyximport
os.environ['CFLAGS'] = '-O2 -w -ffast-math'
pyximport.install(setup_args={'include_dirs': [numpy.get_include()]})
del pyximport

# -----------------------------------------------------------------------------
# Clean name space
#
del os, sys, numpy, scipy, multiprocessing