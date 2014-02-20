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
import os
import sys
import platform
import multiprocessing

import qutip.settings
import qutip.version 
from qutip.version import version as __version__


#------------------------------------------------------------------------------
# Check for minimum requirements of dependencies, give the user a warning
# if the requirements aren't fulfilled
#
def _version2int(version_string):
    str_list = version_string.split("-dev")[0].split("rc")[0].split("b")[0].split("post")[0].split('.')
    return sum([int(d if len(d) > 0 else 0) * (100 ** (3 - n)) for n, d in enumerate(str_list[:3])])

numpy_requirement = "1.6.0"
try:
    import numpy
    if _version2int(numpy.__version__) < _version2int(numpy_requirement):
        print("QuTiP warning: old version of numpy detected " +
              ("(%s), requiring %s." %
               (numpy.__version__, numpy_requirement)))
except:
    print("QuTiP warning: numpy not found.")

scipy_requirement = "0.11.0"
try:
    import scipy
    if _version2int(scipy.__version__) < _version2int(scipy_requirement):
        print("QuTiP warning: old version of scipy detected " +
              ("(%s), requiring %s." %
               (scipy.__version__, scipy_requirement)))
except:
    print("QuTiP warning: scipy not found.")

#------------------------------------------------------------------------------
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


#------------------------------------------------------------------------------
# setup the cython environment
#
_cython_requirement = "0.15.0"
try:
    import Cython
    if _version2int(Cython.__version__) < _version2int(_cython_requirement):
        print("QuTiP warning: old version of cython detected " +
              ("(%s), requiring %s." %
               (Cython.__version__, _cython_requirement)))

    import pyximport
    os.environ['CFLAGS'] = '-O3 -w -ffast-math -march=native -mfpmath=sse'
    pyximport.install(setup_args={'include_dirs': [numpy.get_include()]})

except Exception as e:
    print("QuTiP warning: Cython setup failed: " + str(e))


#------------------------------------------------------------------------------
# default configuration settings
#

#load cpus
from qutip.hardware_info import hardware_info
info = hardware_info()
if 'cpus' in info:
    qutip.settings.num_cpus = info['cpus']
else:
    qutip.settings.num_cpus = multiprocessing.cpu_count()

qutip.settings.qutip_graphics = "YES"
qutip.settings.qutip_gui = "NONE"


#------------------------------------------------------------------------------
# Load user configuration if present: override defaults.
#
try:
    qutip_rc_file = os.environ['HOME'] + "/.qutiprc"
    qutip.settings.load_rc_file(qutip_rc_file)

except Exception as e:
    pass


#------------------------------------------------------------------------------
# Load configuration from environment variables: override defaults and
# configuration file.
#

if not ('QUTIP_GRAPHICS' in os.environ):
    os.environ['QUTIP_GRAPHICS'] = qutip.settings.qutip_graphics
else:
    qutip.settings.qutip_graphics = os.environ['QUTIP_GRAPHICS']

# check if being run remotely
if not sys.platform in ['darwin', 'win32'] and not ('DISPLAY' in os.environ):
    # no graphics if DISPLAY isn't set
    os.environ['QUTIP_GRAPHICS'] = "NO"
    qutip.settings.qutip_graphics = "NO"

# automatically set number of threads used by MKL and openblas to 1
# prevents errors when running things in parallel.  Should be set 
# by user directly in a script or notebook if >1 is needed.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

try:
    from qutip.fortran import qutraj_run
except:
    qutip.settings.fortran = False
else:
    qutip.settings.fortran = True
    from qutip.fortran import *
#------------------------------------------------------------------------------
# Check that import modules are compatible with requested configuration
#

# Check for Matplotlib
try:
    import matplotlib
except:
    os.environ['QUTIP_GRAPHICS'] = "NO"
    qutip.settings.qutip_graphics = 'NO'

#------------------------------------------------------------------------------
# Load modules
#

# core
from qutip.qobj import *
from qutip.states import *
from qutip.operators import *
from qutip.expect import *
from qutip.superoperator import *
from qutip.tensor import *
from qutip.parfor import *
from qutip.sparse import *

# graphics
if qutip.settings.qutip_graphics == 'YES':
    from qutip.bloch import Bloch
    from qutip.visualization import (hinton, energy_level_diagram, wigner_cmap,
                                     sphereplot, fock_distribution,
                                     wigner_fock_distribution,
                                     plot_expectation_values,
                                     plot_spin_distribution_2d,
                                     plot_spin_distribution_3d)
    from qutip.orbital import *
    # load mayavi dependent functions if available
    try:
        import mayavi
    except:
        pass
    else:
        from qutip.bloch3d import Bloch3d

# library functions
from qutip.tomography import *
from qutip.wigner import *
from qutip.random_objects import *
from qutip.simdiag import *
from qutip.entropy import (entropy_vn, entropy_linear, entropy_mutual,
                           concurrence, entropy_conditional)
from qutip.metrics import fidelity, tracedist
from qutip.partial_transpose import partial_transpose
from qutip.continuous_variables import *
from qutip.distributions import *

# evolution
from qutip.odeconfig import odeconfig
from qutip.odeoptions import Odeoptions
from qutip.odedata import Odedata
from qutip.rhs_generate import rhs_generate, rhs_clear
from qutip.mesolve import mesolve, odesolve
from qutip.sesolve import sesolve
from qutip.mcsolve import mcsolve
from qutip.stochastic import ssesolve, sepdpsolve, smesolve, smepdpsolve
from qutip.essolve import *
from qutip.eseries import *
from qutip.steadystate import *
from qutip.correlation import *
from qutip.propagator import *
from qutip.floquet import *
from qutip.bloch_redfield import *
from qutip.superop_reps import *
from qutip.subsystem_apply import subsystem_apply
from qutip.sparse import sparse_bandwidth
from qutip.graph import *

# quantum information
from qutip.quantum_info import *

# utilities
from qutip.utilities import *
from qutip.fileio import *
from qutip.about import *

