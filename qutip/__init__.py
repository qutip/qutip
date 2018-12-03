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

numpy_requirement = "1.8.0"
try:
    import numpy
    if _version2int(numpy.__version__) < _version2int(numpy_requirement):
        print("QuTiP warning: old version of numpy detected " +
              ("(%s), requiring %s." %
               (numpy.__version__, numpy_requirement)))
except:
    warnings.warn("numpy not found.")

scipy_requirement = "0.15.0"
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
_cython_requirement = "0.21.0"
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
# Look to see if we are running with OPENMP
#
# Set environ variable to determin if running in parallel mode
# (i.e. in parfor or parallel_map)
os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'

try:
    from qutip.cy.openmp.parfuncs import spmv_csr_openmp
except:
    qutip.settings.has_openmp = False
else:
    qutip.settings.has_openmp = True
    # See Pull #652 for why this is here.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
        try:
            qutip.settings.num_cpus = multiprocessing.cpu_count()
        except:
            qutip.settings.num_cpus = 1


# Find MKL library if it exists
import qutip._mkl


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
from qutip.qobjevo import *
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
from qutip.cy.br_tensor import bloch_redfield_tensor
from qutip.steadystate import *
from qutip.correlation import *
from qutip.countstat import *
from qutip.rcsolve import *
from qutip.nonmarkov import *
from qutip.interpolate import *
from qutip.scattering import *

# quantum information
from qutip.qip import *

# utilities
from qutip.parallel import *
from qutip.utilities import *
from qutip.fileio import *
from qutip.about import *
from qutip.cite import *

# Remove -Wstrict-prototypes from cflags
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

# Setup pyximport
import qutip.cy.pyxbuilder as pbldr
pbldr.install(setup_args={'include_dirs': [numpy.get_include()]})
del pbldr

# -----------------------------------------------------------------------------
# Load user configuration if present: override defaults.
#
import qutip.configrc
has_rc, rc_file = qutip.configrc.has_qutip_rc()

# Make qutiprc and benchmark OPENMP if has_rc = False
if qutip.settings.has_openmp and (not has_rc):
    from qutip.cy.openmp.bench_openmp import calculate_openmp_thresh
    #bench OPENMP
    print('Calibrating OPENMP threshold...')
    thrsh = calculate_openmp_thresh()
    qutip.configrc.generate_qutiprc()
    has_rc, rc_file = qutip.configrc.has_qutip_rc()
    if has_rc:
        qutip.configrc.write_rc_key(rc_file, 'openmp_thresh', thrsh)
# Make OPENMP if has_rc but 'openmp_thresh' not in keys
elif qutip.settings.has_openmp and has_rc:
    from qutip.cy.openmp.bench_openmp import calculate_openmp_thresh
    has_omp_key = qutip.configrc.has_rc_key(rc_file, 'openmp_thresh')
    if not has_omp_key:
        print('Calibrating OPENMP threshold...')
        thrsh = calculate_openmp_thresh()
        qutip.configrc.write_rc_key(rc_file, 'openmp_thresh', thrsh)

# Load the config file
if has_rc:
    qutip.configrc.load_rc_config(rc_file)

# -----------------------------------------------------------------------------
# Clean name space
#
del os, sys, numpy, scipy, multiprocessing, distutils
