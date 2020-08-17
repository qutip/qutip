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

from .settings import settings
from . import version
from .version import version as __version__
from .utilities import _version2int

# -----------------------------------------------------------------------------
# Check for minimum requirements of dependencies, give the user a warning
# if the requirements aren't fulfilled
#

numpy_requirement = "1.12.0"
try:
    import numpy
    if _version2int(numpy.__version__) < _version2int(numpy_requirement):
        print("QuTiP warning: old version of numpy detected " +
              ("(%s), requiring %s." %
               (numpy.__version__, numpy_requirement)))
except:
    warnings.warn("numpy not found.")

scipy_requirement = "1.0.0"
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
    if ('QuTiP' in setup_file.readlines()[1][3:]) and version.release:
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
    # Setup pyximport
    from . import _pyxbuilder as pbldr
    pbldr.install(setup_args={'include_dirs': [numpy.get_include()]})
    del pbldr
except Exception:
    pass
else:
    del Cython


# -----------------------------------------------------------------------------
# cpu/process configuration
#
import multiprocessing

"""
# Check if environ flag for qutip processes is set
if 'QUTIP_NUM_PROCESSES' in os.environ:
    settings.num_cpus = int(os.environ['QUTIP_NUM_PROCESSES'])
else:
    os.environ['QUTIP_NUM_PROCESSES'] = str(settings.num_cpus)

if settings.num_cpus == 0:
    # if num_cpu is 0 set it to the available number of cores
    from . import hardware_info
    info = hardware_info.hardware_info()
    if 'cpus' in info:
        settings.num_cpus = info['cpus']
    else:
        try:
            settings.num_cpus = multiprocessing.cpu_count()
        except:
            settings.num_cpus = 1
"""


# Find MKL library if it exists
from .installsettings import *
from . import _mkl


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

from .core import *
from .solve import *

# graphics
from .bloch import *
from .visualization import *
from .orbital import *
from .bloch3d import *
from .matplotlib_utilities import *

# library functions
from .tomography import *
from .wigner import *
from .random_objects import *
from .simdiag import *
from .entropy import *
from .partial_transpose import *
from .continuous_variables import *
from .distributions import *
from .three_level_atom import *

########################################################################
# This section exists only for the deprecation warning of qip importation.
# It can be deleted for a major release.

# quantum information
from .qip import *
########################################################################

# utilities
from .parallel import *
from .utilities import *
from .fileio import *
from .about import *
from .cite import *

# Remove -Wstrict-prototypes from cflags
import setuptools
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

# -----------------------------------------------------------------------------
# Load user configuration if present: override defaults.
#
from . import configrc
if configrc.has_qutip_rc():
    settings.load()

# -----------------------------------------------------------------------------
# Clean name space
#
del os, sys, numpy, scipy, multiprocessing, distutils, configrc, warnings
