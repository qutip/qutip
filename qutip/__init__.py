import os
import warnings

import qutip.settings
import qutip.version
from qutip.version import version as __version__

# -----------------------------------------------------------------------------
# Check if we're in IPython.
try:
    __IPYTHON__
    qutip.settings.ipython = True
except NameError:
    qutip.settings.ipython = False


# -----------------------------------------------------------------------------
# Look to see if we are running with OPENMP
#
# Set environ variable to determin if running in parallel mode
# (i.e. in parfor or parallel_map)
os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'

try:
    from qutip.cy.openmp.parfuncs import spmv_csr_openmp
except ImportError:
    qutip.settings.has_openmp = False
else:
    qutip.settings.has_openmp = True
    # See Pull #652 for why this is here.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import platform
import scipy
from packaging import version as pac_version
from qutip.utilities import _blas_info

is_old_scipy = pac_version.parse(scipy.__version__) < pac_version.parse("1.5")
qutip.settings.eigh_unsafe = (
    # macOS OpenBLAS eigh is unstable, see #1288
    (_blas_info() == "OPENBLAS" and platform.system() == 'Darwin')
    # The combination of scipy<1.5 and MKL causes wrong results when calling
    # eigh for big matrices.  See #1495, #1491 and #1498.
    or (is_old_scipy and (_blas_info() == 'INTEL MKL'))
)

del platform, _blas_info, scipy, pac_version, is_old_scipy
# -----------------------------------------------------------------------------
# setup the cython environment
#
try:
    import Cython as _Cython
except ImportError:
    pass
else:
    from qutip.utilities import _version2int
    import sys
    _cy_require = "0.29.20"
    _cy_unsupported = "3.0.0"
    if _version2int(_Cython.__version__) < _version2int(_cy_require):
        warnings.warn(
            "Old version of Cython detected: needed {}, got {}."
            .format(_cy_require, _Cython.__version__)
        )
    elif _version2int(_Cython.__version__) >= _version2int(_cy_unsupported):
        warnings.warn(
            "The new version of Cython, (>= 3.0.0) is not supported."
            .format(_Cython.__version__)
        )
    elif _version2int(sys.version.split()[0]) >= _version2int("3.12.0"):
        warnings.warn(
            "Runtime cython compilation does not work on Python 3.12."
        )
    else:
        # Setup pyximport
        import qutip.cy.pyxbuilder as _pyxbuilder
        _pyxbuilder.install()
        del _pyxbuilder, _Cython, _version2int
        qutip.settings.has_cython = True
    del sys


# -----------------------------------------------------------------------------
# cpu/process configuration
#
from qutip.utilities import available_cpu_count

# Check if environ flag for qutip processes is set
if 'QUTIP_NUM_PROCESSES' in os.environ:
    qutip.settings.num_cpus = int(os.environ['QUTIP_NUM_PROCESSES'])
else:
    qutip.settings.num_cpus = available_cpu_count()
    os.environ['QUTIP_NUM_PROCESSES'] = str(qutip.settings.num_cpus)

del available_cpu_count


# Find MKL library if it exists
import qutip._mkl


# -----------------------------------------------------------------------------
# Check that import modules are compatible with requested configuration
#

# Check for Matplotlib
try:
    import matplotlib
except ImportError:
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
from qutip.krylovsolve import *
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

# lattice models
from qutip.lattice import *
from qutip.topology import *

########################################################################
# This section exists only for the deprecation warning of qip importation.
# It can be deleted for a major release.

# quantum information
from qutip.qip import *
########################################################################

# utilities
from qutip.parallel import *
from qutip.utilities import *
from qutip.fileio import *
from qutip.about import *
from qutip.cite import *

# -----------------------------------------------------------------------------
# Load user configuration if present: override defaults.
#
import qutip.configrc
has_rc, rc_file = qutip.configrc.has_qutip_rc()

# Read the OpenMP threshold out if it already exists, or calibrate and save it
# if it doesn't.
if qutip.settings.has_openmp:
    _calibrate_openmp = qutip.settings.num_cpus > 1
    if has_rc:
        _calibrate_openmp = (
            _calibrate_openmp
            and not qutip.configrc.has_rc_key('openmp_thresh', rc_file=rc_file)
        )
    else:
        qutip.configrc.generate_qutiprc()
        has_rc, rc_file = qutip.configrc.has_qutip_rc()
    if _calibrate_openmp:
        print('Calibrating OpenMP threshold...')
        from qutip.cy.openmp.bench_openmp import calculate_openmp_thresh
        thresh = calculate_openmp_thresh()
        qutip.configrc.write_rc_key('openmp_thresh', thresh, rc_file=rc_file)
        del calculate_openmp_thresh

# Load the config file
if has_rc:
    qutip.configrc.load_rc_config(rc_file)

# -----------------------------------------------------------------------------
# Clean name space
#
del os, warnings
