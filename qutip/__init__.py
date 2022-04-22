import os
import warnings

import qutip.settings
import qutip.version
from qutip.version import version as __version__

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
from .installsettings import *
from . import _mkl


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

from .core import *
from .solve import *
from .solver.brmesolve import *

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

qutip.settings.install.read_only_options["idxint_size"] = data.base.idxint_size

# -----------------------------------------------------------------------------
# Load user configuration if present: override defaults.
#
from . import configrc
if configrc.has_qutip_rc():
    settings.load()

# -----------------------------------------------------------------------------
# Clean name space
#
del os, warnings
