import os
import warnings

import qutip.settings
from qutip.settings import settings
import qutip.version
from qutip.version import version as __version__
# -----------------------------------------------------------------------------
# Look to see if we are running with OPENMP
#
# Set environ variable to determin if running in parallel mode
# (i.e. in parfor or parallel_map)
os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'


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
from .solver import *
from .solver import nonmarkov
import qutip.piqs.piqs as piqs

# graphics
from .bloch import *
from .visualization import *
from .animation import *
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
from . import measurement

# utilities
from .utilities import *
from .fileio import *
from .about import *
from .cite import *

# -----------------------------------------------------------------------------
# Clean name space
#
del os, warnings
