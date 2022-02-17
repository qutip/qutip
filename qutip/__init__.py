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

# -----------------------------------------------------------------------------
# setup the cython environment
#
try:
    import Cython as _Cython
except ImportError:
    pass
else:
    from qutip.utilities import _version2int
    _cy_require = "0.29.20"
    if _version2int(_Cython.__version__) < _version2int(_cy_require):
        warnings.warn(
            "Old version of Cython detected: needed {}, got {}."
            .format(_cy_require, _Cython.__version__)
        )
    # Setup pyximport
    from qutip import _pyxbuilder
    _pyxbuilder.install()
    del _pyxbuilder, _Cython, _version2int


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

# -----------------------------------------------------------------------------
# Clean name space
#
del os, warnings
