# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import os
import sys
import platform
import multiprocessing

import qutip.settings
import qutip._version
from qutip._version import version as __version__


#------------------------------------------------------------------------------
# Check for minimum requirements of dependencies, give the user a warning
# if the requirements aren't fulfilled
#
def version2int(version_string):
    str_list = version_string.split("-dev")[0].split("rc")[0].split("b")[0].split('.')
    return sum([int(d) * (100 ** (3 - n)) for n, d in enumerate(str_list[:3])])

numpy_requirement = "1.6.0"
try:
    import numpy
    if version2int(numpy.__version__) < version2int(numpy_requirement):
        print("QuTiP warning: old version of numpy detected " +
              ("(%s), requiring %s." %
               (numpy.__version__, numpy_requirement)))
except:
    print("QuTiP warning: numpy not found.")

scipy_requirement = "0.9.0"
try:
    import scipy
    if version2int(scipy.__version__) < version2int(scipy_requirement):
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
    if ('QuTiP' in setup_file.readlines()[1][3:]) and qutip._version.release:
        print("You are in the installation directory. " +
              "Change directories before running QuTiP.")
    setup_file.close()


#------------------------------------------------------------------------------
# setup the cython environment
#
_cython_requirement = "0.15.0"
try:
    import Cython
    if version2int(Cython.__version__) < version2int(_cython_requirement):
        print("QuTiP warning: old version of cython detected " +
              ("(%s), requiring %s." %
               (Cython.__version__, _cython_requirement)))

    import pyximport
    os.environ['CFLAGS'] = '-O3 -w'
    pyximport.install(setup_args={'include_dirs': [numpy.get_include()]})

except Exception as e:
    print("QuTiP warning: cython setup failed: " + str(e))


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

if not ('QUTIP_GUI' in os.environ):
    os.environ['QUTIP_GUI'] = qutip.settings.qutip_gui
else:
    qutip.settings.qutip_gui = os.environ['QUTIP_GUI']

# check if being run remotely
if not sys.platform in ['darwin', 'win32'] and not ('DISPLAY' in os.environ):
    # no graphics if DISPLAY isn't set
    os.environ['QUTIP_GRAPHICS'] = "NO"
    qutip.settings.qutip_graphics = "NO"
    os.environ['QUTIP_GUI'] = "NONE"
    qutip.settings.qutip_gui = "NONE"

# automatically set number of threads used by MKL
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['NUM_THREADS'] = str(multiprocessing.cpu_count())

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

# if being run locally, check for installed gui modules
if qutip.settings.qutip_graphics == 'YES':

    if qutip.settings.qutip_gui == "NONE":
        # no preference, try PYSIDE first, the PYQT4
        try:
            import PySide
            os.environ['QUTIP_GUI'] = "PYSIDE"
            qutip.settings.qutip_gui = "PYSIDE"
        except:
            try:
                import PyQt4
                os.environ['QUTIP_GUI'] = "PYQT4"
                qutip.settings.qutip_gui = "PYQT4"
            except:
                qutip.settings.qutip_gui = "NONE"

    elif qutip.settings.qutip_gui == "PYSIDE":
        # PYSIDE was requested
        try:
            import PySide
            os.environ['QUTIP_GUI'] = "PYSIDE"
            qutip.settings.qutip_gui = "PYSIDE"
        except:
            qutip.settings.qutip_gui = "NONE"

    elif qutip.settings.qutip_gui == "PYQT4":
        # PYQT4 was requested
        try:
            import PyQt4
            os.environ['QUTIP_GUI'] = "PYQT4"
            qutip.settings.qutip_gui = "PYQT4"
        except:
            qutip.settings.qutip_gui = "NONE"


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
from qutip.sparse import sp_eigs

# graphics
if qutip.settings.qutip_graphics == 'YES':
    from qutip.bloch import Bloch
    from qutip.visualization import (hinton, energy_level_diagram, wigner_cmap,
                                     sphereplot, fock_distribution,
                                     wigner_fock_distribution,
                                     plot_expectation_values)
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
from qutip.gates import *
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
from qutip.essolve import *
from qutip.eseries import *
from qutip.steady import *
from qutip.correlation import *
from qutip.propagator import *
from qutip.floquet import *
from qutip.bloch_redfield import *
from qutip.superop_reps import *
from qutip.subsystem_apply import subsystem_apply

# utilities
from qutip.utilities import *
from qutip.fileio import *
from qutip.demos import demos
from qutip.about import *
