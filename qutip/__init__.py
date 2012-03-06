#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import os,sys,platform,multiprocessing

#automatically set number of threads used by MKL
os.environ['MKL_NUM_THREADS']=str(multiprocessing.cpu_count())
os.environ['NUM_THREADS']=str(multiprocessing.cpu_count())

#
# default, use graphics (unless QUTIP_GRAPHICS is already set)
#
if not ('QUTIP_GRAPHICS' in os.environ):
    os.environ['QUTIP_GRAPHICS']="YES"

# default to no gui until run local is checked
if not ('QUTIP_GUI' in os.environ):
    os.environ['QUTIP_GUI']="NONE"

#check if being run remotely
if not ('DISPLAY' in os.environ):
    #no graphics if DISPLAY isn't set
    os.environ['QUTIP_GRAPHICS']="NO"
    os.environ['QUTIP_GUI']="NONE"

# check for windows platform
if sys.platform[0:3] == 'win':
    # graphics always available on windows
    os.environ['QUTIP_GRAPHICS']="YES"

#-Check for Matplotlib
try:
    import matplotlib
except:
    os.environ['QUTIP_GRAPHICS']="NO"


#if being run locally, check for installed gui modules
if os.environ['QUTIP_GRAPHICS']=="YES":
    try:
        import PySide
        os.environ['QUTIP_GUI']="PYSIDE"
    except:
        try:
            import PyQt4
            os.environ['QUTIP_GUI']="PYQT4"
        except:
            pass
#----------------------------------------------------
from scipy import *
import scipy.linalg as la
import scipy.sparse as sp
from qutip.Qobj import Qobj,shape,dims,dag,trans,sp_expm
from qutip.about import *

if os.environ['QUTIP_GRAPHICS']=="YES":
    from qutip.Bloch import Bloch
    from qutip.graph import hinton

from qutip.correlation import *
from qutip.clebsch import clebsch
from qutip.eseries import *

from qutip.demos import *
import qutip.examples

from qutip.entropy import *
from qutip.expect import *
from qutip.gates import *
from qutip.istests import *
from qutip.Odeoptions import Odeoptions
from qutip.Mcdata import Mcdata
from qutip.mcsolve import mcsolve
from qutip.metrics import fidelity,tracedist
import qutip.odeconfig
from qutip.Odedata import Odedata
from qutip.odesolve import odesolve
from qutip.essolve import *
from qutip.operators import *
from qutip.orbital import *
from qutip.parfor import *
from qutip.ptrace import ptrace

from qutip.propagator import *
from qutip.floquet import *

from qutip.qstate import *
from qutip.rand import *
from qutip.rhs_generate import rhs_generate
from qutip.simdiag import *
from qutip.sphereplot import *
from qutip.states import *
from qutip.steady import *
from qutip.superoperator import *
from qutip.tensor import *
from qutip.tidyup import tidyup
from qutip.wigner import *
from qutip.fileio import *
from qutip.bloch_redfield import *


