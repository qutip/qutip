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
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import os
import multiprocessing
import sys
##
# @mainpage QuTiP: Quantum Toolbox in Python
#
# Software package for simulation of quantum
# systems using python, scipy and numpy (derived from the Quantum Optics toolbox
# for MATLAB). 
#
# These pages contains automatically generated API documentation.
#

#automatically set number of threads used by MKL
os.environ['MKL_NUM_THREADS']=str(multiprocessing.cpu_count())

#
# default, use graphics (unless QUTIP_GRPAHICS is already set)
#
if not os.environ.has_key('QUTIP_GRAPHICS'):
    os.environ['QUTIP_GRAPHICS']="YES"

if sys.platform=='linux2':
    if not os.environ.has_key('DISPLAY'):
        # in X, no graphics if DISPLAY isn't set
        os.environ['QUTIP_GRAPHICS']="NO"

from scipy import *
import scipy.linalg as la
import scipy.sparse as sp
from Qobj import Qobj,shape,dims,dag,trans,isherm,sp_expm
from about import *
from basis import *
from cnot import *
from expect import *
from fredkin import *
from istests import *
from jmat import *
from list2ind import *
from m2trace import *
from operators import *
from qstate import *
from selct import *
from snot import *
from spost import *
from spre import *
from tensor import *
from toffoli import *
from uminus import *
from uplus import *
from wigner import *
from fseries import *
from fstidy import *
from steady import *
from probss import *
from parfor import *
from orbital import *
from mcsolve import *
from mcoptions import Mcoptions
from ode_solve import *
from eseries import *
from mcsolve import *
from ode2es import *
from states import *
from correlation import *
from metrics import fidelity,trace_dist

if os.environ['QUTIP_GRAPHICS'] == "YES":
    from sphereplot import *

