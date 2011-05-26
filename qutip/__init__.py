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
##
# @mainpage QuTiP: Quantum Toolbox in Python
#
# Software package for simulation of quantum
# systems using python, scipy and numpy (derived from the Quantum Optics toolbox
# for MATLAB). 
#
# These pages contains automatically generated API documentation.
#
import os,sys,multiprocessing
#automatically set number of threads used by MKL
os.environ['MKL_NUM_THREADS']=str(multiprocessing.cpu_count())

#
# default, use graphics (unless QUTIP_GRPAHICS is already set)
#
if not os.environ.has_key('QUTIP_GRAPHICS'):
    os.environ['QUTIP_GRAPHICS']="YES"

#check if being run remotely
if not os.environ.has_key('DISPLAY'):
    #no graphics if DISPLAY isn't set
    os.environ['QUTIP_GRAPHICS']="NO"
    os.environ['QUTIP_GUI']="NONE"


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
            os.environ['QUTIP_GRAPHICS']="NO"
            os.environ['QUTIP_GUI']="NONE"
#----------------------------------------------------
from scipy import *
import scipy.linalg as la
import scipy.sparse as sp
from Qobj import Qobj,shape,dims,dag,trans,sp_expm
from about import *
from basis import *
from cnot import *
from expect import *
from fredkin import *
from istests import *
from list2ind import *
from m2trace import *
from operators import *
from superoperator import *
from qstate import *
from selct import *
from snot import *
from tensor import *
from toffoli import *
from wigner import *
from fseries import *
from steady import *
from parfor import *
from orbital import *
from mcsolve import *
from Mcoptions import Mcoptions
from ode_solve import *
from eseries import *
from ode2es import *
from states import *
from correlation import *
from propagator import *
from metrics import fidelity,trace_dist
from Bloch import Bloch


if os.environ['QUTIP_GRAPHICS'] == "YES":
    from sphereplot import *
    from gui import *

