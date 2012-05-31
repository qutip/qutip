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
import numpy as np
from types import FunctionType
from qutip.Qobj import *


def _ode_checks(H,c_ops,solver='me'):
    """
    Checks on time-dependent format.
    """
    h_const = []
    h_func  = []
    h_str   = []
    # check H for incorrect format
    if isinstance(H, Qobj):    
        pass
    elif isinstance(H, FunctionType):
        pass #n_func += 1
    elif isinstance(H, list):
        for k in range(len(H)):
            if isinstance(H[k], Qobj):
                h_const.append(k)
            elif isinstance(H[k], list):
                if len(H[k]) != 2 or not isinstance(H[k][0], Qobj):
                    raise TypeError("Incorrect hamiltonian specification")
                else:
                    if isinstance(H[k][1], FunctionType):
                        h_func.append(k)
                    elif isinstance(H[k][1], str):
                        h_str.append(k)
                    else:
                        raise TypeError("Incorrect hamiltonian specification")
    else:
        raise TypeError("Incorrect hamiltonian specification")
    
    # the the whole thing again for c_ops 
    c_const = []
    c_func  = []
    c_str   = []
    if isinstance(c_ops, list):
        for k in range(len(c_ops)):
            if isinstance(c_ops[k], Qobj):
                c_const.append(k)
            elif isinstance(c_ops[k], list):
                if len(c_ops[k]) != 2 or not isinstance(c_ops[k][0], Qobj):
                    raise TypeError("Incorrect collapse operator specification")
                else:
                    if isinstance(c_ops[k][1], FunctionType):
                        c_func.append(k)
                    elif isinstance(c_ops[k][1], str):
                        c_str.append(k)
                    else:
                        raise TypeError("Incorrect collapse operator specification")
    else:
        raise TypeError("Incorrect collapse operator specification")       

    #
    # if n_str == 0 and n_func == 0:
    #     # no time-dependence at all
    #
    if (len(h_str) > 0 and len(h_func) > 0) or (len(c_str) > 0 and len(c_func) > 0):
        raise TypeError("Cannot mix string and function type time-dependence formats")       
    
    #check to see if Cython is installed and version is high enough.
    if len(h_str) > 0 or len(c_str) > 0:
        try:
            import Cython
        except:
            raise Exception("Unable to load Cython. Use Python function format.")
        else:
            if Cython.__version__ < '0.14':
                raise Exception("Cython version (%s) is too old.  Upgrade to version 0.14+" % Cython.__version__)
    
    if solver=='me':
        return [len(h_const+c_const),len(h_func)+len(c_func),len(h_str)+len(c_str)]
    elif solver=='mc':
        
        #   H      C_ops    #
        #   --     -----    --
        #   NO      NO      00
        #   NO     STR      01
        #   NO     FUNC     02
        #
        #   STR    NO       10
        #   STR    STR      11
        #
        #   FUNC   NO       20
        #
        #   FUNC   FUNC     22
        
        if isinstance(H, FunctionType):
            time_type=3
        #Time-indepdent problems
        elif (len(h_func)==len(h_str)==0) and (len(c_func)==len(c_str)==0):
            time_type=0
        
        #constant Hamiltonian, time-dependent collapse operators
        elif len(h_func)==len(h_str)==0:
            if len(c_str)>0:
                time_type=1
            elif len(c_func)>0:
                time_type=2
            else:
                raise Exception("Error determining time-dependence.")
        
        
        #list style Hamiltonian
        elif len(h_str)>0:
            if len(c_func)==len(c_str)==0:
                time_type=10
            elif len(c_str)>0:
                time_type=11
            else:
                raise Exception("Error determining time-dependence.")
    
    
        #Python function style Hamiltonian
        elif len(h_func)>0:
            if len(c_func)==len(c_str)==0:
                time_type=20
            elif len(c_func)>0:
                time_type=22
            else:
                raise Exception("Error determining time-dependence.")
        
        return time_type,[h_const,h_func,h_str],[c_const,c_func,c_str]
        

 
    

