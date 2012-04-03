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

from types import FunctionType
from Qobj import *


def _ode_checks(H,c_ops,solver='me'):
    # do some basic sanity checks on the format of H and c_ops: this should
    # be moved to a function that can be reused.
    h_const = 0
    h_func  = 0
    h_str   = 0
    # check H for incorrect format
    if isinstance(H, Qobj):    
        h_const += 1
    elif isinstance(H, FunctionType):
        pass #n_func += 1
    elif isinstance(H, list):
        for h in H:
            if isinstance(h, Qobj):
                h_const += 1
            elif isinstance(h, list):
                if len(h) != 2 or not isinstance(h[0], Qobj):
                    raise TypeError("Incorrect hamiltonian specification")
                else:
                    if isinstance(h[1], FunctionType):
                        h_func += 1
                    elif isinstance(h[1], str):
                        h_str += 1
                    else:
                        raise TypeError("Incorrect hamiltonian specification")
    else:
        raise TypeError("Incorrect hamiltonian specification")
    
    # the the whole thing again for c_ops 
    c_const = 0
    c_func  = 0
    c_str   = 0
    if isinstance(c_ops, list):
        for c in c_ops:
            if isinstance(c, Qobj):
                c_const += 1
            elif isinstance(c, list):
                if len(c) != 2 or not isinstance(c[0], Qobj):
                    raise TypeError("Incorrect collapse operator specification")
                else:
                    if isinstance(c[1], FunctionType):
                        c_func += 1
                    elif isinstance(c[1], str):
                        c_str += 1
                    else:
                        raise TypeError("Incorrect collapse operator specification")
    else:
        raise TypeError("Incorrect collapse operator specification")       

    #
    # if n_str == 0 and n_func == 0:
    #     # no time-dependence at all
    #
    if (h_str > 0 and h_func > 0) or (c_str > 0 and c_func > 0):
        raise TypeError("Cannot mix string and function type time-dependence formats")       
    
    if solver=='me':
        return [h_const+c_const,h_func+c_func,h_str+c_str]
    elif solver=='mc':
        return [h_const,h_func,h_str],[c_const,c_func,c_str]