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
from Qobj import *


def _ode_checks(H,c_ops,solver='me'):
    """
    Checks on time-dependent format.
    """
    h_const = 0
    h_func  = 0
    h_str   = []
    # check H for incorrect format
    if isinstance(H, Qobj):    
        h_const += 1
    elif isinstance(H, FunctionType):
        pass #n_func += 1
    elif isinstance(H, list):
        for k in xrange(len(H)):
            if isinstance(H[k], Qobj):
                h_const += 1
            elif isinstance(H[k], list):
                if len(H[k]) != 2 or not isinstance(H[k][0], Qobj):
                    raise TypeError("Incorrect hamiltonian specification")
                else:
                    if isinstance(H[k][1], FunctionType):
                        h_func += 1
                    elif isinstance(H[k][1], str):
                        h_str.append(k)
                    else:
                        raise TypeError("Incorrect hamiltonian specification")
    else:
        raise TypeError("Incorrect hamiltonian specification")
    
    # the the whole thing again for c_ops 
    c_const = 0
    c_func  = 0
    c_str   = []
    if isinstance(c_ops, list):
        for k in xrange(len(c_ops)):
            if isinstance(c_ops[k], Qobj):
                c_const += 1
            elif isinstance(c_ops[k], list):
                if len(c_ops[k]) != 2 or not isinstance(c_ops[k][0], Qobj):
                    raise TypeError("Incorrect collapse operator specification")
                else:
                    if isinstance(c_ops[k][1], FunctionType):
                        c_func += 1
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
    if (h_str > 0 and h_func > 0) or (c_str > 0 and c_func > 0):
        raise TypeError("Cannot mix string and function type time-dependence formats")       
    
    #check to see if Cython is installed and version is high enough.
    if h_str>0 or c_str>0:
        try:
            import Cython
        except:
            raise Exception("Unable to load Cython. Use Python function format.")
        else:
            if Cython.__version__ < '0.14':
                raise Exception("Cython version (%s) is too old.  Upgrade to version 0.14+" % Cython.__version__)
    
    if solver=='me':
        return [h_const+c_const,h_func+c_func,len(h_str)+len(c_str)]
    elif solver=='mc':
        #Time-indepdent problems
        if (h_func==h_str==0) and (c_func==c_str==0):
            time_type=0
        
        #Python function style Hamiltonian
        elif h_func>0:
            if len(c_func)==len(c_str)==0:
                time_type=10
            elif c_func>0:
                time_type=11
            elif c_str>0:
                time_type=12
        
        #list style Hamiltonian
        elif h_str>0:
            if c_func==len(c_str)==0:
                time_type=20
            elif c_func>0:
                time_type=21
            elif c_str>0:
                time_type=22
        
        #constant Hamiltonian, time-dependent collapse operators
        elif h_func==len(h_str)==0:
            if c_func>0:
                time_type=31
            elif len(c_str)>0:
                time_type=32
        
        return time_type,[h_const,h_func,h_str],[c_const,c_func,c_str]
        


def _args_check(H,h_inds,c_ops,c_inds,args):
    """
    Checks args variables to make sure they are not too close to the built in numpy math commands
    """
    keys=args.keys()
    func_list=array([func+'(' for func in dir(np.math)[4:-1]]) #add a '(' on the end to guarentee function is selected 
    if any(['e'==j for j in keys]) or any(['pi'==j for j in keys]):
        raise ValueError("'e' and 'pi' are not allowed arguments.")
    for k in keys:
        #checks if key is in np.math
        math_key=where([text.find(k)!=-1 for text in func_list])[0]
        if len(math_key)>0:
            #checks if any math functions with key in the string are in the Hamiltonian strings
            math_in_h_str=where(array([[H[x][1].find(jj)!=-1 for jj in func_list[math_key]] for x in h_inds]))[0]
            math_in_c_str=where(array([[c_ops[x][1].find(jj)!=-1 for jj in func_list[math_key]] for x in c_inds]))[0]
            #if the args and math function names are too close or equal, raise an error
            if (len(math_in_h_str)>0 or len(math_in_c_str)>0) and len(k)>2:
                raise Exception("Argument "+k+" is too close to, or equal to, a math function name used in Hamiltonian list. Switch argument name.")


def _args_sort(H,h_inds,c_ops,c_inds,args):
    """
    Sorts mcsolve args into two dicts, one
    for Hamiltonian, one for collapse operators.
    
    Takes indices from _ode_checks
    """
    keys=args.keys()
    
    in_h=where([any(array([H[x][1].find(k)!=-1 for x in h_inds])) for k in keys])[0]
    in_c=where([any(array([c_ops[x][1].find(k)!=-1 for x in c_inds])) for k in keys])[0]
    
    h_args={ k : v for k,v in args.iteritems() if k in [keys[j] for j in in_h] }
    c_args={ k : v for k,v in args.iteritems() if k in [keys[j] for j in in_c] }
    
    if not any(h_args):
        h_args=None
    if not any(c_args):
        c_args=None
    return h_args,c_args
 
    

