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
from scipy import *
from Qobj import Qobj

##
# @package basis
# Generate the vector representation of a number state.
#

def basis(N,*args):
    """
    @brief Generate the vector representation of a number state.

    a subtle incompability with the quantum optics toolbox: here
        basis(N, 0) = ground state
    but in QO toolbox:
        basis(N, 1) = ground state

    @param N the number of states
    """
    if logical_or(not isinstance(N,int),N<0):#check if N is int and N>0
        raise ValueError("N must be integer N>=0")
    if not any(args):#if no args then assume vacuum state 
        args=0
    if not isinstance(args,int):#if input arg!=0
        if not isinstance(args[0],int):#check if args is not int
            raise ValueError("need integer for basis vector index")
        args=args[0]
    if logical_or(args<0,args>N-1): #check if args is within bounds
        raise ValueError("basis vector index need to be in 0=<indx<=N-1")
    bas=zeros([N,1]) #column vector of zeros
    bas[args]=1 # 1 located at position args
    return Qobj(bas) #return Qobj

