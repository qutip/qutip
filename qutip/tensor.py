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
import scipy.sparse as sp
from scipy.linalg import *
from qobj import qobj

def tensor(*args):
    if not any(args): #error: needs at least one input
        raise TypeError("Requires at least one input arguement")
    if isinstance(args[0],list):#checks if input is list of qobjs
        args=args[0]

    items=len(args) #number of inputs
    num_qobjs=sum([isinstance(args[k],qobj) for k in xrange(items)])#check to see if inputs are qobj's
    if num_qobjs!=items: #raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")
    if items==1:# if only one qobj, do nothing
        return args[0]
    #set initial qobj values to those of arg[0]
    dat=args[0].data
    dim=args[0].dims
    shp=args[0].shape
    for k in xrange(items-1): #cycle over all items
        dat=sp.kron(dat,args[k+1].data) #sparse Kronecker product
        dim=[dim[0]+args[k+1].dims[0],dim[1]+args[k+1].dims[1]] #append dimensions of qobjs
        shp=[dat.shape[0],dat.shape[1]] #new shape of matrix
    out=qobj()
    out.data=dat
    out.dims=dim
    out.shape=shp
    return qobj(out) #returns output qobj
