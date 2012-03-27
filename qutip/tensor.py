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
"""
Module for the creation of composite quantum
objects via the tensor product.

"""
from scipy import *
import scipy.sparse as sp
from scipy.linalg import *
from qutip.Qobj import Qobj
from qutip.tidyup import tidyup
import qutip.settings

def tensor(*args):
    """Calculates the tensor product of input operators. 
    
    Parameters
    ----------
    args : array_like 
        ``list`` or ``array`` of quantum objects for tensor product.
        
    Returns
    --------
    obj : qobj
        A composite quantum object.
    
    Examples
    --------    
    >>> tensor([sigmax(), sigmax()])
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = True
    Qobj data = 
    [[ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]]

    """
    if not args:
        raise TypeError("Requires at least one input argument")
    num_args=len(args)
    step=0
    for n in xrange(num_args):
        if isinstance(args[n],Qobj):
            qos=args[n]
            if step==0:
                dat=qos.data
                dim=qos.dims
                shp=qos.shape
                step=1
            else:
                dat=sp.kron(dat,qos.data, format='csr') #sparse Kronecker product
                dim=[dim[0]+qos.dims[0],dim[1]+qos.dims[1]] #append dimensions of Qobjs
                shp=[dat.shape[0],dat.shape[1]] #new shape of matrix
                
        elif isinstance(args[n],(list,ndarray)):#checks if input is list/array of Qobjs
            qos=args[n]
            items=len(qos) #number of inputs
            if not all([isinstance(k,Qobj) for k in qos]): #raise error if one of the inputs is not a quantum object
                raise TypeError("One of inputs is not a quantum object")
            if items==1:# if only one Qobj, do nothing
                if step==0: 
                    dat=qos[0].data
                    dim=qos[0].dims
                    shp=qos[0].shape
                    step=1
                else:
                    dat=sp.kron(dat,qos[0].data, format='csr') #sparse Kronecker product
                    dim=[dim[0]+qos[0].dims[0],dim[1]+qos[0].dims[1]] #append dimensions of Qobjs
                    shp=[dat.shape[0],dat.shape[1]] #new shape of matrix
            elif items!=1:
                if step==0:
                    dat=qos[0].data
                    dim=qos[0].dims
                    shp=qos[0].shape
                    step=1
                for k in xrange(items-1): #cycle over all items
                    dat=sp.kron(dat,qos[k+1].data, format='csr') #sparse Kronecker product
                    dim=[dim[0]+qos[k+1].dims[0],dim[1]+qos[k+1].dims[1]] #append dimensions of Qobjs
                    shp=[dat.shape[0],dat.shape[1]] #new shape of matrix
    out=Qobj()
    out.data=dat
    out.dims=dim
    out.shape=shp
    if qutip.settings.auto_tidyup:
        return Qobj(out).tidyup() #returns tidy Qobj
    else:
        return Qobj(out)
