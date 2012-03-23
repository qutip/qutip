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
This module is a collection of random state and operator generators.
The sparsity of the ouput Qobj's is controlled by varing the  
`density` parameter.

"""
import numpy as np
from numpy.random import random
import scipy.linalg as la
import scipy.sparse as sp
from Qobj import *



def rand_herm(N,density=0.75,dims=None):
    """Creates a random NxN sparse Hermitian quantum object.
    
    Uses :math:`H=X+X^{+}` where :math:`X` is
    a randomly generated quantum operator with a given `density`.
    
    Parameters
    ----------
    N : int
        Shape of output quantum operator.
    density : float
        Density etween [0,1] of output Hermitian operator.
    
    Returns
    -------
    oper : qobj
        NxN Hermitian quantum operator.
        
    Other Parameters
    ----------------
    dims : list 
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].
    
    
    """
    if dims:
        _check_dims(dims,N,N)
    # to get appropriate density of output
    # Hermitian operator must convert via:
    herm_density=2.0*arcsin(density)/pi
    
    X = sp.rand(N,N,herm_density,format='csr')
    X.data=X.data-0.5
    Y=X.copy()
    Y.data=1.0j*random(len(X.data))-(0.5+0.5j)
    X=X+Y
    X=Qobj(X)
    if dims:
        return Qobj((X+X.dag())/2.0,dims=dims,shape=[N,N])
    else:
        return Qobj((X+X.dag())/2.0)


def rand_unitary(N,density=0.75,dims=None):
    """Creates a random NxN sparse unitary quantum object.
    
    Uses :math:`\exp(-iH)` where H is a randomly generated 
    Hermitian operator.
    
    Parameters
    ----------
    N : int
        Shape of output quantum operator.
    density : float
        Density between [0,1] of output Unitary operator.
    
    Returns
    -------
    oper : qobj
        NxN Unitary quantum operator.
        
    Other Parameters
    ----------------
    dims : list 
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].
    
    
    """
    if dims:
        _check_dims(dims,N,N)
    U=(-1.0j*rand_herm(N,density)).expm()
    if dims:
        return Qobj(U,dims=dims,shape=[N,N])
    else:
        return Qobj(U)


def rand_ket(N,density=0.75,dims=None):
    """Creates a random Nx1 sparse ket vector.
    
    Parameters
    ----------
    N : int
        Number of rows for output quantum operator.
    density : float
        Density between [0,1] of output ket state.
    
    Returns
    -------
    oper : qobj
        Nx1 ket state quantum operator.
        
    Other Parameters
    ----------------
    dims : list 
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[1]].
    
    
    """
    if dims:
        _check_dims(dims,N,1)
    X = sp.rand(N,1,density,format='csr')
    X.data=X.data-0.5
    Y=X.copy()
    Y.data=1.0j*random(len(X.data))-(0.5+0.5j)
    X=X+Y
    X=Qobj(X)
    if dims:
        return Qobj(X/X.norm(),dims=dims,shape=[N,1])
    else:
        return Qobj(X/X.norm())


def rand_dm(N,density=0.75,pure=False,dims=None):
    """Creates a random NxN density matrix.
    
    Parameters
    ----------
    N : int
        Shape of output density matrix.
    density : float
        Density etween [0,1] of output density matrix.
    
    
    Returns
    -------
    oper : qobj
        NxN density matrix quantum operator.
    
        
    Other Parameters
    ----------------
    dims : list 
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].
    
    .. note::
    
        For small density matricies, choosing a low `density` will result in an error
        as no diagonal elements will be generated such that :math:`Tr(\rho)=1`.
    
    
    """
    if dims:
        _check_dims(dims,N,N)
    if pure:
        dm_density=sqrt(density)
        psi=rand_ket(N,dm_density)
        rho=psi*psi.dag()
    else:
        non_zero=0
        tries=0
        while non_zero==0 and tries<10: 
            H = rand_herm(N,density)
            non_zero=sum(H.tr())
            tries+=1
        if tries>=10:
            raise ValueError("Requested density is too low to generate density matrix.")
        H=sp.triu(H.data,format='csr')#take only upper triangle
        rho = 0.5*sp.eye(N,N,format='csr')*(H+H.conj().T)
        rho=Qobj(rho)
    if dims:
        return Qobj(rho/rho.tr(),dims=dims,shape=[N,N])
    else:
        return Qobj(rho/rho.tr())



def _check_dims(dims,N1,N2):
    if len(dims)!=2:
        raise Exception("Qobj dimensions must be list of length 2.")
    if (not isinstance(dims[0],list)) or (not isinstance(dims[1],list)):
        raise TypeError("Qobj dimension components must be lists. i.e. dims=[[N],[N]]")
    if prod(dims[0])!=N1 or prod(dims[1])!=N2:
        raise ValueError("Qobj dimensions must match matrix shape.")
    if len(dims[0])!=len(dims[1]):
        raise TypeError("Qobj dimension components must have same length.")
