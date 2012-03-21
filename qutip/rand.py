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
from Qobj import *
import scipy.linalg as la
import numpy as np
from numpy.random import random
import scipy.sparse as sp

def rand_herm(N,density=0.75):
    """
    Creates a random NxN Hermitian Qobj.
    
    Args:
    
        N (int): Dimension of matrix
        
        density (float): Sparsity of output Hermitian matrix.
    
    Returns:
    
        NxN Hermitian Qobj
    """
    # to get appropriate density of output
    # Hermitian operator must convert via:
    herm_density=2.0*arcsin(density)/pi
    
    X = sp.rand(N,N,herm_density,format='csr')
    X.data=X.data-0.5
    Y=X.copy()
    Y.data=1.0j*random(len(X.data))-(0.5+0.5j)
    X=X+Y
    X=Qobj(X)
    return Qobj((X+X.dag())/2.0)


def rand_unitary(N,density=0.75):
    """
    Creates a random NxN unitary Qobj.
    
    Args:
    
        N (int): Dimension of matrix
        
        density (float): Sparsity of Hermitian operator used to construct Unitary Operator.
    
    Returns:
    
        NxN unitary Qobj
    """
    U=(-1.0j*rand_herm(N,density)).expm()
    return Qobj(U)


def rand_ket(N,density=0.75):
    """
    Creates a random Nx1 ket vector Qobj.
    
    Args:
    
        N (int): Dimension of matrix
        
        density (float): Sparsity of output ket vector.
    
    Returns:
    
        Nx1 ket vector Qobj
    """
    X = sp.rand(N,1,density,format='csr')
    X.data=X.data-0.5
    Y=X.copy()
    Y.data=1.0j*random(len(X.data))-(0.5+0.5j)
    X=X+Y
    X=Qobj(X)
    return Qobj(X/X.norm())


def rand_dm(N,density=0.75,pure=False):
    """
    Creates a random NxN density matrix Qobj.
    
    Args:
    
        N (int): Dimension of matrix
        
        density (float): Sparsity of output density matrix.
    
    Returns:
    
        NxN density matrix Qobj
    """
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
        rho = H/sum(H.tr())
    return Qobj(rho)


