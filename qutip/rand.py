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
from tidyup import tidyup
import scipy.linalg as la
import numpy as np
from numpy.random import random

def rand_herm(N):
    """
    Creates a random NxN Hermitian Qobj.
    
    Args:
    
        N (int): Dimension of matrix
    
    Returns:
    
        NxN Hermitian Qobj
    """
    X = 2.0*(random((N,N)) + 1.0j*random((N,N)))-(1.0+1.0j) #makes random from [-1,1]
    X=Qobj(X)
    return X+X.dag() 


def rand_unitary(N):
    """
    Creates a random NxN unitary Qobj.
    
    Args:
    
        N (int): Dimension of matrix
    
    Returns:
    
        NxN unitary Qobj
    """
    vec=(random((N,N)) + 1.0j*random((N,N)))/sqrt(2.0)
    Q,R = la.qr(vec)
    diags=np.diag(R)
    R = np.diag(diags/abs(diags))
    U = np.dot(Q,R)
    return Qobj(U)


def rand_ket(N):
    """
    Creates a random Nx1 ket vector Qobj.
    
    Args:
    
        N (int): Dimension of matrix
    
    Returns:
    
        Nx1 ket vector Qobj
    """
    U=rand_unitary(N)
    psi=Qobj(ones((N,1)))
    return U*psi


def rand_dm(N):
    """
    Creates a random NxN density matrix Qobj.
    
    Args:
    
        N (int): Dimension of matrix
    
    Returns:
    
        NxN density matrix Qobj
    """
    H = rand_herm(N)
    rho = H*H.dag()
    return rho/sum(rho.tr())


