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
Collection of metrics (distance measures) between density matricies
"""

from qutip.Qobj import *
import scipy.linalg as la
from scipy import real
from qutip.sparse import sp_eigs


def fidelity(A,B):
    """
    Calculates the fidelity (pseudo-metric) between two density matricies.
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"
    
    Args:
    
        A (Qobj): Density matrix.
        
        B (Qobj): Density matrix with same dimensions as A.
    
    Returns:
        float for fidelity.
    
    Example::
    
        >>> x=fock_dm(5,3)
        >>> y=coherent_dm(5,1)
        >>> fidelity(x,y)
        0.24104350624628332
        
    """
    if A.dims!=B.dims:
        raise TypeError('Density matricies do not have same dimensions.')
    else:
        A=A.sqrtm()
        return float(real((A*(B*A)).sqrtm().tr()))


def tracedist(A,B,tol=0):
    """
    Calculates the trace distance between two density matricies.
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"
    
    Args:
    
        A (Qobj): Density matrix.
        
        B (Qobj): Sensity matrix with same dimensions as A.
        
        tol (float): Tolerance used by sparse eigensolver. (0=Machine precision)
    
    Returns:
        float for trace distance
    
    Example::
    
        >>> x=fock_dm(5,3)
        >>> y=coherent_dm(5,1)
        >>> tracedist(x,y)
        0.9705143161472971
    
    """
    if A.dims!=B.dims:
        raise TypeError('Density matricies do not have same dimensions.')
    else:
        diff=A-B
        diff=diff.dag()*diff
        vals=sp_eigs(diff,vecs=False,tol=0)
        return float(real(0.5*sum(sqrt(vals))))
