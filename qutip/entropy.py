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
from Qobj import *
import scipy.linalg as la
from states import ket2dm

def entropy_vn(rho,base='2'):
    """
    Von-Neumann entropy of density matrix
    
    Parameter rho *Qobj* density matrix
    
    Returns *float* entropy
    
    Example::
    
        >>> rho=0.5*fock_dm(2,0)+0.5*fock_dm(2,1)
        >>> entropy_vn(rho)
        1.0
    """
    vals,vecs=la.eigh(rho.full())
    nzvals=vals[vals!=0]
    if base=='2':
        logvals=log2(nzvals)
    elif base=='e':
        logvals=log(nzvals)
    else:
        raise ValueError("Base must be '2' or 'e'.")
    return float(real(-sum(nzvals*logvals)))


def entropy_linear(rho):
    """
    Linear entropy of density matrix
    
    Parameter rho *Qobj* density matrix or ket/bra
    
    Returns *float*
    
    Example:: 
        
        >>> rho=0.5*fock_dm(2,0)+0.5*fock_dm(2,1)
        >>> entropy_linear(rho)
        0.5
    """
    if rho.type=='ket' or rho.type=='bra':
        rho=ket2dm(rho)
    return float(real(1.0-(rho**2).tr()))



def concurrence(rho):
    """
    Calculate the concurrence entanglement measure for 
        a two-qubit state.
    
    Parameter rho *Qobj* density matrix
    
    Returns *float* concurrence
    """
    sysy = tensor(sigmay(), sigmay())

    rho_tilde = (rho * sysy) * (rho.conj() * sysy)

    evals = rho_tilde.eigenenergies()

    evals = abs(sort(real(evals))) # abs to avoid problems with sqrt for very small negative numbers

    lsum = sqrt(evals[3]) - sqrt(evals[2]) - sqrt(evals[1]) - sqrt(evals[0])

    return max(0, lsum)

