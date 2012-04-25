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
from scipy import *
from qutip.Qobj import *
import scipy.linalg as la
from qutip.states import ket2dm
from qutip.tensor import tensor
from qutip.operators import sigmay
from qutip.sparse import sp_eigs

def entropy_vn(rho,base=e,sparse=False):
    """
    Von-Neumann entropy of density matrix
    
    Parameters
    ----------
    rho : qobj
        Density matrix.
    base : {e,2} 
        Base of logarithm.
    
    Other Parameters
    ----------------
    sparse : {False,True}
        Use sparse eigensolver.
    
    Returns
    ------- 
    entropy : float
        Von-Neumann entropy of `rho`.
    
    Examples
    --------
    >>> rho=0.5*fock_dm(2,0)+0.5*fock_dm(2,1)
    >>> entropy_vn(rho,2)
    1.0
    
    """
    if rho.type=='ket' or rho.type=='bra':
        rho=ket2dm(rho)
    vals=sp_eigs(rho,vecs=False,sparse=sparse)
    nzvals=vals[vals!=0]
    nzvals=nzvals[nzvals>=1e-15]
    if base==2:
        logvals=log2(nzvals)
    elif base==e:
        logvals=log(nzvals)
    else:
        raise ValueError("Base must be 2 or e.")
    return float(real(-sum(nzvals*logvals)))


def entropy_linear(rho):
    """
    Linear entropy of a density matrix.
    
    Parameters
    ----------
    rho : qobj
        sensity matrix or ket/bra vector.
    
    Returns
    -------
    entropy : float
        Linear entropy of rho.
    
    Examples
    -------- 
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
    
    Args:
        
        rho (Qobj): density matrix.
    
    Returns:
        
        float for concurrence
    """
    sysy = tensor(sigmay(), sigmay())

    rho_tilde = (rho * sysy) * (rho.conj() * sysy)

    evals = rho_tilde.eigenenergies()

    evals = abs(sort(real(evals))) # abs to avoid problems with sqrt for very small negative numbers

    lsum = sqrt(evals[3]) - sqrt(evals[2]) - sqrt(evals[1]) - sqrt(evals[0])

    return max(0, lsum)


def entropy_mutual(rho,base=e):
    """
    Calculates the mutual information S(A:B) of a bipartite system density matrix.
    
    Args:
    
        rho (Qobj): density matrix.
        
        base (float): base of logarithm, e (default) or 2
    
    Returns:
    
        float value of mutual information
    """
    if rho.type!='oper':
        raise TypeError("Input must be a density matrix.")
    if len(rho.dims[0])!=2:
        raise TypeError("Input must be bipartite system.")
    
    rhoA=ptrace(rho,0)
    rhoB=ptrace(rho,1)
    out=entropy_vn(rhoA,base)+entropy_vn(rhoB,base)-entropy_vn(rho,base)
    return out


def entropy_relative(rho,sigma,base=e):
    """
    Calculates the relative entropy S(rho||sigma) between two density matricies.
    Relative entropy of rho to sigma
    
    Args:
    
        rho (Qobj): density matrix.
        
        sigma (Qobj): density matrix.
        
        base (float): base of logarithm, e (default) or 2
    
    Returns:
    
        float value of relative entropy
    """ 
    if rho.type!='oper' or sigma.type!='oper':
        raise TypeError("Inputs must be density matricies.")
    vals,vecs=la.eigh(rho.full())
    nzvals=vals[vals!=0]
    if base==2:
        logvals=log2(nzvals)
    elif base==e:
        logvals=log(nzvals)
    else:
        raise ValueError("Base must be 2 or e.")
    vals2,vecs2=la.eigh(rho.full())
    nzvals2=vals2[vals2!=0]
    rel_ent=float(real(-sum(nzvals2*logvals)))
    return -1.0*entropy_vn(rho,base)-rel_ent


def entropy_conditional(rho,sel,base=e):
    """
    Calculates the conditional entropy S(A|B) of a bipartite system density matrix.

    Args:

        rho (Qobj): density matrix.

        sel (int): Which component is "A" component (0 or 1) 

        base (float): base of logarithm, e (default) or 2

    Returns:

        float value of conditional entropy
    """
    if rho.type!='oper':
        raise TypeError("Input must be density matrix.")
    if len(rho.dims[0])!=2:
        raise TypeError("Input must be bipartite system.")
    if sel!=0 or sel!=1:
        raise ValueError("Choice of density matrix 'A' component must be 0 or 1.")
    if sel==0:
        rhoB=ptrace(rho,1)
    else:
        rhoB=ptrace(rho,0)
    out=entropy_vn(rho,base)-entropy_vn(rhoB,base)
    return out

