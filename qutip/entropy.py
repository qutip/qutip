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

import scipy.linalg as la
from numpy import e, real, imag, sort, sqrt
from scipy import log, log2
from qutip.Qobj import Qobj, ptrace
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
    
    Parameters
    ----------
    rho : qobj 
        Density matrix for two-qubits.
    
    Returns
    -------
    concur : float
        Concurrence
    
    """
    if rho.dims!=[[2, 2], [2, 2]]:
        raise Exception("Density matrix must be tensor product of two qubits.")
    
    sysy = tensor(sigmay(), sigmay())

    rho_tilde = (rho * sysy) * (rho.conj() * sysy)

    evals = rho_tilde.eigenenergies()

    evals = abs(sort(real(evals))) # abs to avoid problems with sqrt for very small negative numbers

    lsum = sqrt(evals[3]) - sqrt(evals[2]) - sqrt(evals[1]) - sqrt(evals[0])

    return max(0, lsum)


def entropy_mutual(rho,selA,selB,base=e,sparse=False):
    """
    Calculates the mutual information S(A:B) between selection components of a system density matrix.
    
    Parameters
    ----------
    rho : qobj
        Density matrix for composite quantum systems
    selA : int/list
        `int` or `list` of first selected density matrix components.
    selB : int/list
        `int` or `list` of second selected density matrix components.
    base : {e,2} 
        Base of logarithm.
    
    Other Parameters
    ----------------
    sparse : {False,True}
        Use sparse eigensolver.
    
    Returns
    -------
    ent_mut : float 
       Mutual information between selected components.
    
    """
    if isinstance(selA,int):
        selA=[selA]
    if isinstance(selB,int):
        selB=[selB]
    if rho.type!='oper':
        raise TypeError("Input must be a density matrix.")
    if (len(selA)+len(selB))!=len(rho.dims[0]):
        raise TypeError("Number of selected components must match total number.")
    
    rhoA=ptrace(rho,selA)
    rhoB=ptrace(rho,selB)
    out=entropy_vn(rhoA,base,sparse=sparse)+entropy_vn(rhoB,base,sparse=sparse)-entropy_vn(rho,base,sparse=sparse)
    return out


def _entropy_relative(rho,sigma,base=e,sparse=False):
    """
    ****NEEDS TO BE WORKED ON**** (after 2.0 release)
    
    Calculates the relative entropy S(rho||sigma) between two density matricies.
    
    Parameters
    ----------
    rho : qobj 
        First density matrix.
    sigma : qobj 
        Second density matrix.
    base : {e,2} 
        Base of logarithm.
    
    Returns
    -------
    rel_ent : float
        Value of relative entropy.
    
    """ 
    if rho.type!='oper' or sigma.type!='oper':
        raise TypeError("Inputs must be density matricies.")
    #sigma terms
    svals=sp_eigs(sigma,vecs=False,sparse=sparse)
    snzvals=svals[svals!=0]
    if base==2:
        slogvals=log2(snzvals)
    elif base==e:
        slogvals=log(snzvals)
    else:
        raise ValueError("Base must be 2 or e.")
    #rho terms
    rvals=sp_eigs(rho,vecs=False,sparse=sparse)
    rnzvals=rvals[rvals!=0]
    #calculate tr(rho*log sigma)
    rel_trace=float(real(sum(rnzvals*slogvals)))
    return -entropy_vn(rho,base,sparse)-rel_trace


def entropy_conditional(rho,selB,base=e,sparse=False):
    """
    Calculates the conditional entropy :math:`S(A|B)=S(A,B)-S(B)` 
    of a slected density matrix component.

    Parameters
    ----------
    rho : qobj 
        Density matrix of composite object
    selB : int/list 
        Selected components for density matrix B
    base : {e,2} 
        Base of logarithm.

    Other Parameters
    ----------------
    sparse : {False,True}
        Use sparse eigensolver.
    
    Returns
    -------
    ent_cond : float
        Value of conditional entropy
    
    """
    if rho.type!='oper':
        raise TypeError("Input must be density matrix.")
    if isinstance(selB,int):
        selB=[selB]
    B=ptrace(rho,selB)
    out=entropy_vn(rho,base,sparse=sparse)-entropy_vn(B,base,sparse=sparse)
    return out

