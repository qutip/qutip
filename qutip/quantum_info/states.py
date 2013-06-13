# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import numpy as np
from qutip.states import basis
from qutip.tensor import tensor


def bell_state1():
    """
    Returns the Bell state:
    
        |B1>=1/sqrt(2)*[|0>|0>+|1>|1>]
    
    Returns
    -------
    B1 : qobj
        B1 Bell state
    
    """
    B=tensor(basis(2),basis(2))+tensor(basis(2,1),basis(2,1))
    return B.unit()


def bell_state2():
    """
    Returns the Bell state:
    
        |B1>=1/sqrt(2)*[|0>|0>-|1>|1>]
    
    Returns
    -------
    B2 : qobj
        B2 Bell state
    
    """
    B=tensor(basis(2),basis(2))-tensor(basis(2,1),basis(2,1))
    return B.unit()

    
def bell_state3():
    """
    Returns the Bell state:
    
        |B3>=1/sqrt(2)*[|0>|1>+|1>|0>]
    
    Returns
    -------
    B4 : qobj
        B4 Bell state
    
    """
    B=tensor(basis(2),basis(2,1))+tensor(basis(2,1),basis(2))
    return B.unit()


def bell_state4():
    """
    Returns the Bell state:
    
        |B4>=1/sqrt(2)*[|0>|1>-|1>|0>]
    
    Returns
    -------
    B4 : qobj
        B4 Bell state
    
    """
    B=tensor(basis(2),basis(2,1))-tensor(basis(2,1),basis(2))
    return B.unit()


def singlet_state():
    """
    Returns the two particle singlet-state:
    
        |S>=1/sqrt(2)*[|0>|1>-|1>|0>]
    
    that is identical to the fourth bell state.
    
    Returns
    -------
    B4 : qobj
        B4 Bell state
    
    """
    return bell_state4()


def w_state(N=3):
    """
    Returns the N-qubit W-state.
    
    Parameters
    ----------
    N : int (default=3)
        Number of qubits in state
    
    Returns
    -------
    W : qobj
        N-qubit W-state
    
    """
    inds=np.zeros(N,dtype=int)
    inds[0]=1
    state=tensor([basis(2,x) for x in inds])
    for kk in range(1,N):
        perm_inds=np.roll(inds,kk)
        state+=tensor([basis(2,x) for x in perm_inds])
    return state.unit()


def ghz_state(N=3):
    """
    Returns the N-qubit GHZ-state.
    
    Parameters
    ----------
    N : int (default=3)
        Number of qubits in state
    
    Returns
    -------
    G : qobj
        N-qubit GHZ-state
    
    """
    state=tensor([basis(2) for k in range(N)])+tensor([basis(2,1) for k in range(N)])
    return state/np.sqrt(2)
