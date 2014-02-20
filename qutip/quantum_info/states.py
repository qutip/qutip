# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without 
#    modification, are permitted provided that the following conditions are 
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
import numpy as np
from qutip.states import basis
from qutip.tensor import tensor


def bell_state00():
    """
    Returns the B00 Bell state:
    
        |B00>=1/sqrt(2)*[|0>|0>+|1>|1>]
    
    Returns
    -------
    B00 : qobj
        B00 Bell state
    
    """
    B=tensor(basis(2),basis(2))+tensor(basis(2,1),basis(2,1))
    return B.unit()


def bell_state01():
    """
    Returns the B01 Bell state:
    
        |B01>=1/sqrt(2)*[|0>|0>-|1>|1>]
    
    Returns
    -------
    B01 : qobj
        B01 Bell state
    
    """
    B=tensor(basis(2),basis(2))-tensor(basis(2,1),basis(2,1))
    return B.unit()

    
def bell_state10():
    """
    Returns the B10 Bell state:
    
        |B10>=1/sqrt(2)*[|0>|1>+|1>|0>]
    
    Returns
    -------
    B10 : qobj
        B10 Bell state
    
    """
    B=tensor(basis(2),basis(2,1))+tensor(basis(2,1),basis(2))
    return B.unit()


def bell_state11():
    """
    Returns the B11 Bell state:
    
        |B11>=1/sqrt(2)*[|0>|1>-|1>|0>]
    
    Returns
    -------
    B11 : qobj
        B11 Bell state
    
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
