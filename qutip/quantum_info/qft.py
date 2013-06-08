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
import scipy.sparse as sp
from qutip.qobj import *

def qft(N):
    """
    Quantum Fourier Transform operator on N qubits.
    
    Parameters
    ----------
    N : int
        Number of qubits.
    
    Returns
    -------
    QFT: qobj
        Quantum Fourier transform operator.
    
    """
    N2=2**N
    phase=2.0j*np.pi/N2
    arr=np.arange(N2)
    L, M = np.meshgrid(arr, arr)
    L=phase*(L*M)
    L=np.exp(L)
    dims=[[2]*N,[2]*N]
    return Qobj(1.0/np.sqrt(N2)*L,dims=dims)
