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
from qutip.states import basis
from qutip.tensor import tensor


def quantum_register(N,state=None):
    """
    Creates a quantum register of N qubits.
    
    Parameters
    ----------
    N : int
        Number of qubits in register.
    
    Returns
    -------
    reg : qobj
        Quantum register for N qubits.
    
    """
    if state==None:
        return tensor([basis(2) for k in range(N)])

