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
"""
This module provides the circuit implementation for Quantum Fourier Transform.
"""

__all__ = ['qft', 'qft_steps', 'qft_gate_sequence']

import numpy as np
import scipy.sparse as sp
from qutip.qobj import *
from qutip.qip.gates import snot, cphase, swap
from qutip.qip.circuit import QubitCircuit


def qft(N=1):
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
    if N < 1:
        raise ValueError("Minimum value of N can be 1")

    N2 = 2 ** N
    phase = 2.0j * np.pi / N2
    arr = np.arange(N2)
    L, M = np.meshgrid(arr, arr)
    L = phase * (L * M)
    L = np.exp(L)
    dims = [[2] * N, [2] * N]
    return Qobj(1.0 / np.sqrt(N2) * L, dims=dims)


def qft_steps(N=1, swapping=True):
    """
    Quantum Fourier Transform operator on N qubits returning the individual
    steps as unitary matrices operating from left to right.

    Parameters
    ----------
    N: int
        Number of qubits.
    swap: boolean
        Flag indicating sequence of swap gates to be applied at the end or not.

    Returns
    -------
    U_step_list: list of qobj
        List of Hadamard and controlled rotation gates implementing QFT.

    """
    if N < 1:
        raise ValueError("Minimum value of N can be 1")

    U_step_list = []
    if N == 1:
        U_step_list.append(snot())
    else:
        for i in range(N):
            for j in range(i):
                U_step_list.append(cphase(np.pi / (2 ** (i - j)), N,
                                          control=i, target=j))
            U_step_list.append(snot(N, i))
        if swapping:
            for i in range(N // 2):
                U_step_list.append(swap(N, [N - i - 1, i]))

    return U_step_list


def qft_gate_sequence(N=1, swapping=True):
    """
    Quantum Fourier Transform operator on N qubits returning the gate sequence.

    Parameters
    ----------
    N: int
        Number of qubits.
    swap: boolean
        Flag indicating sequence of swap gates to be applied at the end or not.

    Returns
    -------
    qc: instance of QubitCircuit
        Gate sequence of Hadamard and controlled rotation gates implementing
        QFT.
    """

    if N < 1:
        raise ValueError("Minimum value of N can be 1")

    qc = QubitCircuit(N)
    if N == 1:
        qc.add_gate("SNOT", targets=[0])
    else:
        for i in range(N):
            for j in range(i):
                qc.add_gate("CPHASE", targets=[j], controls=[i],
                            arg_label=r"{\pi/2^{%d}}" % (i - j),
                            arg_value=np.pi / (2 ** (i - j)))
            qc.add_gate("SNOT", targets=[i])
        if swapping:
            for i in range(N // 2):
                qc.add_gate("SWAP", targets=[N - i - 1, i])

    return qc
