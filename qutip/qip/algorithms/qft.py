"""
This module provides the circuit implementation for Quantum Fourier Transform.
"""


import numpy as np
from qutip.qip.operations.gates import snot, cphase, swap
from qutip.qip.circuit import QubitCircuit
from qutip.qobj import Qobj


__all__ = ['qft', 'qft_steps', 'qft_gate_sequence']


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
