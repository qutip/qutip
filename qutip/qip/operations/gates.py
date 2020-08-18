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
import numbers
from collections.abc import Iterable
from itertools import product
from functools import partial, reduce
from operator import mul

import numpy as np
import scipy.sparse as sp
from qutip.qobj import Qobj
from qutip.operators import identity, qeye, sigmax, sigmay, sigmaz
from qutip.tensor import tensor
from qutip.states import fock_dm


__all__ = ['rx', 'ry', 'rz', 'sqrtnot', 'snot', 'phasegate', 'qrot',
           'x_gate', 'y_gate', 'z_gate', 'cy_gate', 'cz_gate', 's_gate',
           't_gate', 'qasmu_gate', 'cs_gate', 'ct_gate', 'cphase', 'cnot',
           'csign', 'berkeley', 'swapalpha', 'swap', 'iswap', 'sqrtswap',
           'sqrtiswap', 'fredkin', 'molmer_sorensen',
           'toffoli', 'rotation', 'controlled_gate',
           'globalphase', 'hadamard_transform', 'gate_sequence_product',
           'gate_expand_1toN', 'gate_expand_2toN', 'gate_expand_3toN',
           'qubit_clifford_group', 'expand_operator']

#
# Single Qubit Gates
#


def x_gate(N=None, target=0):
    """Pauli-X gate or sigmax operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the x-axis.

    """
    if N is not None:
        return gate_expand_1toN(x_gate(), N, target)
    return sigmax()


def y_gate(N=None, target=0):
    """Pauli-Y gate or sigmay operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the y-axis.

    """
    if N is not None:
        return gate_expand_1toN(y_gate(), N, target)
    return sigmay()


def cy_gate(N=None, control=0, target=1):
    """Controlled Y gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cy_gate(), N, control, target)
    return Qobj([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, -1j],
                 [0, 0, 1j, 0]],
                dims=[[2, 2], [2, 2]])


def z_gate(N=None, target=0):
    """Pauli-Z gate or sigmaz operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the z-axis.

    """
    if N is not None:
        return gate_expand_1toN(z_gate(), N, target)
    return sigmaz()


def cz_gate(N=None, control=0, target=1):
    """Controlled Z gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cz_gate(), N, control, target)
    return Qobj([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, -1j]],
                dims=[[2, 2], [2, 2]])


def s_gate(N=None, target=0):
    """Single-qubit rotation also called Phase gate or the Z90 gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a 90 degree rotation around the z-axis.

    """
    if N is not None:
        return gate_expand_1toN(s_gate(), N, target)
    return Qobj([[1, 0],
                 [0, 1j]])


def cs_gate(N=None, control=0, target=1):
    """Controlled S gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cs_gate(), N, control, target)
    return Qobj([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1j]],
                dims=[[2, 2], [2, 2]])


def t_gate(N=None, target=0):
    """Single-qubit rotation related to the S gate by the relationship S=T*T.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing a phase shift of pi/4.

    """
    if N is not None:
        return gate_expand_1toN(t_gate(), N, target)
    return Qobj([[1, 0],
                 [0, np.exp(1j*np.pi/4)]])


def ct_gate(N=None, control=0, target=1):
    """Controlled T gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(ct_gate(), N, control, target)
    return Qobj([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, np.exp(1j * np.pi / 4)]],
                dims=[[2, 2], [2, 2]])


def rx(phi, N=None, target=0):
    """Single-qubit rotation for operator sigmax with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if N is not None:
        return gate_expand_1toN(rx(phi), N, target)
    return Qobj([[np.cos(phi / 2), -1j * np.sin(phi / 2)],
                 [-1j * np.sin(phi / 2), np.cos(phi / 2)]])


def ry(phi, N=None, target=0):
    """Single-qubit rotation for operator sigmay with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if N is not None:
        return gate_expand_1toN(ry(phi), N, target)
    return Qobj([[np.cos(phi / 2), -np.sin(phi / 2)],
                 [np.sin(phi / 2), np.cos(phi / 2)]])


def rz(phi, N=None, target=0):
    """Single-qubit rotation for operator sigmaz with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if N is not None:
        return gate_expand_1toN(rz(phi), N, target)
    return Qobj([[np.exp(-1j * phi / 2), 0],
                 [0, np.exp(1j * phi / 2)]])


def sqrtnot(N=None, target=0):
    """Single-qubit square root NOT gate.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the square root NOT gate.

    """
    if N is not None:
        return gate_expand_1toN(sqrtnot(), N, target)
    return Qobj([[0.5 + 0.5j, 0.5 - 0.5j],
                 [0.5 - 0.5j, 0.5 + 0.5j]])


def snot(N=None, target=0):
    """Quantum object representing the SNOT (Hadamard) gate.

    Returns
    -------
    snot_gate : qobj
        Quantum object representation of SNOT gate.

    Examples
    --------
    >>> snot() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 0.70710678+0.j  0.70710678+0.j]
     [ 0.70710678+0.j -0.70710678+0.j]]

    """
    if N is not None:
        return gate_expand_1toN(snot(), N, target)
    return 1 / np.sqrt(2.0) * Qobj([[1, 1],
                                    [1, -1]])


def phasegate(theta, N=None, target=0):
    """
    Returns quantum object representing the phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    Returns
    -------
    phase_gate : qobj
        Quantum object representation of phase shift gate.

    Examples
    --------
    >>> phasegate(pi/4) # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 1.00000000+0.j          0.00000000+0.j        ]
     [ 0.00000000+0.j          0.70710678+0.70710678j]]

    """
    if N is not None:
        return gate_expand_1toN(phasegate(theta), N, target)
    return Qobj([[1, 0],
                 [0, np.exp(1.0j * theta)]],
                dims=[[2], [2]])


def qrot(theta, phi, N=None, target=0):
    """
    Single qubit rotation driving by Rabi oscillation with 0 detune.

    Parameters
    ----------
    phi : float
        The inital phase of the rabi pulse.
    theta : float
        The duration of the rabi pulse.
    N : int
        Number of qubits in the system.
    target : int
        The index of the target qubit.

    Returns
    -------
    qrot_gate : :class:`qutip.Qobj`
        Quantum object representation of physical qubit rotation under
        a rabi pulse.
    """
    if N is not None:
        return expand_operator(qrot(theta, phi), N=N, targets=target)
    return Qobj(
        [
            [np.cos(theta/2.), -1.j*np.exp(-1.j*phi)*np.sin(theta/2.)],
            [-1.j*np.exp(1.j*phi)*np.sin(theta/2.), np.cos(theta/2.)]
        ])


def qasmu_gate(args, N=None, target=0):
    """
    QASM U-gate as defined in the OpenQASM standard.

    Parameters
    ----------

    theta : float
        The argument supplied to the last RZ rotation.
    phi : float
        The argument supplied to the middle RY rotation.
    gamma : float
        The argument supplied to the first RZ rotation.
    N : int
        Number of qubits in the system.
    target : int
        The index of the target qubit.

    Returns
    -------
    qasmu_gate : :class:`qutip.Qobj`
        Quantum object representation of the QASM U-gate as defined in the
        OpenQASM standard.
    """

    theta, phi, gamma = args
    if N is not None:
        return expand_operator(qasmu_gate([theta, phi, gamma]), N=N,
                               targets=target)
    return Qobj(rz(phi) * ry(theta) * rz(gamma))


#
# 2 Qubit Gates
#

def cphase(theta, N=2, control=0, target=1):
    """
    Returns quantum object representing the controlled phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    N : integer
        The number of qubits in the target space.

    control : integer
        The index of the control qubit.

    target : integer
        The index of the target qubit.

    Returns
    -------
    U : qobj
        Quantum object representation of controlled phase gate.
    """

    if N < 1 or target < 0 or control < 0:
        raise ValueError("Minimum value: N=1, control=0 and target=0")

    if control >= N or target >= N:
        raise ValueError("control and target need to be smaller than N")

    U_list1 = [identity(2)] * N
    U_list2 = [identity(2)] * N

    U_list1[control] = fock_dm(2, 1)
    U_list1[target] = phasegate(theta)

    U_list2[control] = fock_dm(2, 0)

    U = tensor(U_list1) + tensor(U_list2)
    return U


def cnot(N=None, control=0, target=1):
    """
    Quantum object representing the CNOT gate.

    Returns
    -------
    cnot_gate : qobj
        Quantum object representation of CNOT gate

    Examples
    --------
    >>> cnot() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]]

    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cnot(), N, control, target)
    return Qobj([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]],
                dims=[[2, 2], [2, 2]])


def csign(N=None, control=0, target=1):
    """
    Quantum object representing the CSIGN gate.

    Returns
    -------
    csign_gate : qobj
        Quantum object representation of CSIGN gate

    Examples
    --------
    >>> csign() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  -1.+0.j]]

    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(csign(), N, control, target)
    return Qobj([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, -1]],
                dims=[[2, 2], [2, 2]])


def berkeley(N=None, targets=[0, 1]):
    """
    Quantum object representing the Berkeley gate.

    Returns
    -------
    berkeley_gate : qobj
        Quantum object representation of Berkeley gate

    Examples
    --------
    >>> berkeley() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
        [[ cos(pi/8).+0.j  0.+0.j           0.+0.j           0.+sin(pi/8).j]
         [ 0.+0.j          cos(3pi/8).+0.j  0.+sin(3pi/8).j  0.+0.j]
         [ 0.+0.j          0.+sin(3pi/8).j  cos(3pi/8).+0.j  0.+0.j]
         [ 0.+sin(pi/8).j  0.+0.j           0.+0.j           cos(pi/8).+0.j]]

    """
    if (targets[0] == 1 and targets[1] == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cnot(), N, targets=targets)
    return Qobj([[np.cos(np.pi / 8), 0, 0, 1.0j * np.sin(np.pi / 8)],
                 [0, np.cos(3 * np.pi / 8), 1.0j *
                  np.sin(3 * np.pi / 8), 0],
                 [0, 1.0j * np.sin(3 * np.pi / 8),
                  np.cos(3 * np.pi / 8), 0],
                 [1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)]],
                dims=[[2, 2], [2, 2]])


def swapalpha(alpha, N=None, targets=[0, 1]):
    """
    Quantum object representing the SWAPalpha gate.

    Returns
    -------
    swapalpha_gate : qobj
        Quantum object representation of SWAPalpha gate

    Examples
    --------
    >>> swapalpha(alpha) # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[ 1.+0.j  0.+0.j                    0.+0.j                    0.+0.j]
     [ 0.+0.j  0.5*(1 + exp(j*pi*alpha)  0.5*(1 - exp(j*pi*alpha)  0.+0.j]
     [ 0.+0.j  0.5*(1 - exp(j*pi*alpha)  0.5*(1 + exp(j*pi*alpha)  0.+0.j]
     [ 0.+0.j  0.+0.j                    0.+0.j                    1.+0.j]]

    """
    if (targets[0] == 1 and targets[1] == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cnot(), N, targets=targets)
    return Qobj([[1, 0, 0, 0],
                 [0, 0.5 * (1 + np.exp(1.0j * np.pi * alpha)),
                  0.5 * (1 - np.exp(1.0j * np.pi * alpha)), 0],
                 [0, 0.5 * (1 - np.exp(1.0j * np.pi * alpha)),
                  0.5 * (1 + np.exp(1.0j * np.pi * alpha)), 0],
                 [0, 0, 0, 1]],
                dims=[[2, 2], [2, 2]])


def swap(N=None, targets=[0, 1]):
    """Quantum object representing the SWAP gate.

    Returns
    -------
    swap_gate : qobj
        Quantum object representation of SWAP gate

    Examples
    --------
    >>> swap() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]

    """
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(swap(), N, targets=targets)
    return Qobj([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]],
                dims=[[2, 2], [2, 2]])


def iswap(N=None, targets=[0, 1]):
    """Quantum object representing the iSWAP gate.

    Returns
    -------
    iswap_gate : qobj
        Quantum object representation of iSWAP gate

    Examples
    --------
    >>> iswap() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]
     [ 0.+0.j  0.+1.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]
    """
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(iswap(), N, targets=targets)
    return Qobj([[1, 0, 0, 0],
                 [0, 0, 1j, 0],
                 [0, 1j, 0, 0],
                 [0, 0, 0, 1]],
                dims=[[2, 2], [2, 2]])


def sqrtswap(N=None, targets=[0, 1]):
    """Quantum object representing the square root SWAP gate.

    Returns
    -------
    sqrtswap_gate : qobj
        Quantum object representation of square root SWAP gate

    """
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(sqrtswap(), N, targets=targets)
    return Qobj(np.array([[1, 0, 0, 0],
                          [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                          [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                          [0, 0, 0, 1]]),
                dims=[[2, 2], [2, 2]])


def sqrtiswap(N=None, targets=[0, 1]):
    """Quantum object representing the square root iSWAP gate.

    Returns
    -------
    sqrtiswap_gate : qobj
        Quantum object representation of square root iSWAP gate

    Examples
    --------
    >>> sqrtiswap() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 1.00000000+0.j   0.00000000+0.j   \
       0.00000000+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.70710678+0.j   \
       0.00000000-0.70710678j  0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000-0.70710678j\
       0.70710678+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000+0.j   \
       0.00000000+0.j          1.00000000+0.j]]

    """
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(sqrtiswap(), N, targets=targets)
    return Qobj(np.array([[1, 0, 0, 0],
                          [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                          [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                          [0, 0, 0, 1]]), dims=[[2, 2], [2, 2]])


def molmer_sorensen(theta, N=None, targets=[0, 1]):
    """
    Quantum object of a Mølmer–Sørensen gate.

    Parameters
    ----------
    theta: float
        The duration of the interaction pulse.
    N: int
        Number of qubits in the system.
    target: int
        The indices of the target qubits.

    Returns
    -------
    molmer_sorensen_gate: :class:`qutip.Qobj`
        Quantum object representation of the Mølmer–Sørensen gate.
    """
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        return expand_operator(molmer_sorensen(theta), N, targets=targets)
    return Qobj(
        [
            [np.cos(theta/2.), 0, 0, -1.j*np.sin(theta/2.)],
            [0, np.cos(theta/2.), -1.j*np.sin(theta/2.), 0],
            [0, -1.j*np.sin(theta/2.), np.cos(theta/2.), 0],
            [-1.j*np.sin(theta/2.), 0, 0, np.cos(theta/2.)]
        ],
        dims=[[2, 2], [2, 2]])


#
# 3 Qubit Gates
#

def fredkin(N=None, control=0, targets=[1, 2]):
    """Quantum object representing the Fredkin gate.

    Returns
    -------
    fredkin_gate : qobj
        Quantum object representation of Fredkin gate.

    Examples
    --------
    >>> fredkin() # doctest: +SKIP
    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], \
shape = [8, 8], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j]]

    """
    if [control, targets[0], targets[1]] != [0, 1, 2] and N is None:
        N = 3

    if N is not None:
        return gate_expand_3toN(fredkin(), N,
                                [control, targets[0]], targets[1])
    return Qobj([[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]],
                dims=[[2, 2, 2], [2, 2, 2]])


def toffoli(N=None, controls=[0, 1], target=2):
    """Quantum object representing the Toffoli gate.

    Returns
    -------
    toff_gate : qobj
        Quantum object representation of Toffoli gate.

    Examples
    --------
    >>> toffoli() # doctest: +SKIP
    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], \
                    shape = [8, 8], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]]


    """
    if [controls[0], controls[1], target] != [0, 1, 2] and N is None:
        N = 3

    if N is not None:
        return gate_expand_3toN(toffoli(), N, controls, target)
    return Qobj([[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0]],
                dims=[[2, 2, 2], [2, 2, 2]])


#
# Miscellaneous Gates
#

def rotation(op, phi, N=None, target=0):
    """Single-qubit rotation for operator op with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if N is not None:
        return gate_expand_1toN(rotation(op, phi), N, target)
    return (-1j * op * phi / 2).expm()


def controlled_gate(U, N=2, control=0, target=1, control_value=1):
    """
    Create an N-qubit controlled gate from a single-qubit gate U with the given
    control and target qubits.

    Parameters
    ----------
    U : Qobj
        Arbitrary single-qubit gate.

    N : integer
        The number of qubits in the target space.

    control : integer
        The index of the first control qubit.

    target : integer
        The index of the target qubit.

    control_value : integer (1)
        The state of the control qubit that activates the gate U.

    Returns
    -------
    result : qobj
        Quantum object representing the controlled-U gate.

    """

    if [N, control, target] == [2, 0, 1]:
        return (tensor(fock_dm(2, control_value), U) +
                tensor(fock_dm(2, 1 - control_value), identity(2)))
    U2 = controlled_gate(U, control_value=control_value)
    return gate_expand_2toN(U2, N=N, control=control, target=target)


def globalphase(theta, N=1):
    """
    Returns quantum object representing the global phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    Returns
    -------
    phase_gate : qobj
        Quantum object representation of global phase shift gate.

    Examples
    --------
    >>> phasegate(pi/4) # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.70710678+0.70710678j          0.00000000+0.j]
     [ 0.00000000+0.j          0.70710678+0.70710678j]]

    """
    data = (np.exp(1.0j * theta) * sp.eye(2 ** N, 2 ** N,
                                          dtype=complex, format="csr"))
    return Qobj(data, dims=[[2] * N, [2] * N])


#
# Operation on Gates
#

def _hamming_distance(x, bits=32):
    """
    Calculate the bit-wise Hamming distance of x from 0: That is, the number
    1s in the integer x.
    """
    tot = 0
    while x:
        tot += 1
        x &= x - 1
    return tot


def hadamard_transform(N=1):
    """Quantum object representing the N-qubit Hadamard gate.

    Returns
    -------
    q : qobj
        Quantum object representation of the N-qubit Hadamard gate.

    """
    data = 2 ** (-N / 2) * np.array([[(-1) ** _hamming_distance(i & j)
                                      for i in range(2 ** N)]
                                     for j in range(2 ** N)])

    return Qobj(data, dims=[[2] * N, [2] * N])


def gate_sequence_product(U_list, left_to_right=True):
    """
    Calculate the overall unitary matrix for a given list of unitary operations

    Parameters
    ----------
    U_list : list
        List of gates implementing the quantum circuit.

    left_to_right : Boolean
        Check if multiplication is to be done from left to right.

    Returns
    -------
    U_overall : qobj
        Overall unitary matrix of a given quantum circuit.

    """
    U_overall = 1
    for U in U_list:
        if left_to_right:
            U_overall = U * U_overall
        else:
            U_overall = U_overall * U

    return U_overall


def _powers(op, N):
    """
    Generator that yields powers of an operator `op`,
    through to `N`.
    """
    acc = qeye(op.dims[0])
    yield acc

    for _ in range(N - 1):
        acc *= op
        yield acc


def qubit_clifford_group(N=None, target=0):
    """
    Generates the Clifford group on a single qubit,
    using the presentation of the group given by Ross and Selinger
    (http://www.mathstat.dal.ca/~selinger/newsynth/).

    Parameters
    -----------

    N : int or None
        Number of qubits on which each operator is to be defined
        (default: 1).
    target : int
        Index of the target qubit on which the single-qubit
        Clifford operators are to act.

    Yields
    ------

    op : Qobj
        Clifford operators, represented as Qobj instances.

    """

    # The Ross-Selinger presentation of the single-qubit Clifford
    # group expresses each element in the form C_{ijk} = E^i X^j S^k
    # for gates E, X and S, and for i in range(3), j in range(2) and
    # k in range(4).
    #
    # We start by defining these gates. E is defined in terms of H,
    # \omega and S, so we define \omega and H first.
    w = np.exp(1j * 2 * np.pi / 8)
    H = snot()

    X = sigmax()
    S = phasegate(np.pi / 2)
    E = H * (S ** 3) * w ** 3

    for op in map(partial(reduce, mul), product(_powers(E, 3),
                                                _powers(X, 2),
                                                _powers(S, 4))):

        # partial(reduce, mul) returns a function that takes products
        # of its argument, by analogy to sum. Note that by analogy,
        # sum can be written as partial(reduce, add).

        # product(...) yields the Cartesian product of its arguments.
        # Here, each element is a tuple (E**i, X**j, S**k) such that
        # partial(reduce, mul) acting on the tuple yields E**i * X**j * S**k.

        # Finally, we optionally expand the gate.
        if N is not None:
            yield gate_expand_1toN(op, N, target)
        else:
            yield op

#
# Gate Expand
#


def gate_expand_1toN(U, N, target):
    """
    Create a Qobj representing a one-qubit gate that act on a system with N
    qubits.

    Parameters
    ----------
    U : Qobj
        The one-qubit gate

    N : integer
        The number of qubits in the target space.

    target : integer
        The index of the target qubit.

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.

    """

    if N < 1:
        raise ValueError("integer N must be larger or equal to 1")

    if target >= N:
        raise ValueError("target must be integer < integer N")

    return tensor([identity(2)] * (target) + [U] +
                  [identity(2)] * (N - target - 1))


def gate_expand_2toN(U, N, control=None, target=None, targets=None):
    """
    Create a Qobj representing a two-qubit gate that act on a system with N
    qubits.

    Parameters
    ----------
    U : Qobj
        The two-qubit gate

    N : integer
        The number of qubits in the target space.

    control : integer
        The index of the control qubit.

    target : integer
        The index of the target qubit.

    targets : list
        List of target qubits.

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.

    """

    if targets is not None:
        control, target = targets

    if control is None or target is None:
        raise ValueError("Specify value of control and target")

    if N < 2:
        raise ValueError("integer N must be larger or equal to 2")

    if control >= N or target >= N:
        raise ValueError("control and not target must be integer < integer N")

    if control == target:
        raise ValueError("target and not control cannot be equal")

    p = list(range(N))

    if target == 0 and control == 1:
        p[control], p[target] = p[target], p[control]

    elif target == 0:
        p[1], p[target] = p[target], p[1]
        p[1], p[control] = p[control], p[1]

    else:
        p[1], p[target] = p[target], p[1]
        p[0], p[control] = p[control], p[0]

    return tensor([U] + [identity(2)] * (N - 2)).permute(p)


def gate_expand_3toN(U, N, controls=[0, 1], target=2):
    """
    Create a Qobj representing a three-qubit gate that act on a system with N
    qubits.

    Parameters
    ----------
    U : Qobj
        The three-qubit gate

    N : integer
        The number of qubits in the target space.

    controls : list
        The list of the control qubits.

    target : integer
        The index of the target qubit.

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.

    """

    if N < 3:
        raise ValueError("integer N must be larger or equal to 3")

    if controls[0] >= N or controls[1] >= N or target >= N:
        raise ValueError("control and not target is None."
                         " Must be integer < integer N")

    if (controls[0] == target or
            controls[1] == target or
            controls[0] == controls[1]):

        raise ValueError("controls[0], controls[1], and target"
                         " cannot be equal")

    p = list(range(N))
    p1 = list(range(N))
    p2 = list(range(N))

    if controls[0] <= 2 and controls[1] <= 2 and target <= 2:
        p[controls[0]] = 0
        p[controls[1]] = 1
        p[target] = 2

    #
    # N > 3 cases
    #

    elif controls[0] == 0 and controls[1] == 1:
        p[2], p[target] = p[target], p[2]

    elif controls[0] == 0 and target == 2:
        p[1], p[controls[1]] = p[controls[1]], p[1]

    elif controls[1] == 1 and target == 2:
        p[0], p[controls[0]] = p[controls[0]], p[0]

    elif controls[0] == 1 and controls[1] == 0:
        p[controls[1]], p[controls[0]] = p[controls[0]], p[controls[1]]
        p2[2], p2[target] = p2[target], p2[2]
        p = [p2[p[k]] for k in range(N)]

    elif controls[0] == 2 and target == 0:
        p[target], p[controls[0]] = p[controls[0]], p[target]
        p1[1], p1[controls[1]] = p1[controls[1]], p1[1]
        p = [p1[p[k]] for k in range(N)]

    elif controls[1] == 2 and target == 1:
        p[target], p[controls[1]] = p[controls[1]], p[target]
        p1[0], p1[controls[0]] = p1[controls[0]], p1[0]
        p = [p1[p[k]] for k in range(N)]

    elif controls[0] == 1 and controls[1] == 2:
        #  controls[0] -> controls[1] -> target -> outside
        p[0], p[1] = p[1], p[0]
        p[0], p[2] = p[2], p[0]
        p[0], p[target] = p[target], p[0]

    elif controls[0] == 2 and target == 1:
        #  controls[0] -> target -> controls[1] -> outside
        p[0], p[2] = p[2], p[0]
        p[0], p[1] = p[1], p[0]
        p[0], p[controls[1]] = p[controls[1]], p[0]

    elif controls[1] == 0 and controls[0] == 2:
        #  controls[1] -> controls[0] -> target -> outside
        p[1], p[0] = p[0], p[1]
        p[1], p[2] = p[2], p[1]
        p[1], p[target] = p[target], p[1]

    elif controls[1] == 2 and target == 0:
        #  controls[1] -> target -> controls[0] -> outside
        p[1], p[2] = p[2], p[1]
        p[1], p[0] = p[0], p[1]
        p[1], p[controls[0]] = p[controls[0]], p[1]

    elif target == 1 and controls[1] == 0:
        #  target -> controls[1] -> controls[0] -> outside
        p[2], p[1] = p[1], p[2]
        p[2], p[0] = p[0], p[2]
        p[2], p[controls[0]] = p[controls[0]], p[2]

    elif target == 0 and controls[0] == 1:
        #  target -> controls[0] -> controls[1] -> outside
        p[2], p[0] = p[0], p[2]
        p[2], p[1] = p[1], p[2]
        p[2], p[controls[1]] = p[controls[1]], p[2]

    elif controls[0] == 0 and controls[1] == 2:
        #  controls[0] -> self, controls[1] -> target -> outside
        p[1], p[2] = p[2], p[1]
        p[1], p[target] = p[target], p[1]

    elif controls[1] == 1 and controls[0] == 2:
        #  controls[1] -> self, controls[0] -> target -> outside
        p[0], p[2] = p[2], p[0]
        p[0], p[target] = p[target], p[0]

    elif target == 2 and controls[0] == 1:
        #  target -> self, controls[0] -> controls[1] -> outside
        p[0], p[1] = p[1], p[0]
        p[0], p[controls[1]] = p[controls[1]], p[0]

    #
    # N > 4 cases
    #

    elif controls[0] == 1 and controls[1] > 2 and target > 2:
        #  controls[0] -> controls[1] -> outside, target -> outside
        p[0], p[1] = p[1], p[0]
        p[0], p[controls[1]] = p[controls[1]], p[0]
        p[2], p[target] = p[target], p[2]

    elif controls[0] == 2 and controls[1] > 2 and target > 2:
        #  controls[0] -> target -> outside, controls[1] -> outside
        p[0], p[2] = p[2], p[0]
        p[0], p[target] = p[target], p[0]
        p[1], p[controls[1]] = p[controls[1]], p[1]

    elif controls[1] == 2 and controls[0] > 2 and target > 2:
        #  controls[1] -> target -> outside, controls[0] -> outside
        p[1], p[2] = p[2], p[1]
        p[1], p[target] = p[target], p[1]
        p[0], p[controls[0]] = p[controls[0]], p[0]

    else:
        p[0], p[controls[0]] = p[controls[0]], p[0]
        p1[1], p1[controls[1]] = p1[controls[1]], p1[1]
        p2[2], p2[target] = p2[target], p2[2]
        p = [p[p1[p2[k]]] for k in range(N)]

    return tensor([U] + [identity(2)] * (N - 3)).permute(p)


def _check_qubits_oper(oper, dims=None, targets=None):
    """
    Check if the given operator is valid.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        The quantum object to be checked.
    dims : list, optional
        A list of integer for the dimension of each composite system.
        e.g ``[2, 2, 2, 2, 2]`` for 5 qubits system. If None, qubits system
        will be the default.
    targets : int or list of int, optional
        The indices of qubits that are acted on.
    """
    # if operator matches N
    if not isinstance(oper, Qobj) or oper.dims[0] != oper.dims[1]:
        raise ValueError(
            "The operator is not an "
            "Qobj with the same input and output dimensions.")
    # if operator dims matches the target dims
    if dims is not None and targets is not None:
        targ_dims = [dims[t] for t in targets]
        if oper.dims[0] != targ_dims:
            raise ValueError(
                "The operator dims {} do not match "
                "the target dims {}.".format(
                    oper.dims[0], targ_dims))


def _targets_to_list(targets, oper=None, N=None):
    """
    transform targets to a list and check validity.

    Parameters
    ----------
    targets : int or list of int
        The indices of qubits that are acted on.
    oper : :class:`qutip.Qobj`, optional
        An operator acts on qubits, the type of the :class:`qutip.Qobj`
        has to be an operator
        and the dimension matches the tensored qubit Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    N : int, optional
        The number of qubits in the system.
    """
    # if targets is a list of integer
    if targets is None:
        targets = list(range(len(oper.dims[0])))
    if not isinstance(targets, Iterable):
        targets = [targets]
    if not all([isinstance(t, numbers.Integral) for t in targets]):
        raise TypeError(
            "targets should be "
            "an integer or a list of integer")
    # if targets has correct length
    if oper is not None:
        req_num = len(oper.dims[0])
        if len(targets) != req_num:
            raise ValueError(
                "The given operator needs {} "
                "target qutbis, "
                "but {} given.".format(
                    req_num, len(targets)))
    # if targets is smaller than N
    if N is not None:
        if not all([t < N for t in targets]):
            raise ValueError("Targets must be smaller than N={}.".format(N))
    return targets


def expand_operator(oper, N, targets, dims=None, cyclic_permutation=False):
    """
    Expand a qubits operator to one that acts on a N-qubit system.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        An operator acts on qubits, the type of the :class:`qutip.Qobj`
        has to be an operator
        and the dimension matches the tensored qubit Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    N : int
        The number of qubits in the system.
    targets : int or list of int
        The indices of qubits that are acted on.
    dims : list, optional
        A list of integer for the dimension of each composite system.
        E.g ``[2, 2, 2, 2, 2]`` for 5 qubits system. If None, qubits system
        will be the default option.
    cyclic_permutation : boolean, optional
        Expand for all cyclic permutation of the targets.
        E.g. if ``N=3`` and `oper` is a 2-qubit operator,
        the result will be a list of three operators,
        each acting on qubits 0 and 1, 1 and 2, 2 and 0.

    Returns
    -------
    expanded_oper : :class:`qutip.Qobj`
        The expanded qubits operator acting on a system with N qubits.

    Notes
    -----
    This is equivalent to gate_expand_1toN, gate_expand_2toN,
    gate_expand_3toN in ``qutip.qip.gate.py``, but works for any dimension.
    """
    if dims is None:
        dims = [2] * N
    targets = _targets_to_list(targets, oper=oper, N=N)
    _check_qubits_oper(oper, dims=dims, targets=targets)

    # Call expand_operator for all cyclic permutation of the targets.
    if cyclic_permutation:
        oper_list = []
        for i in range(N):
            new_targets = np.mod(np.array(targets)+i, N)
            oper_list.append(
                expand_operator(oper, N=N, targets=new_targets, dims=dims))
        return oper_list

    # Generate the correct order for qubits permutation,
    # eg. if N = 5, targets = [3,0], the order is [1,2,3,0,4].
    # If the operator is cnot,
    # this order means that the 3rd qubit controls the 0th qubit.
    new_order = [0] * N
    for i, t in enumerate(targets):
        new_order[t] = i
    # allocate the rest qutbits (not targets) to the empty
    # position in new_order
    rest_pos = [q for q in list(range(N)) if q not in targets]
    rest_qubits = list(range(len(targets), N))
    for i, ind in enumerate(rest_pos):
        new_order[ind] = rest_qubits[i]
    id_list = [identity(dims[i]) for i in rest_pos]
    return tensor([oper] + id_list).permute(new_order)
