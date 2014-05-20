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

from numpy import sqrt, array, exp, where, prod
import scipy.sparse as sp
from qutip.states import (basis, qstate, state_number_index,
                          state_number_enumerate)
from qutip.qobj import Qobj
from qutip.operators import *
from qutip.tensor import tensor
from qutip.states import fock_dm


def rotation(op, phi, N=None, target=None):
    """Single-qubit rotation for operator op with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.
    """
    if not N is None and not target is None:
        return gate_expand_1toN(rotation(op, phi), N, target)
    else:
        return (-1j * op * phi / 2).expm()



def cphase(theta, N=2, control=0, target=1):
    """
    Returns quantum object representing the phase shift gate.

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


def rx(phi, N=None, target=None):
    """Single-qubit rotation for operator sigmax with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if not N is None and not N is None and not target is None:
        return gate_expand_1toN(rx(phi), N, target)
    else:
        return Qobj([[np.cos(phi / 2), -1j * np.sin(phi / 2)],
                     [-1j * np.sin(phi / 2), np.cos(phi / 2)]])


def ry(phi, N=None, target=None):
    """Single-qubit rotation for operator sigmay with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if not N is None and not target is None:
        return gate_expand_1toN(ry(phi), N, target)
    else:
        return Qobj([[np.cos(phi / 2), -np.sin(phi / 2)],
                     [np.sin(phi / 2), np.cos(phi / 2)]])


def rz(phi, N=None, target=None):
    """Single-qubit rotation for operator sigmaz with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if not N is None and not target is None:
        return gate_expand_1toN(rz(phi), N, target)
    else:
        return Qobj([[np.exp(-1j * phi / 2), 0],
                     [0, np.exp(1j * phi / 2)]])


def cnot(N=None, control=None, target=None):
    """
    Quantum object representing the CNOT gate.

    Returns
    -------
    cnot_gate : qobj
        Quantum object representation of CNOT gate

    Examples
    --------
    >>> cnot()
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]]

    """
    if not N is None and not control is None and not target is None:
        return gate_expand_2toN(cnot(), N, control, target)
    else:
        uu = tensor(basis(2), basis(2))
        ud = tensor(basis(2), basis(2, 1))
        du = tensor(basis(2, 1), basis(2))
        dd = tensor(basis(2, 1), basis(2, 1))
        Q = uu * uu.dag() + ud * ud.dag() + dd * du.dag() + du * dd.dag()
        return Qobj(Q)


def fredkin(N=None, control1=None, control2=None, target=None):
    """Quantum object representing the Fredkin gate.

    Returns
    -------
    fred_gate : qobj
        Quantum object representation of Fredkin gate.

    Examples
    --------
    >>> fredkin()
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
    if (not N is None and not control1 is None
            and not control2 is None and not target is None):
        return gate_expand_3toN(fredkin(), N, control1, control2, target)

    else:
        uuu = qstate('uuu')
        uud = qstate('uud')
        udu = qstate('udu')
        udd = qstate('udd')
        duu = qstate('duu')
        dud = qstate('dud')
        ddu = qstate('ddu')
        ddd = qstate('ddd')
        Q = ddd * ddd.dag() + ddu * ddu.dag() + dud * dud.dag() + \
            duu * duu.dag() + udd * udd.dag() + uud * udu.dag() + \
            udu * uud.dag() + uuu * uuu.dag()
        return Qobj(Q)


def toffoli(N=None, control1=None, control2=None, target=None):
    """Quantum object representing the Toffoli gate.

    Returns
    -------
    toff_gate : qobj
        Quantum object representation of Toffoli gate.

    Examples
    --------
    >>> toffoli()
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
    if (not N is None and not control1 is None
            and not control2 is None and not target is None):
        return gate_expand_3toN(toffoli(), N, control1, control2, target)

    else:
        uuu = qstate('uuu')
        uud = qstate('uud')
        udu = qstate('udu')
        udd = qstate('udd')
        duu = qstate('duu')
        dud = qstate('dud')
        ddu = qstate('ddu')
        ddd = qstate('ddd')
        Q = ddd * ddd.dag() + ddu * ddu.dag() + dud * dud.dag() + \
            duu * duu.dag() + udd * udd.dag() + udu * udu.dag() + \
            uuu * uud.dag() + uud * uuu.dag()
        return Qobj(Q)


def swap(N=None, control=None, target=None, mask=None):
    """Quantum object representing the SWAP gate.

    Returns
    -------
    swap_gate : qobj
        Quantum object representation of SWAP gate

    Examples
    --------
    >>> swap()
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]

    """
    if mask:
        warnings.warn("The mask argument to iswap is deprecated. " +
                      "Use the N, control and target arguments instead.")

        if sum(mask) != 2:
            raise ValueError("mask must only have two ones, rest zeros")

        dims = [2] * len(mask)
        idx, = where(mask)
        N = prod(dims)
        data = sp.lil_matrix((N, N))

        for s1 in state_number_enumerate(dims):
            i1 = state_number_index(dims, s1)

            if s1[idx[0]] == s1[idx[1]]:
                i2 = i1
            else:
                s2 = array(s1).copy()
                s2[idx[0]], s2[idx[1]] = s2[idx[1]], s2[idx[0]]
                i2 = state_number_index(dims, s2)

            data[i1, i2] = 1

        return Qobj(data, dims=[dims, dims], shape=[N, N])

    elif not N is None and not control is None and not target is None:
        return gate_expand_2toN(swap(), N, control, target)

    else:
        uu = qstate('uu')
        ud = qstate('ud')
        du = qstate('du')
        dd = qstate('dd')
        Q = uu * uu.dag() + ud * du.dag() + du * ud.dag() + dd * dd.dag()
        return Q


def iswap(N=None, control=None, target=None, mask=None):
    """Quantum object representing the iSWAP gate.

    Returns
    -------
    iswap_gate : qobj
        Quantum object representation of iSWAP gate

    Examples
    --------
    >>> iswap()
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]
     [ 0.+0.j  0.+1.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]
    """
    if mask:
        warnings.warn("The mask argument to iswap is deprecated. " +
                      "Use the N, control and target arguments instead.")

        if sum(mask) != 2:
            raise ValueError("mask must only have two ones, rest zeros")

        dims = [2] * len(mask)
        idx, = where(mask)
        N = prod(dims)
        data = sp.lil_matrix((N, N), dtype=complex)

        for s1 in state_number_enumerate(dims):
            i1 = state_number_index(dims, s1)

            if s1[idx[0]] == s1[idx[1]]:
                i2 = i1
                val = 1.0
            else:
                s2 = s1.copy()
                s2[idx[0]], s2[idx[1]] = s2[idx[1]], s2[idx[0]]
                i2 = state_number_index(dims, s2)
                val = 1.0j

            data[i1, i2] = val

        return Qobj(data, dims=[dims, dims], shape=[N, N])

    elif not N is None and not control is None and not target is None:
        return gate_expand_2toN(iswap(), N, control, target)

    else:
        return Qobj(array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0],
                           [0, 0, 0, 1]]),
                    dims=[[2, 2], [2, 2]])


def sqrtiswap(N=None, control=None, target=None):
    """Quantum object representing the square root iSWAP gate.

    Returns
    -------
    sqrtiswap_gate : qobj
        Quantum object representation of square root iSWAP gate

    Examples
    --------
    >>> sqrtiswap()
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
    if not N is None and not control is None and not target is None:
        return gate_expand_2toN(sqrtiswap(), N, control, target)
    else:
        return Qobj(array([[1, 0, 0, 0],
                           [0, 1 / sqrt(2), -1j / sqrt(2), 0],
                           [0, -1j / sqrt(2), 1 / sqrt(2), 0],
                           [0, 0, 0, 1]]), dims=[[2, 2], [2, 2]])


def sqrtswap(N=None, control=None, target=None):
    """Quantum object representing the square root SWAP gate.

    Returns
    -------
    sqrtswap_gate : qobj
        Quantum object representation of square root SWAP gate

    """
    if not N is None and not control is None and not target is None:
        return gate_expand_2toN(sqrtswap(), N, control, target)
    else:
        return Qobj(array([[1, 0, 0, 0],
                           [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                           [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                           [0, 0, 0, 1]]),
                    dims=[[2, 2], [2, 2]])


def snot(N=None, target=0):
    """Quantum object representing the SNOT (2-qubit Hadamard) gate.

    Returns
    -------
    snot_gate : qobj
        Quantum object representation of SNOT gate.

    Examples
    --------
    >>> snot()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 0.70710678+0.j  0.70710678+0.j]
     [ 0.70710678+0.j -0.70710678+0.j]]

    """
    if not N is None:
        return gate_expand_1toN(snot(), N, target)
    else:
        u = basis(2, 0)
        d = basis(2, 1)
        Q = 1.0 / sqrt(2.0) * (sigmax() + sigmaz())
        return Q


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


def hadamard(N=1):
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


def phasegate(theta, N=None, target=None):
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
    >>> phasegate(pi/4)
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 1.00000000+0.j          0.00000000+0.j        ]
     [ 0.00000000+0.j          0.70710678+0.70710678j]]

    """
    if not N is None and not target is None:
        return gate_expand_1toN(phasegate(theta), N, target)
    else:
        u = basis(2)
        d = basis(2, 1)
        Q = u * u.dag() + (exp(1.0j * theta) * d * d.dag())
        return Qobj(Q)


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


def gate_expand_2toN(U, N, control, target):
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

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.
    """

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


def gate_expand_3toN(U, N, control1, control2, target):
    """
    Create a Qobj representing a three-qubit gate that act on a system with N
    qubits.

    Parameters
    ----------
    U : Qobj
        The three-qubit gate

    N : integer
        The number of qubits in the target space.

    control1 : integer
        The index of the first control qubit.

    control2 : integer
        The index of the second control qubit.

    target : integer
        The index of the target qubit.

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.
    """

    if N < 3:
        raise ValueError("integer N must be larger or equal to 3")

    if control1 >= N or control2 >= N or target >= N:
        raise ValueError(
            "control and not target is None must be integer < integer N")

    if control1 == target or control2 == target or control1 == control2:
        raise ValueError("control1, control2, and target cannot be equal")

    p = list(range(N))
    p1 = list(range(N))
    p2 = list(range(N))

    if control1 <= 2 and control2 <= 2 and target <= 2:
        p[0], p[control1] = p[control1], p[0]
        p1[1], p1[control2] = p1[control2], p1[1]
        p2[2], p2[target] = p2[target], p2[2]
        p = [p[p1[p2[k]]] for k in range(N)]

    #
    # N >= 3 cases
    #

    elif control1 == 0 and control2 == 1:
        p[2], p[target] = p[target], p[2]

    elif control1 == 0 and target == 2:
        p[1], p[control2] = p[control2], p[1]

    elif control2 == 1 and target == 2:
        p[0], p[control1] = p[control1], p[0]

    elif control1 == 1 and control2 == 0:
        p[control2], p[control1] = p[control1], p[control2]
        p2[2], p2[target] = p2[target], p2[2]
        p = [p2[p[k]] for k in range(N)]

    elif control1 == 2 and target == 0:
        p[target], p[control1] = p[control1], p[target]
        p1[1], p1[control2] = p1[control2], p1[1]
        p = [p1[p[k]] for k in range(N)]

    elif control2 == 2 and target == 1:
        p[target], p[control2] = p[control2], p[target]
        p1[0], p1[control1] = p1[control1], p1[0]
        p = [p1[p[k]] for k in range(N)]

    elif control1 == 1 and control2 == 2:
        #  control1 -> control2 -> target -> outside
        p[0], p[1] = p[1], p[0]
        p[0], p[2] = p[2], p[0]
        p[0], p[target] = p[target], p[0]

    elif control1 == 2 and target == 1:
        #  control1 -> target -> control2 -> outside
        p[0], p[2] = p[2], p[0]
        p[0], p[1] = p[1], p[0]
        p[0], p[control2] = p[control2], p[0]

    elif control2 == 0 and control1 == 2:
        #  control2 -> control1 -> target -> outside
        p[1], p[0] = p[0], p[1]
        p[1], p[2] = p[2], p[1]
        p[1], p[target] = p[target], p[1]

    elif control2 == 2 and target == 0:
        #  control2 -> target -> control1 -> outside
        p[1], p[2] = p[2], p[1]
        p[1], p[0] = p[0], p[1]
        p[1], p[control1] = p[control1], p[1]

    elif target == 1 and control2 == 0:
        #  target -> control2 -> control1 -> outside
        p[2], p[1] = p[1], p[2]
        p[2], p[0] = p[0], p[2]
        p[2], p[control1] = p[control1], p[2]

    elif target == 0 and control1 == 1:
        #  target -> control1 -> control2 -> outside
        p[2], p[0] = p[0], p[2]
        p[2], p[1] = p[1], p[2]
        p[2], p[control2] = p[control2], p[2]

    elif control1 == 0 and control2 == 2:
        #  control1 -> self, control2 -> target -> outside
        p[1], p[2] = p[2], p[1]
        p[1], p[target] = p[target], p[1]

    elif control2 == 1 and control1 == 2:
        #  control2 -> self, control1 -> target -> outside
        p[0], p[2] = p[2], p[0]
        p[0], p[target] = p[target], p[0]

    elif target == 2 and control1 == 1:
        #  target -> self, control1 -> control2 -> outside
        p[0], p[1] = p[1], p[0]
        p[0], p[control2] = p[control2], p[0]

    #
    # N >= 4 cases
    #

    elif control1 == 1 and control2 > 2 and target > 2:
        #  control1 -> control2 -> outside, target -> outside
        p[0], p[1] = p[1], p[0]
        p[0], p[control2] = p[control2], p[0]
        p[2], p[target] = p[target], p[2]

    elif control1 == 2 and control2 > 2 and target > 2:
        #  control1 -> target -> outside, control2 -> outside
        p[0], p[2] = p[2], p[0]
        p[0], p[target] = p[target], p[0]
        p[1], p[control2] = p[control2], p[1]

    elif control2 == 2 and control1 > 2 and target > 2:
        #  control2 -> target -> outside, control1 -> outside
        p[1], p[2] = p[2], p[1]
        p[1], p[target] = p[target], p[1]
        p[0], p[control1] = p[control1], p[0]

    else:
        p[0], p[control1] = p[control1], p[0]
        p1[1], p1[control2] = p1[control2], p1[1]
        p2[2], p2[target] = p2[target], p2[2]
        p = [p[p1[p2[k]]] for k in range(N)]

    return tensor([U] + [identity(2)] * (N - 3)).permute(p)
