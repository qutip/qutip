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
        return Qobj([[np.cos(phi/2), -1j * np.sin(phi/2)],
                     [-1j * np.sin(phi/2), np.cos(phi/2)]])


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
        return Qobj([[np.cos(phi/2), -np.sin(phi/2)],
                     [np.sin(phi/2), np.cos(phi/2)]])


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
        uu = tensor(basis(2),basis(2))
        ud = tensor(basis(2),basis(2,1))
        du = tensor(basis(2,1),basis(2))
        dd = tensor(basis(2,1),basis(2,1))
        Q = uu * uu.dag() + ud * ud.dag() + dd * du.dag() + du * dd.dag()
        return Qobj(Q)


def fredkin():
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


def toffoli(target=2):
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
                      "Use the N, control and not target is None arguments instead.")

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
                      "Use the N, control and not target is None arguments instead.")

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


def snot(N=None, target=None):
    """Quantum object representing the SNOT (Hadamard) gate.

    Returns
    -------
    snot_gate : qobj
        Quantum object representation of SNOT (Hadamard) gate.

    Examples
    --------
    >>> snot()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 0.70710678+0.j  0.70710678+0.j]
     [ 0.70710678+0.j -0.70710678+0.j]]

    """
    if not N is None and not target is None:
        return gate_expand_1toN(snot(), N, target)
    else:
        u = basis(2,0)
        d = basis(2,1)
        Q = 1.0 / sqrt(2.0) * (sigmax()+sigmaz())
        return Q


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
        d = basis(2,1)
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
        raise ValueError("control and not target is None must be integer < integer N")

    if control == target:
        raise ValueError("target and not control is None cannot be equal")

    p = list(range(N))
    p[0], p[control] = p[control], p[0]
    p[1], p[target] = p[target], p[1]

    return tensor([U] +[identity(2)] * (N - 2)).permute(p)

