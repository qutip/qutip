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
This module contains functions for generating Qobj representation of a variety
of commonly occuring quantum operators.
"""

__all__ = ['jmat', 'spin_Jx', 'spin_Jy', 'spin_Jz', 'spin_Jm', 'spin_Jp',
           'spin_J_set', 'sigmap', 'sigmam', 'sigmax', 'sigmay', 'sigmaz',
           'destroy', 'create', 'qeye', 'identity', 'position', 'momentum',
           'num', 'squeeze', 'squeezing', 'displace', 'commutator',
           'qutrit_ops', 'qdiags', 'phase', 'qzero', 'charge', 'tunneling']

import numbers

import numpy as np
import scipy.sparse

from . import data as _data
from .qobj import Qobj
from .dimensions import flatten, Space


def qdiags(diagonals, offsets, dims=None, shape=None, *, dtype=_data.CSR):
    """
    Constructs an operator from an array of diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Array of elements to place along the selected diagonals.

    offsets : sequence of ints
        Sequence for diagonals to be set:
            - k=0 main diagonal
            - k>0 kth upper diagonal
            - k<0 kth lower diagonal
    dims : list, optional
        Dimensions for operator

    shape : list, tuple, optional
        Shape of operator.  If omitted, a square operator large enough
        to contain the diagonals is generated.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Examples
    --------
    >>> qdiags(sqrt(range(1, 4)), 1) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isherm = False
    Qobj data =
    [[ 0.          1.          0.          0.        ]
     [ 0.          0.          1.41421356  0.        ]
     [ 0.          0.          0.          1.73205081]
     [ 0.          0.          0.          0.        ]]

    """
    data = _data.diag[dtype](diagonals, offsets, shape)
    return Qobj(data, dims=dims, type='oper', copy=False)


def jmat(j, which=None, *, dtype=_data.CSR):
    """Higher-order spin operators:

    Parameters
    ----------
    j : float
        Spin of operator

    which : str
        Which operator to return 'x','y','z','+','-'.
        If no args given, then output is ['x','y','z']

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    jmat : Qobj or tuple of Qobj
        ``qobj`` for requested spin operator(s).


    Examples
    --------
    >>> jmat(1) # doctest: +SKIP
    [ Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.          0.70710678  0.        ]
     [ 0.70710678  0.          0.70710678]
     [ 0.          0.70710678  0.        ]]
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j          0.-0.70710678j  0.+0.j        ]
     [ 0.+0.70710678j  0.+0.j          0.-0.70710678j]
     [ 0.+0.j          0.+0.70710678j  0.+0.j        ]]
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 1.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0. -1.]]]


    Notes
    -----
    If no 'args' input, then returns array of ['x','y','z'] operators.

    """
    if int(2 * j) != 2 * j or j < 0:
        raise ValueError('j must be a non-negative integer or half-integer')

    if not which:
        return (
            jmat(j, 'x', dtype=dtype),
            jmat(j, 'y', dtype=dtype),
            jmat(j, 'z', dtype=dtype)
        )

    dims = [[int(2*j + 1)]]*2
    if which == '+':
        return Qobj(_jplus(j, dtype=dtype), dims=dims, type='oper',
                    isherm=False, isunitary=False, copy=False)
    if which == '-':
        return Qobj(_jplus(j, dtype=dtype).adjoint(), dims=dims, type='oper',
                    isherm=False, isunitary=False, copy=False)
    if which == 'x':
        A =  _jplus(j, dtype=dtype)
        return Qobj(_data.add(A, A.adjoint()), dims=dims, type='oper',
                    isherm=True, isunitary=False, copy=False) * 0.5
    if which == 'y':
        A =  _data.mul(_jplus(j, dtype=dtype), -0.5j)
        return Qobj(_data.add(A, A.adjoint()), dims=dims, type='oper',
                    isherm=True, isunitary=False, copy=False)
    if which == 'z':
        return Qobj(_jz(j, dtype=dtype), dims=dims, type='oper',
                    isherm=True, isunitary=False, copy=False)
    raise ValueError('invalid spin operator: ' + which)


def _jplus(j, *, dtype=_data.CSR):
    """
    Internal functions for generating the data representing the J-plus
    operator.
    """
    m = np.arange(j, -j - 1, -1, dtype=complex)
    data = np.sqrt(j * (j + 1) - m * (m + 1))[1:]
    return _data.diag[dtype](data, 1)


def _jz(j, *, dtype=_data.CSR):
    """
    Internal functions for generating the data representing the J-z operator.
    """
    N = int(2*j + 1)
    data = np.array([j-k for k in range(N)], dtype=complex)
    return _data.diag[dtype](data, 0)


#
# Spin j operators:
#
def spin_Jx(j, *, dtype=_data.CSR):
    """Spin-j x operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'x', dtype=dtype)


def spin_Jy(j, *, dtype=_data.CSR):
    """Spin-j y operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'y', dtype=dtype)


def spin_Jz(j, *, dtype=_data.CSR):
    """Spin-j z operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'z', dtype=dtype)


def spin_Jm(j, *, dtype=_data.CSR):
    """Spin-j annihilation operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '-', dtype=dtype)


def spin_Jp(j, *, dtype=_data.CSR):
    """Spin-j creation operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '+', dtype=dtype)


def spin_J_set(j, *, dtype=_data.CSR):
    """Set of spin-j operators (x, y, z)

    Parameters
    ----------
    j : float
        Spin of operators

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    list : list of Qobj
        list of ``qobj`` representating of the spin operator.

    """
    return jmat(j, dtype=dtype)


# Pauli spin-1/2 operators.
#
# These are so common in quantum information that we want them to be
# near-instantaneous to initialise, so we cache them at package import, and
# just return copies when someone requests one.
_SIGMAP = jmat(0.5, '+')
_SIGMAM = jmat(0.5, '-')
_SIGMAX = 2 * jmat(0.5, 'x')
_SIGMAY = 2 * jmat(0.5, 'y')
_SIGMAZ = 2 * jmat(0.5, 'z')


def sigmap():
    """Creation operator for Pauli spins.

    Examples
    --------
    >>> sigmap() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 0.  0.]]

    """
    return _SIGMAP.copy()


def sigmam():
    """Annihilation operator for Pauli spins.

    Examples
    --------
    >>> sigmam() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  0.]
     [ 1.  0.]]

    """
    return _SIGMAM.copy()


def sigmax():
    """Pauli spin 1/2 sigma-x operator

    Examples
    --------
    >>> sigmax() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 1.  0.]]

    """
    return _SIGMAX.copy()


def sigmay():
    """Pauli spin 1/2 sigma-y operator.

    Examples
    --------
    >>> sigmay() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.-1.j]
     [ 0.+1.j  0.+0.j]]

    """
    return _SIGMAY.copy()


def sigmaz():
    """Pauli spin 1/2 sigma-z operator.

    Examples
    --------
    >>> sigmaz() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]

    """
    return _SIGMAZ.copy()


def destroy(N, offset=0, *, dtype=_data.CSR):
    """
    Destruction (lowering) operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Qobj for lowering operator.

    Examples
    --------
    >>> destroy(4) # doctest: +SKIP
    Quantum object: dims=[[4], [4]], shape=(4, 4), type='oper', isherm=False
    Qobj data =
    [[ 0.00000000+0.j  1.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  1.41421356+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  1.73205081+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]]
    """
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset+1, N+offset, dtype=complex))
    return qdiags(data, 1, dtype=dtype)


def create(N, offset=0, *, dtype=_data.CSR):
    """
    Creation (raising) operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Qobj for raising operator.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Examples
    --------
    >>> create(4) # doctest: +SKIP
    Quantum object: dims=[[4], [4]], shape=(4, 4), type='oper', isherm=False
    Qobj data =
    [[ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 1.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  1.41421356+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  1.73205081+0.j  0.00000000+0.j]]
    """
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset+1, N+offset, dtype=complex))
    return qdiags(data, -1, dtype=dtype)


def _implicit_tensor_dimensions(dimensions):
    """
    Total flattened size and operator dimensions for operator creation routines
    that automatically perform tensor products.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        First dimension of an operator which can create an implicit tensor
        product.  If the type is `int`, it is promoted first to `[dimensions]`.
        From there, it should be one of the two-elements `dims` parameter of a
        `qutip.Qobj` representing an `oper` or `super`, with possible tensor
        products.

    Returns
    -------
    size : int
        Dimension of backing matrix required to represent operator.
    dimensions : list
        Dimension list in the form required by ``Qobj`` creation.
    """
    if not isinstance(dimensions, (list, Space)):
        dimensions = [dimensions]
    dimensions = Space(dimensions)
    flat = dimensions.flat()
    return np.prod(flat), [dimensions, dimensions]


def qzero(dimensions, *, dtype=_data.CSR):
    """
    Zero operator.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    qzero : qobj
        Zero operator Qobj.

    """
    size, dimensions = _implicit_tensor_dimensions(dimensions)
    # A sparse matrix with no data is equal to a zero matrix.
    return Qobj(_data.zeros[dtype](size, size), dims=dimensions,
                isherm=True, isunitary=False, copy=False)


def qeye(dimensions, *, dtype=_data.CSR):
    """
    Identity operator.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Identity operator Qobj.

    Examples
    --------
    >>> qeye(3) # doctest: +SKIP
    Quantum object: dims = [[3], [3]], shape = (3, 3), type = oper, \
isherm = True
    Qobj data =
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    >>> qeye([2,2]) # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, \
isherm = True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    """
    size, dimensions = _implicit_tensor_dimensions(dimensions)
    return Qobj(_data.identity[dtype](size), dims=dimensions,
                isherm=True, isunitary=True, copy=False)


# Name alias.
identity = qeye


def position(N, offset=0, *, dtype=_data.CSR):
    """
    Position operator x=1/sqrt(2)*(a+a.dag())

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Position operator as Qobj.
    """
    a = destroy(N, offset=offset, dtype=dtype)
    return np.sqrt(0.5) * (a + a.dag())


def momentum(N, offset=0, *, dtype=_data.CSR):
    """
    Momentum operator p=-1j/sqrt(2)*(a-a.dag())

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Momentum operator as Qobj.
    """
    a = destroy(N, offset=offset, dtype=dtype)
    return -1j * np.sqrt(0.5) * (a - a.dag())


def num(N, offset=0, *, dtype=_data.CSR):
    """
    Quantum object for number operator.

    Parameters
    ----------
    N : int
        The dimension of the Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper: qobj
        Qobj for number operator.

    Examples
    --------
    >>> num(4) # doctest: +SKIP
    Quantum object: dims=[[4], [4]], shape=(4, 4), type='oper', isherm=True
    Qobj data =
    [[0 0 0 0]
     [0 1 0 0]
     [0 0 2 0]
     [0 0 0 3]]
    """
    data = np.arange(offset, offset + N, dtype=complex)
    return qdiags(data, 0, dtype=dtype)


def squeeze(N, z, offset=0, *, dtype=_data.CSR):
    """Single-mode squeezing operator.

    Parameters
    ----------
    N : int
        Dimension of hilbert space.

    z : float/complex
        Squeezing parameter.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : :class:`qutip.qobj.Qobj`
        Squeezing operator.


    Examples
    --------
    >>> squeeze(4, 0.25) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.98441565+0.j  0.00000000+0.j  0.17585742+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.95349007+0.j  0.00000000+0.j  0.30142443+0.j]
     [-0.17585742+0.j  0.00000000+0.j  0.98441565+0.j  0.00000000+0.j]
     [ 0.00000000+0.j -0.30142443+0.j  0.00000000+0.j  0.95349007+0.j]]

    """
    asq = destroy(N, offset=offset) ** 2
    op = 0.5*np.conj(z)*asq - 0.5*z*asq.dag()
    return op.expm(dtype=dtype)


def squeezing(a1, a2, z):
    """Generalized squeezing operator.

    .. math::

        S(z) = \\exp\\left(\\frac{1}{2}\\left(z^*a_1a_2
        - za_1^\\dagger a_2^\\dagger\\right)\\right)

    Parameters
    ----------
    a1 : :class:`qutip.qobj.Qobj`
        Operator 1.

    a2 : :class:`qutip.qobj.Qobj`
        Operator 2.

    z : float/complex
        Squeezing parameter.

    Returns
    -------
    oper : :class:`qutip.qobj.Qobj`
        Squeezing operator.

    """
    b = 0.5 * (np.conj(z)*(a1 @ a2) - z*(a1.dag() @ a2.dag()))
    return b.expm()


def displace(N, alpha, offset=0, *, dtype=_data.Dense):
    """Single-mode displacement operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    alpha : float/complex
        Displacement amplitude.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Displacement operator.

    Examples
    ---------
    >>> displace(4,0.25) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.96923323+0.j -0.24230859+0.j  0.04282883+0.j -0.00626025+0.j]
     [ 0.24230859+0.j  0.90866411+0.j -0.33183303+0.j  0.07418172+0.j]
     [ 0.04282883+0.j  0.33183303+0.j  0.84809499+0.j -0.41083747+0.j]
     [ 0.00626025+0.j  0.07418172+0.j  0.41083747+0.j  0.90866411+0.j]]

    """
    a = destroy(N, offset=offset)
    return (alpha * a.dag() - np.conj(alpha) * a).expm(dtype=dtype)


def commutator(A, B, kind="normal"):
    """
    Return the commutator of kind `kind` (normal, anti) of the
    two operators A and B.
    """
    if kind == 'normal':
        return A @ B - B @ A

    elif kind == 'anti':
        return A @ B + B @ A

    else:
        raise TypeError("Unknown commutator kind '%s'" % kind)


def qutrit_ops(*, dtype=_data.CSR):
    """
    Operators for a three level system (qutrit).

    Parameters
    ----------
    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    opers: array
        `array` of qutrit operators.

    """
    from .states import qutrit_basis

    out = np.empty((6,), dtype=object)
    one, two, three = qutrit_basis(dtype=dtype)
    out[0] = one * one.dag()
    out[1] = two * two.dag()
    out[2] = three * three.dag()
    out[3] = one * two.dag()
    out[4] = two * three.dag()
    out[5] = three * one.dag()
    return out


def phase(N, phi0=0, *, dtype=_data.Dense):
    """
    Single-mode Pegg-Barnett phase operator.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    phi0 : float
        Reference phase.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Phase operator with respect to reference phase.

    Notes
    -----
    The Pegg-Barnett phase operator is Hermitian on a truncated Hilbert space.

    """
    phim = phi0 + (2 * np.pi * np.arange(N)) / N  # discrete phase angles
    n = np.arange(N)[:, np.newaxis]
    states = np.array([np.sqrt(kk) / np.sqrt(N) * np.exp(1j * n * kk)
                       for kk in phim])
    ops = np.sum([np.outer(st, st.conj()) for st in states], axis=0)
    return Qobj(ops, dims=[[N], [N]], type='oper', copy=False).to(dtype)


def charge(Nmax, Nmin=None, frac=1, *, dtype=_data.CSR):
    """
    Generate the diagonal charge operator over charge states
    from Nmin to Nmax.

    Parameters
    ----------
    Nmax : int
        Maximum charge state to consider.

    Nmin : int (default = -Nmax)
        Lowest charge state to consider.

    frac : float (default = 1)
        Specify fractional charge if needed.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    C : Qobj
        Charge operator over [Nmin, Nmax].

    Notes
    -----
    .. versionadded:: 3.2

    """
    if Nmin is None:
        Nmin = -Nmax
    diag = frac * np.arange(Nmin, Nmax+1, dtype=float)
    out = qdiags(diag, 0, dtype=dtype)
    out.isherm = True
    return out


def tunneling(N, m=1, *, dtype=_data.CSR):
    r"""
    Tunneling operator with elements of the form
    :math:`\\sum |N><N+m| + |N+m><N|`.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    m : int (default = 1)
        Number of excitations in tunneling event.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    T : Qobj
        Tunneling operator.

    Notes
    -----
    .. versionadded:: 3.2

    """
    diags = [np.ones(N-m, dtype=int), np.ones(N-m, dtype=int)]
    T = qdiags(diags, [m, -m], dtype=dtype)
    T.isherm = True
    return T
