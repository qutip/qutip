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
           'qutrit_ops', 'qdiags', 'phase', 'qzero', 'enr_destroy',
           'enr_identity', 'charge', 'tunneling']

import numpy as np
import scipy
import scipy.sparse as sp

from qutip.qobj import Qobj


#
# Spin operators
#
def jmat(j, *args):
    """Higher-order spin operators:

    Parameters
    ----------
    j : float
        Spin of operator

    args : str
        Which operator to return 'x','y','z','+','-'.
        If no args given, then output is ['x','y','z']

    Returns
    -------
    jmat : qobj / ndarray
        ``qobj`` for requested spin operator(s).


    Examples
    --------
    >>> jmat(1)
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
    if (scipy.fix(2 * j) != 2 * j) or (j < 0):
        raise TypeError('j must be a non-negative integer or half-integer')

    if not args:
        return jmat(j, 'x'), jmat(j, 'y'), jmat(j, 'z') 

    if args[0] == '+':
        A = _jplus(j)
    elif args[0] == '-':
        A = _jplus(j).conj().T
    elif args[0] == 'x':
        A = 0.5 * (_jplus(j) + _jplus(j).conj().T)
    elif args[0] == 'y':
        A = -0.5 * 1j * (_jplus(j) - _jplus(j).conj().T)
    elif args[0] == 'z':
        A = _jz(j)
    else:
        raise TypeError('Invalid type')

    return Qobj(A.tocsr())


def _jplus(j):
    """
    Internal functions for generating the data representing the J-plus
    operator.
    """
    m = np.arange(j, -j - 1, -1)
    N = len(m)
    return sp.spdiags(np.sqrt(j * (j + 1.0) - (m + 1.0) * m),
                      1, N, N, format='csr')


def _jz(j):
    """
    Internal functions for generating the data representing the J-z operator.
    """
    m = np.arange(j, -j - 1, -1)
    N = len(m)
    return sp.spdiags(m, 0, N, N, format='csr')


#
# Spin j operators:
#
def spin_Jx(j):
    """Spin-j x operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'x')


def spin_Jy(j):
    """Spin-j y operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'y')


def spin_Jz(j):
    """Spin-j z operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'z')


def spin_Jm(j):
    """Spin-j annihilation operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '-')


def spin_Jp(j):
    """Spin-j creation operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '+')


def spin_J_set(j):
    """Set of spin-j operators (x, y, z)

    Parameters
    ----------
    j : float
        Spin of operators

    Returns
    -------
    list : list of Qobj
        list of ``qobj`` representating of the spin operator.

    """
    return jmat(j)


#
# Pauli spin 1/2 operators:
#
def sigmap():
    """Creation operator for Pauli spins.

    Examples
    --------
    >>> sigmam()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 0.  0.]]

    """
    return jmat(1 / 2., '+')


def sigmam():
    """Annihilation operator for Pauli spins.

    Examples
    --------
    >>> sigmam()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  0.]
     [ 1.  0.]]

    """
    return jmat(1 / 2., '-')


def sigmax():
    """Pauli spin 1/2 sigma-x operator

    Examples
    --------
    >>> sigmax()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 1.  0.]]

    """
    return 2.0 * jmat(1.0 / 2, 'x')


def sigmay():
    """Pauli spin 1/2 sigma-y operator.

    Examples
    --------
    >>> sigmay()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.-1.j]
     [ 0.+1.j  0.+0.j]]

    """
    return 2.0 * jmat(1.0 / 2, 'y')


def sigmaz():
    """Pauli spin 1/2 sigma-z operator.

    Examples
    --------
    >>> sigmaz()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]

    """
    return 2.0 * jmat(1.0 / 2, 'z')


#
# DESTROY returns annihilation operator for N dimensional Hilbert space
# out = destroy(N), N is integer value &  N>0
#
def destroy(N, offset=0):
    '''Destruction (lowering) operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : qobj
        Qobj for lowering operator.

    Examples
    --------
    >>> destroy(4)
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.00000000+0.j  1.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  1.41421356+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  1.73205081+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]]

    '''
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    return Qobj(sp.spdiags(np.sqrt(range(offset, N+offset)),
                           1, N, N, format='csr'), isherm=False)


#
# create returns creation operator for N dimensional Hilbert space
# out = create(N), N is integer value &  N>0
#
def create(N, offset=0):
    '''Creation (raising) operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    Returns
    -------
    oper : qobj
        Qobj for raising operator.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Examples
    --------
    >>> create(4)
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 1.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  1.41421356+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  1.73205081+0.j  0.00000000+0.j]]

    '''
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    qo = destroy(N, offset=offset)  # create operator using destroy function
    qo.data = qo.data.T.tocsr()  # transpose data in Qobj and convert to csr
    return qo


#
# QEYE returns identity operator for an N dimensional space
# a = qeye(N), N is integer & N>0
#
def qeye(N):
    """
    Identity operator

    Parameters
    ----------
    N : int or list of ints
        Dimension of Hilbert space. If provided as a list of ints,
        then the dimension is the product over this list, but the
        ``dims`` property of the new Qobj are set to this list.

    Returns
    -------
    oper : qobj
        Identity operator Qobj.

    Examples
    --------
    >>> qeye(3)
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]

    """
    if isinstance(N, list):
        return tensor.tensor(*[identity(n) for n in N])
    N = int(N)
    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N>=0")
    return Qobj(sp.eye(N, N, dtype=complex, format='csr'), isherm=True)


def identity(N):
    """Identity operator. Alternative name to :func:`qeye`.

    Parameters
    ----------
    N : int or list of ints
        Dimension of Hilbert space. If provided as a list of ints,
        then the dimension is the product over this list, but the
        ``dims`` property of the new Qobj are set to this list.

    Returns
    -------
    oper : qobj
        Identity operator Qobj.
    """
    return qeye(N)


def position(N, offset=0):
    """
    Position operator x=1/sqrt(2)*(a+a.dag())

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : qobj
        Position operator as Qobj.
    """
    a = destroy(N, offset=offset)
    return 1.0 / np.sqrt(2.0) * (a + a.dag())


def momentum(N, offset=0):
    """
    Momentum operator p=-1j/sqrt(2)*(a-a.dag())

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : qobj
        Momentum operator as Qobj.
    """
    a = destroy(N, offset=offset)
    return -1j / np.sqrt(2.0) * (a - a.dag())


def num(N, offset=0):
    """Quantum object for number operator.

    Parameters
    ----------
    N : int
        The dimension of the Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper: qobj
        Qobj for number operator.

    Examples
    --------
    >>> num(4)
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[0 0 0 0]
     [0 1 0 0]
     [0 0 2 0]
     [0 0 0 3]]

    """
    data = sp.spdiags(np.arange(offset, offset + N), 0, N, N, format='csr')
    return Qobj(data)


def squeeze(N, z, offset=0):
    """Single-mode Squeezing operator.


    Parameters
    ----------
    N : int
        Dimension of hilbert space.

    z : float/complex
        Squeezing parameter.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : :class:`qutip.qobj.Qobj`
        Squeezing operator.


    Examples
    --------
    >>> squeeze(4, 0.25)
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.98441565+0.j  0.00000000+0.j  0.17585742+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.95349007+0.j  0.00000000+0.j  0.30142443+0.j]
     [-0.17585742+0.j  0.00000000+0.j  0.98441565+0.j  0.00000000+0.j]
     [ 0.00000000+0.j -0.30142443+0.j  0.00000000+0.j  0.95349007+0.j]]

    """
    a = destroy(N, offset=offset)
    op = (1 / 2.0) * np.conj(z) * (a ** 2) - (1 / 2.0) * z * (a.dag()) ** 2
    return op.expm()


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
    b = 0.5 * (np.conj(z) * (a1 * a2) - z * (a1.dag() * a2.dag()))
    return b.expm()


def displace(N, alpha, offset=0):
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

    Returns
    -------
    oper : qobj
        Displacement operator.

    Examples
    ---------
    >>> displace(4,0.25)
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.96923323+0.j -0.24230859+0.j  0.04282883+0.j -0.00626025+0.j]
     [ 0.24230859+0.j  0.90866411+0.j -0.33183303+0.j  0.07418172+0.j]
     [ 0.04282883+0.j  0.33183303+0.j  0.84809499+0.j -0.41083747+0.j]
     [ 0.00626025+0.j  0.07418172+0.j  0.41083747+0.j  0.90866411+0.j]]

    """
    a = destroy(N, offset=offset)
    D = (alpha * a.dag() - np.conj(alpha) * a).expm()
    return D


def commutator(A, B, kind="normal"):
    """
    Return the commutator of kind `kind` (normal, anti) of the
    two operators A and B.
    """
    if kind == 'normal':
        return A * B - B * A

    elif kind == 'anti':
        return A * B + B * A

    else:
        raise TypeError("Unknown commutator kind '%s'" % kind)


def qutrit_ops():
    """
    Operators for a three level system (qutrit).

    Returns
    -------
    opers: array
        `array` of qutrit operators.

    """
    from qutip.states import qutrit_basis

    one, two, three = qutrit_basis()
    sig11 = one * one.dag()
    sig22 = two * two.dag()
    sig33 = three * three.dag()
    sig12 = one * two.dag()
    sig23 = two * three.dag()
    sig31 = three * one.dag()
    return np.array([sig11, sig22, sig33, sig12, sig23, sig31],
                    dtype=object)


def qdiags(diagonals, offsets, dims=None, shape=None):
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

    See Also
    --------
    scipy.sparse.diags for usage information.

    Notes
    -----
    This function requires SciPy 0.11+.

    Examples
    --------
    >>> qdiags(sqrt(range(1,4)),1)
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isherm = False
    Qobj data =
    [[ 0.          1.          0.          0.        ]
     [ 0.          0.          1.41421356  0.        ]
     [ 0.          0.          0.          1.73205081]
     [ 0.          0.          0.          0.        ]]

    """
    try:
        data = sp.diags(diagonals, offsets, shape, format='csr', dtype=complex)
    except:
        raise NotImplementedError("This function requires SciPy 0.11+.")
    if not dims:
        dims = [[], []]
    if not shape:
        shape = []
    return Qobj(data, dims, list(shape))


def phase(N, phi0=0):
    """
    Single-mode Pegg-Barnett phase operator.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.
    phi0 : float
        Reference phase.

    Returns
    -------
    oper : qobj
        Phase operator with respect to reference phase.

    Notes
    -----
    The Pegg-Barnett phase operator is Hermitian on a truncated Hilbert space.

    """
    phim = phi0 + (2.0 * np.pi * np.arange(N)) / N  # discrete phase angles
    n = np.arange(N).reshape((N, 1))
    states = np.array([np.sqrt(kk) / np.sqrt(N) * np.exp(1.0j * n * kk)
                       for kk in phim])
    ops = np.array([np.outer(st, st.conj()) for st in states])
    return Qobj(np.sum(ops, axis=0))


def qzero(N):
    """
    Zero operator

    Parameters
    ----------
    N : int or list of ints
        Dimension of Hilbert space. If provided as a list of ints,
        then the dimension is the product over this list, but the
        ``dims`` property of the new Qobj are set to this list.

    Returns
    -------
    qzero : qobj
        Zero operator Qobj.

    """

    if isinstance(N, list):
        return tensor.tensor(*[qzero(n) for n in N])
    N = int(N)
    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N>=0")
    return Qobj(sp.csr_matrix((N, N), dtype=complex), isherm=True)


def enr_destroy(dims, excitations):
    """
    Generate annilation operators for modes in a excitation-number-restricted
    state space. For example, consider a system consisting of 4 modes, each
    with 5 states. The total hilbert space size is 5**4 = 625. If we are
    only interested in states that contain up to 2 excitations, we only need
    to include states such as

        (0, 0, 0, 0)
        (0, 0, 0, 1)
        (0, 0, 0, 2)
        (0, 0, 1, 0)
        (0, 0, 1, 1)
        (0, 0, 2, 0)
        ...

    This function creates annihilation operators for the 4 modes that act
    within this state space:

        a1, a2, a3, a4 = enr_destroy([5, 5, 5, 5], excitations=2)

    From this point onwards, the annihiltion operators a1, ..., a4 can be
    used to setup a Hamiltonian, collapse operators and expectation-value
    operators, etc., following the usual pattern.

    Parameters
    ----------
    dims : list
        A list of the dimensions of each subsystem of a composite quantum
        system.

    excitations : integer
        The maximum number of excitations that are to be included in the
        state space.

    Returns
    -------
    a_ops : list of qobj
        A list of annihilation operators for each mode in the composite
        quantum system described by dims.
    """
    from qutip.states import enr_state_dictionaries

    nstates, state2idx, idx2state = enr_state_dictionaries(dims, excitations)

    a_ops = [sp.lil_matrix((nstates, nstates), dtype=np.complex)
             for _ in range(len(dims))]

    for n1, state1 in idx2state.items():
        for n2, state2 in idx2state.items():
            for idx, a in enumerate(a_ops):
                s1 = [s for idx2, s in enumerate(state1) if idx != idx2]
                s2 = [s for idx2, s in enumerate(state2) if idx != idx2]
                if (state1[idx] == state2[idx] - 1) and (s1 == s2):
                    a_ops[idx][n1, n2] = np.sqrt(state2[idx])

    return [Qobj(a, dims=[dims, dims]) for a in a_ops]


def enr_identity(dims, excitations):
    """
    Generate the identity operator for the excitation-number restricted
    state space defined by the `dims` and `exciations` arguments. See the
    docstring for enr_fock for a more detailed description of these arguments.

    Parameters
    ----------
    dims : list
        A list of the dimensions of each subsystem of a composite quantum
        system.

    excitations : integer
        The maximum number of excitations that are to be included in the
        state space.

    state : list of integers
        The state in the number basis representation.

    Returns
    -------
    op : Qobj
        A Qobj instance that represent the identity operator in the
        exication-number-restricted state space defined by `dims` and
        `exciations`.
    """
    from qutip.states import enr_state_dictionaries

    nstates, _, _ = enr_state_dictionaries(dims, excitations)
    data = sp.eye(nstates, nstates, dtype=np.complex)
    return Qobj(data, dims=[dims, dims])



def charge(Nmax, Nmin=None, frac = 1):
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
    
    Returns
    -------
    C : Qobj
        Charge operator over [Nmin,Nmax].
    
    Notes
    -----
    .. versionadded:: 3.2
    
    """
    if Nmin is None:
        Nmin = -Nmax
    diag = np.arange(Nmin, Nmax+1, dtype=float)
    if frac != 1:
        diag *= frac
    C = sp.diags(diag, 0, format='csr', dtype=complex)
    return Qobj(C, isherm=True)



def tunneling(N, m=1):
    """
    Tunneling operator with elements of the form 
    :math:`\sum |N><N+m| + |N+m><N|`.
    
    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.
    m : int (default = 1)
        Number of excitations in tunneling event.
    
    Returns
    -------
    T : Qobj
        Tunneling operator.
    
    Notes
    -----
    .. versionadded:: 3.2
    
    """
    diags = [np.ones(N-m,dtype=int),np.ones(N-m,dtype=int)]
    T = sp.diags(diags,[m,-m],format='csr', dtype=complex)
    return Qobj(T, isherm=True)



# Break circular dependencies by a trailing import.
# Note that we use a relative import here to deal with that
# qutip.tensor is the *function* tensor, not the module.
from . import tensor
