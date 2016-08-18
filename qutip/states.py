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

__all__ = ['basis', 'qutrit_basis', 'coherent', 'coherent_dm', 'fock_dm',
           'fock', 'thermal_dm', 'maximally_mixed_dm', 'ket2dm', 'projection',
           'qstate', 'ket', 'bra', 'state_number_enumerate',
           'state_number_index', 'state_index_number', 'state_number_qobj',
           'phase_basis', 'zero_ket', 'spin_state', 'spin_coherent',
           'bell_state', 'singlet_state', 'triplet_states', 'w_state',
           'ghz_state', 'enr_state_dictionaries', 'enr_fock',
           'enr_thermal_dm']

import numpy as np
from scipy import arange, conj, prod
import scipy.sparse as sp

from qutip.qobj import Qobj
from qutip.operators import destroy, jmat
from qutip.tensor import tensor


def basis(N, n=0, offset=0):
    """Generates the vector representation of a Fock state.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    n : int
        Integer corresponding to desired number state, defaults
        to 0 if omitted.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state.

    Returns
    -------
    state : qobj
      Qobj representing the requested number state ``|n>``.

    Examples
    --------
    >>> basis(5,2)
    Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
    Qobj data =
    [[ 0.+0.j]
     [ 0.+0.j]
     [ 1.+0.j]
     [ 0.+0.j]
     [ 0.+0.j]]

    Notes
    -----

    A subtle incompatibility with the quantum optics toolbox: In QuTiP::

        basis(N, 0) = ground state

    but in the qotoolbox::

        basis(N, 1) = ground state

    """
    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N >= 0")

    if (not isinstance(n, (int, np.integer))) or n < offset:
        raise ValueError("n must be integer n >= 0")

    if n - offset > (N - 1):  # check if n is within bounds
        raise ValueError("basis vector index need to be in n <= N-1")

    bas = sp.lil_matrix((N, 1))  # column vector of zeros
    bas[n - offset, 0] = 1  # 1 located at position n
    bas = bas.tocsr()

    return Qobj(bas)


def qutrit_basis():
    """Basis states for a three level system (qutrit)

    Returns
    -------
    qstates : array
        Array of qutrit basis vectors

    """
    return np.array([basis(3, 0), basis(3, 1), basis(3, 2)], dtype=object)


def _sqrt_factorial(n_vec):
    # take the square root before multiplying
    return np.array([np.prod(np.sqrt(np.arange(1, n + 1))) for n in n_vec])


def coherent(N, alpha, offset=0, method='operator'):
    """Generates a coherent state with eigenvalue alpha.

    Constructed using displacement operator on vacuum state.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue of coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state. Using a non-zero offset will make the
        default method 'analytic'.

    method : string {'operator', 'analytic'}
        Method for generating coherent state.

    Returns
    -------
    state : qobj
        Qobj quantum object for coherent state

    Examples
    --------
    >>> coherent(5,0.25j)
    Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
    Qobj data =
    [[  9.69233235e-01+0.j        ]
     [  0.00000000e+00+0.24230831j]
     [ -4.28344935e-02+0.j        ]
     [  0.00000000e+00-0.00618204j]
     [  7.80904967e-04+0.j        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent state is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting state is normalized. With 'analytic' method the coherent state
    is generated using the analytical formula for the coherent state
    coefficients in the Fock basis. This method does not guarantee that the
    state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """
    if method == "operator" and offset == 0:

        x = basis(N, 0)
        a = destroy(N)
        D = (alpha * a.dag() - conj(alpha) * a).expm()
        return D * x

    elif method == "analytic" or offset > 0:

        data = np.zeros([N, 1], dtype=complex)
        n = arange(N) + offset
        data[:, 0] = np.exp(-(abs(alpha) ** 2) / 2.0) * (alpha ** (n)) / \
            _sqrt_factorial(n)
        return Qobj(data)

    else:
        raise TypeError(
            "The method option can only take values 'operator' or 'analytic'")


def coherent_dm(N, alpha, offset=0, method='operator'):
    """Density matrix representation of a coherent state.

    Constructed via outer product of :func:`qutip.states.coherent`

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue for coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state.

    method : string {'operator', 'analytic'}
        Method for generating coherent density matrix.

    Returns
    -------
    dm : qobj
        Density matrix representation of coherent state.

    Examples
    --------
    >>> coherent_dm(3,0.25j)
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.93941695+0.j          0.00000000-0.23480733j -0.04216943+0.j        ]
     [ 0.00000000+0.23480733j  0.05869011+0.j          0.00000000-0.01054025j]
     [-0.04216943+0.j          0.00000000+0.01054025j  0.00189294+0.j\
        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent density matrix is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting density matrix is normalized. With 'analytic' method the coherent
    density matrix is generated using the analytical formula for the coherent
    state coefficients in the Fock basis. This method does not guarantee that
    the state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """
    if method == "operator":
        psi = coherent(N, alpha, offset=offset)
        return psi * psi.dag()

    elif method == "analytic":
        psi = coherent(N, alpha, offset=offset, method='analytic')
        return psi * psi.dag()

    else:
        raise TypeError(
            "The method option can only take values 'operator' or 'analytic'")


def fock_dm(N, n=0, offset=0):
    """Density matrix representation of a Fock state

    Constructed via outer product of :func:`qutip.states.fock`.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    n : int
        ``int`` for desired number state, defaults to 0 if omitted.

    Returns
    -------
    dm : qobj
        Density matrix representation of Fock state.

    Examples
    --------
     >>> fock_dm(3,1)
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
     Qobj data =
     [[ 0.+0.j  0.+0.j  0.+0.j]
      [ 0.+0.j  1.+0.j  0.+0.j]
      [ 0.+0.j  0.+0.j  0.+0.j]]

    """
    psi = basis(N, n, offset=offset)

    return psi * psi.dag()


def fock(N, n=0, offset=0):
    """Bosonic Fock (number) state.

    Same as :func:`qutip.states.basis`.

    Parameters
    ----------
    N : int
        Number of states in the Hilbert space.

    n : int
        ``int`` for desired number state, defaults to 0 if omitted.

    Returns
    -------
        Requested number state :math:`\\left|n\\right>`.

    Examples
    --------
    >>> fock(4,3)
    Quantum object: dims = [[4], [1]], shape = [4, 1], type = ket
    Qobj data =
    [[ 0.+0.j]
     [ 0.+0.j]
     [ 0.+0.j]
     [ 1.+0.j]]

    """
    return basis(N, n, offset=offset)


def thermal_dm(N, n, method='operator'):
    """Density matrix for a thermal state of n particles

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    n : float
        Expectation value for number of particles in thermal state.

    method : string {'operator', 'analytic'}
        ``string`` that sets the method used to generate the
        thermal state probabilities

    Returns
    -------
    dm : qobj
        Thermal state density matrix.

    Examples
    --------
    >>> thermal_dm(5, 1)
    Quantum object: dims = [[5], [5]], \
shape = [5, 5], type = oper, isHerm = True
    Qobj data =
    [[ 0.51612903  0.          0.          0.          0.        ]
     [ 0.          0.25806452  0.          0.          0.        ]
     [ 0.          0.          0.12903226  0.          0.        ]
     [ 0.          0.          0.          0.06451613  0.        ]
     [ 0.          0.          0.          0.          0.03225806]]


    >>> thermal_dm(5, 1, 'analytic')
    Quantum object: dims = [[5], [5]], \
shape = [5, 5], type = oper, isHerm = True
    Qobj data =
    [[ 0.5      0.       0.       0.       0.     ]
     [ 0.       0.25     0.       0.       0.     ]
     [ 0.       0.       0.125    0.       0.     ]
     [ 0.       0.       0.       0.0625   0.     ]
     [ 0.       0.       0.       0.       0.03125]]

    Notes
    -----
    The 'operator' method (default) generates
    the thermal state using the truncated number operator ``num(N)``. This
    is the method that should be used in computations. The
    'analytic' method uses the analytic coefficients derived in
    an infinite Hilbert space. The analytic form is not necessarily normalized,
    if truncated too aggressively.

    """
    if n == 0:
        return fock_dm(N, 0)
    else:
        i = arange(N)
        if method == 'operator':
            beta = np.log(1.0 / n + 1.0)
            diags = np.exp(-beta * i)
            diags = diags / np.sum(diags)
            # populates diagonal terms using truncated operator expression
            rm = sp.spdiags(diags, 0, N, N, format='csr')
        elif method == 'analytic':
            # populates diagonal terms using analytic values
            rm = sp.spdiags((1.0 + n) ** (-1.0) * (n / (1.0 + n)) ** (i),
                            0, N, N, format='csr')
        else:
            raise ValueError(
                "'method' keyword argument must be 'operator' or 'analytic'")
    return Qobj(rm)


def maximally_mixed_dm(N):
    """
    Returns the maximally mixed density matrix for a Hilbert space of
    dimension N.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    Returns
    -------
    dm : qobj
        Thermal state density matrix.
    """
    if (not isinstance(N, (int, np.int64))) or N <= 0:
        raise ValueError("N must be integer N > 0")

    dm = sp.spdiags(np.ones(N, dtype=complex)/float(N), 0, N, N, format='csr')

    return Qobj(dm, isherm=True)


def ket2dm(Q):
    """Takes input ket or bra vector and returns density matrix
    formed by outer product.

    Parameters
    ----------
    Q : qobj
        Ket or bra type quantum object.

    Returns
    -------
    dm : qobj
        Density matrix formed by outer product of `Q`.

    Examples
    --------
    >>> x=basis(3,2)
    >>> ket2dm(x)
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j]]

    """
    if Q.type == 'ket':
        out = Q * Q.dag()
    elif Q.type == 'bra':
        out = Q.dag() * Q
    else:
        raise TypeError("Input is not a ket or bra vector.")
    return Qobj(out)


#
# projection operator
#
def projection(N, n, m, offset=0):
    """The projection operator that projects state :math:`|m>` on state :math:`|n>`.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    n, m : float
        The number states in the projection.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the projector.

    Returns
    -------
    oper : qobj
         Requested projection operator.
    
    """
    ket1 = basis(N, n, offset=offset)
    ket2 = basis(N, m, offset=offset)

    return ket1 * ket2.dag()


#
# composite qubit states
#
def qstate(string):
    """Creates a tensor product for a set of qubits in either
    the 'up' :math:`|0>` or 'down' :math:`|1>` state.

    Parameters
    ----------
    string : str
        String containing 'u' or 'd' for each qubit (ex. 'ududd')

    Returns
    -------
    qstate : qobj
        Qobj for tensor product corresponding to input string.

    Notes
    -----
    Look at ket and bra for more general functions
    creating multiparticle states.

    Examples
    --------
    >>> qstate('udu')
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = [8, 1], type = ket
    Qobj data =
    [[ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 1.]
     [ 0.]
     [ 0.]]

    """
    n = len(string)
    if n != (string.count('u') + string.count('d')):
        raise TypeError('String input to QSTATE must consist ' +
                        'of "u" and "d" elements only')
    else:
        up = basis(2, 1)
        dn = basis(2, 0)
    lst = []
    for k in range(n):
        if string[k] == 'u':
            lst.append(up)
        else:
            lst.append(dn)
    return tensor(lst)


#
# different qubit notation dictionary
#
_qubit_dict = {'g': 0,  # ground state
               'e': 1,  # excited state
               'u': 0,  # spin up
               'd': 1,  # spin down
               'H': 0,  # horizontal polarization
               'V': 1}  # vertical polarization


def _character_to_qudit(x):
    """
    Converts a character representing a one-particle state into int.
    """
    if x in _qubit_dict:
        return _qubit_dict[x]
    else:
        return int(x)


def ket(seq, dim=2):
    """
    Produces a multiparticle ket state for a list or string,
    where each element stands for state of the respective particle.

    Parameters
    ----------
    seq : str / list of ints or characters
        Each element defines state of the respective particle.
        (e.g. [1,1,0,1] or a string "1101").
        For qubits it is also possible to use the following conventions:
        - 'g'/'e' (ground and excited state)
        - 'u'/'d' (spin up and down)
        - 'H'/'V' (horizontal and vertical polarization)
        Note: for dimension > 9 you need to use a list.


    dim : int (default: 2) / list of ints
        Space dimension for each particle:
        int if there are the same, list if they are different.

    Returns
    -------
    ket : qobj

    Examples
    --------
    >>> ket("10")
    Quantum object: dims = [[2, 2], [1, 1]], shape = [4, 1], type = ket
    Qobj data =
    [[ 0.]
     [ 0.]
     [ 1.]
     [ 0.]]

    >>> ket("Hue")
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = [8, 1], type = ket
    Qobj data =
    [[ 0.]
     [ 1.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]]

    >>> ket("12", 3)
    Quantum object: dims = [[3, 3], [1, 1]], shape = [9, 1], type = ket
    Qobj data =
    [[ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 1.]
     [ 0.]
     [ 0.]
     [ 0.]]

    >>> ket("31", [5, 2])
    Quantum object: dims = [[5, 2], [1, 1]], shape = [10, 1], type = ket
    Qobj data =
    [[ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 1.]
     [ 0.]
     [ 0.]]
    """
    if isinstance(dim, int):
        dim = [dim] * len(seq)
    return tensor([basis(dim[i], _character_to_qudit(x))
                   for i, x in enumerate(seq)])


def bra(seq, dim=2):
    """
    Produces a multiparticle bra state for a list or string,
    where each element stands for state of the respective particle.

    Parameters
    ----------
    seq : str / list of ints or characters
        Each element defines state of the respective particle.
        (e.g. [1,1,0,1] or a string "1101").
        For qubits it is also possible to use the following conventions:
        - 'g'/'e' (ground and excited state)
        - 'u'/'d' (spin up and down)
        - 'H'/'V' (horizontal and vertical polarization)
        Note: for dimension > 9 you need to use a list.


    dim : int (default: 2) / list of ints
        Space dimension for each particle:
        int if there are the same, list if they are different.

    Returns
    -------
    bra : qobj

    Examples
    --------
    >>> bra("10")
    Quantum object: dims = [[1, 1], [2, 2]], shape = [1, 4], type = bra
    Qobj data =
    [[ 0.  0.  1.  0.]]

    >>> bra("Hue")
    Quantum object: dims = [[1, 1, 1], [2, 2, 2]], shape = [1, 8], type = bra
    Qobj data =
    [[ 0.  1.  0.  0.  0.  0.  0.  0.]]

    >>> bra("12", 3)
    Quantum object: dims = [[1, 1], [3, 3]], shape = [1, 9], type = bra
    Qobj data =
    [[ 0.  0.  0.  0.  0.  1.  0.  0.  0.]]


    >>> bra("31", [5, 2])
    Quantum object: dims = [[1, 1], [5, 2]], shape = [1, 10], type = bra
    Qobj data =
    [[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]]
    """
    return ket(seq, dim=dim).dag()


#
# quantum state number helper functions
#
def state_number_enumerate(dims, excitations=None, state=None, idx=0):
    """
    An iterator that enumerate all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.

    Example:

        >>> for state in state_number_enumerate([2,2]):
        >>>     print(state)
        [ 0  0 ]
        [ 0  1 ]
        [ 1  0 ]
        [ 1  1 ]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        Current state in the iteration. Used internally.

    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.

    idx : integer
        Current index in the iteration. Used internally.

    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.

    """

    if state is None:
        state = np.zeros(len(dims), dtype=int)

    if excitations and sum(state[0:idx]) > excitations:
        pass
    elif idx == len(dims):
        if excitations is None:
            yield np.array(state)
        else:
            yield tuple(state)
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, excitations, state, idx + 1):
                yield s


def state_number_index(dims, state):
    """
    Return the index of a quantum state corresponding to state,
    given a system with dimensions given by dims.

    Example:

        >>> state_number_index([2, 2, 2], [1, 1, 0])
        6

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        State number array.

    Returns
    -------
    idx : int
        The index of the state given by `state` in standard enumeration
        ordering.

    """
    return int(
        sum([state[i] * prod(dims[i + 1:]) for i, d in enumerate(dims)]))


def state_index_number(dims, index):
    """
    Return a quantum number representation given a state index, for a system
    of composite structure defined by dims.

    Example:

        >>> state_index_number([2, 2, 2], 6)
        [1, 1, 0]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    index : integer
        The index of the state in standard enumeration ordering.

    Returns
    -------
    state : list
        The state number array corresponding to index `index` in standard
        enumeration ordering.

    """
    state = np.empty_like(dims)

    D = np.concatenate([np.flipud(np.cumprod(np.flipud(dims[1:]))), [1]])

    for n in range(len(dims)):
        state[n] = index / D[n]
        index -= state[n] * D[n]

    return list(state)


def state_number_qobj(dims, state):
    """
    Return a Qobj representation of a quantum state specified by the state
    array `state`.

    Example:

        >>> state_number_qobj([2, 2, 2], [1, 0, 1])
        Quantum object: dims = [[2, 2, 2], [1, 1, 1]], \
shape = [8, 1], type = ket
        Qobj data =
        [[ 0.]
         [ 0.]
         [ 0.]
         [ 0.]
         [ 0.]
         [ 1.]
         [ 0.]
         [ 0.]]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        State number array.

    Returns
    -------
    state : :class:`qutip.Qobj.qobj`
        The state as a :class:`qutip.Qobj.qobj` instance.


    """
    return tensor([fock(dims[i], s) for i, s in enumerate(state)])


#
# Excitation-number restricted (enr) states
#
def enr_state_dictionaries(dims, excitations):
    """
    Return the number of states, and lookup-dictionaries for translating
    a state tuple to a state index, and vice versa, for a system with a given
    number of components and maximum number of excitations.

    Parameters
    ----------
    dims: list
        A list with the number of states in each sub-system.

    excitations : integer
        The maximum numbers of dimension

    Returns
    -------
    nstates, state2idx, idx2state: integer, dict, dict
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a dictionary for looking up state
        state tuples from state indices.
    """
    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in state_number_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1

    return nstates, state2idx, idx2state


def enr_fock(dims, excitations, state):
    """
    Generate the Fock state representation in a excitation-number restricted
    state space. The `dims` argument is a list of integers that define the
    number of quantums states of each component of a composite quantum system,
    and the `excitations` specifies the maximum number of excitations for
    the basis states that are to be included in the state space. The `state`
    argument is a tuple of integers that specifies the state (in the number
    basis representation) for which to generate the Fock state representation.

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
    ket : Qobj
        A Qobj instance that represent a Fock state in the exication-number-
        restricted state space defined by `dims` and `exciations`.

    """
    nstates, state2idx, idx2state = enr_state_dictionaries(dims, excitations)

    data = sp.lil_matrix((nstates, 1), dtype=np.complex)

    try:
        data[state2idx[tuple(state)], 0] = 1
    except:
        raise ValueError("The state tuple %s is not in the restricted "
                         "state space" % str(tuple(state)))

    return Qobj(data, dims=[dims, 1])


def enr_thermal_dm(dims, excitations, n):
    """
    Generate the density operator for a thermal state in the excitation-number-
    restricted state space defined by the `dims` and `exciations` arguments.
    See the documentation for enr_fock for a more detailed description of
    these arguments. The temperature of each mode in dims is specified by
    the average number of excitatons `n`.

    Parameters
    ----------
    dims : list
        A list of the dimensions of each subsystem of a composite quantum
        system.

    excitations : integer
        The maximum number of excitations that are to be included in the
        state space.

    n : integer
        The average number of exciations in the thermal state. `n` can be
        a float (which then applies to each mode), or a list/array of the same
        length as dims, in which each element corresponds specifies the
        temperature of the corresponding mode.

    Returns
    -------
    dm : Qobj
        Thermal state density matrix.
    """
    nstates, state2idx, idx2state = enr_state_dictionaries(dims, excitations)

    if not isinstance(n, (list, np.ndarray)):
        n = np.ones(len(dims)) * n
    else:
        n = np.asarray(n)

    diags = [np.prod((n / (n + 1)) ** np.array(state))
             for idx, state in idx2state.items()]
    diags /= np.sum(diags)
    data = sp.spdiags(diags, 0, nstates, nstates, format='csr')

    return Qobj(data, dims=[dims, dims])


def phase_basis(N, m, phi0=0):
    """
    Basis vector for the mth phase of the Pegg-Barnett phase operator.

    Parameters
    ----------
    N : int
        Number of basis vectors in Hilbert space.
    m : int
        Integer corresponding to the mth discrete phase phi_m=phi0+2*pi*m/N
    phi0 : float (default=0)
        Reference phase angle.

    Returns
    -------
    state : qobj
        Ket vector for mth Pegg-Barnett phase operator basis state.

    Notes
    -----
    The Pegg-Barnett basis states form a complete set over the truncated
    Hilbert space.

    """
    phim = phi0 + (2.0 * np.pi * m) / N
    n = np.arange(N).reshape((N, 1))
    data = 1.0 / np.sqrt(N) * np.exp(1.0j * n * phim)
    return Qobj(data)


def zero_ket(N, dims=None):
    """
    Creates the zero ket vector with shape Nx1 and
    dimensions `dims`.

    Parameters
    ----------
    N : int
        Hilbert space dimensionality
    dims : list
        Optional dimensions if ket corresponds to
        a composite Hilbert space.

    Returns
    -------
    zero_ket : qobj
        Zero ket on given Hilbert space.

    """
    return Qobj(sp.csr_matrix((N, 1), dtype=complex), dims=dims)


def spin_state(j, m, type='ket'):
    """Generates the spin state |j, m>, i.e.  the eigenstate
    of the spin-j Sz operator with eigenvalue m.

    Parameters
    ----------
    j : float
        The spin of the state ().

    m : int
        Eigenvalue of the spin-j Sz operator.

    type : string {'ket', 'bra', 'dm'}
        Type of state to generate.

    Returns
    -------
    state : qobj
        Qobj quantum object for spin state

    """
    J = 2 * j + 1

    if type == 'ket':
        return basis(int(J), int(j - m))
    elif type == 'bra':
        return basis(int(J), int(j - m)).dag()
    elif type == 'dm':
        return fock_dm(int(J), int(j - m))
    else:
        raise ValueError("invalid value keyword argument 'type'")


def spin_coherent(j, theta, phi, type='ket'):
    """Generates the spin state |j, m>, i.e.  the eigenstate
    of the spin-j Sz operator with eigenvalue m.

    Parameters
    ----------
    j : float
        The spin of the state.

    theta : float
        Angle from z axis.

    phi : float
        Angle from x axis.

    type : string {'ket', 'bra', 'dm'}
        Type of state to generate.

    Returns
    -------
    state : qobj
        Qobj quantum object for spin coherent state

    """
    Sp = jmat(j, '+')
    Sm = jmat(j, '-')
    psi = (0.5 * theta * np.exp(1j * phi) * Sm -
           0.5 * theta * np.exp(-1j * phi) * Sp).expm() * spin_state(j, j)

    if type == 'ket':
        return psi
    elif type == 'bra':
        return psi.dag()
    elif type == 'dm':
        return ket2dm(psi)
    else:
        raise ValueError("invalid value keyword argument 'type'")


def bell_state(state='00'):
    """
    Returns the Bell state:

        |B00> = 1 / sqrt(2)*[|0>|0>+|1>|1>]
        |B01> = 1 / sqrt(2)*[|0>|0>-|1>|1>]
        |B10> = 1 / sqrt(2)*[|0>|1>+|1>|0>]
        |B11> = 1 / sqrt(2)*[|0>|1>-|1>|0>]

    Returns
    -------
    Bell_state : qobj
        Bell state

    """
    if state == '00':
        Bell_state = tensor(
            basis(2), basis(2))+tensor(basis(2, 1), basis(2, 1))
    elif state == '01':
        Bell_state = tensor(
            basis(2), basis(2))-tensor(basis(2, 1), basis(2, 1))
    elif state == '10':
        Bell_state = tensor(
            basis(2), basis(2, 1))+tensor(basis(2, 1), basis(2))
    elif state == '11':
        Bell_state = tensor(
            basis(2), basis(2, 1))-tensor(basis(2, 1), basis(2))

    return Bell_state.unit()


def singlet_state():
    """
    Returns the two particle singlet-state:

        |S>=1/sqrt(2)*[|0>|1>-|1>|0>]

    that is identical to the fourth bell state.

    Returns
    -------
    Bell_state : qobj
        |B11> Bell state

    """
    return bell_state('11')


def triplet_states():
    """
    Returns the two particle triplet-states:

        |T>= |1>|1>
           = 1 / sqrt(2)*[|0>|1>-|1>|0>]
           = |0>|0>
    that is identical to the fourth bell state.

    Returns
    -------
    trip_states : list
        2 particle triplet states

    """
    trip_states = []
    trip_states.append(tensor(basis(2, 1), basis(2, 1)))
    trip_states.append(
       (tensor(basis(2), basis(2, 1)) + tensor(basis(2, 1), basis(2))).unit()
    )
    trip_states.append(tensor(basis(2), basis(2)))
    return trip_states


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
    inds = np.zeros(N, dtype=int)
    inds[0] = 1
    state = tensor([basis(2, x) for x in inds])
    for kk in range(1, N):
        perm_inds = np.roll(inds, kk)
        state += tensor([basis(2, x) for x in perm_inds])
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
    state = (tensor([basis(2) for k in range(N)]) +
             tensor([basis(2, 1) for k in range(N)]))
    return state/np.sqrt(2)
