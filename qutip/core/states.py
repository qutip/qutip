__all__ = ['basis', 'qutrit_basis', 'coherent', 'coherent_dm', 'fock_dm',
           'fock', 'thermal_dm', 'maximally_mixed_dm', 'ket2dm', 'projection',
           'qstate', 'ket', 'bra', 'state_number_enumerate',
           'state_number_index', 'state_index_number', 'state_number_qobj',
           'phase_basis', 'zero_ket', 'spin_state', 'spin_coherent',
           'bell_state', 'singlet_state', 'triplet_states', 'w_state',
           'ghz_state', 'enr_state_dictionaries', 'enr_fock',
           'enr_thermal_dm']

import itertools
import numbers
import warnings

import numpy as np
import scipy.sparse as sp
import itertools

from . import data as _data
from .qobj import Qobj
from .operators import jmat, displace, qdiags
from .tensor import tensor
from .. import settings

def _promote_to_zero_list(arg, length):
    """
    Ensure `arg` is a list of length `length`.  If `arg` is None it is promoted
    to `[0]*length`.  All other inputs are checked that they match the correct
    form.

    Returns
    -------
    list_ : list
        A list of integers of length `length`.
    """
    if arg is None:
        arg = [0]*length
    elif not isinstance(arg, list):
        arg = [arg]
    if not len(arg) == length:
        raise ValueError("All list inputs must be the same length.")
    if all(isinstance(x, numbers.Integral) for x in arg):
        return arg
    raise TypeError("Dimensions must be an integer or list of integers.")


def basis(dimensions, n=None, offset=None, *, dtype=None):
    """Generates the vector representation of a Fock state.

    Parameters
    ----------
    dimensions : int or list of ints
        Number of Fock states in Hilbert space.  If a list, then the resultant
        object will be a tensor product over spaces with those dimensions.

    n : int or list of ints, optional (default 0 for all dimensions)
        Integer corresponding to desired number state, defaults to 0 for all
        dimensions if omitted.  The shape must match ``dimensions``, e.g. if
        ``dimensions`` is a list, then ``n`` must either be omitted or a list
        of equal length.

    offset : int or list of ints, optional (default 0 for all dimensions)
        The lowest number state that is included in the finite number state
        representation of the state in the relevant dimension.

    dtype : type or str
        storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state : :class:`qutip.Qobj`
      Qobj representing the requested number state ``|n>``.

    Examples
    --------
    >>> basis(5,2) # doctest: +SKIP
    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[ 0.+0.j]
     [ 0.+0.j]
     [ 1.+0.j]
     [ 0.+0.j]
     [ 0.+0.j]]
    >>> basis([2,2,2], [0,1,0]) # doctest: +SKIP
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
    Qobj data =
    [[0.]
     [0.]
     [1.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]]


    Notes
    -----
    A subtle incompatibility with the quantum optics toolbox: In QuTiP::

        basis(N, 0) = ground state

    but in the qotoolbox::

        basis(N, 1) = ground state

    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    # Promote all parameters to lists to simplify later logic.
    if not isinstance(dimensions, list):
        dimensions = [dimensions]
    n_dimensions = len(dimensions)
    ns = [m-off for m, off in zip(_promote_to_zero_list(n, n_dimensions),
                                  _promote_to_zero_list(offset, n_dimensions))]
    if any((not isinstance(x, numbers.Integral)) or x < 0 for x in dimensions):
        raise ValueError("All dimensions must be >= 0.")
    if not all(0 <= n < dimension for n, dimension in zip(ns, dimensions)):
        raise ValueError("All basis indices must be "
                         "`offset <= n < dimension+offset`.")
    location, size = 0, 1
    for m, dimension in zip(reversed(ns), reversed(dimensions)):
        location += m*size
        size *= dimension

    data = _data.one_element[dtype]((size, 1), (location, 0), 1)
    return Qobj(data,
                dims=[dimensions, [1]*n_dimensions],
                type='ket',
                isherm=False,
                isunitary=False,
                copy=False)


def qutrit_basis(*, dtype=None):
    """Basis states for a three level system (qutrit)

    dtype : type or str
        storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    qstates : array
        Array of qutrit basis vectors

    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    out = np.empty((3,), dtype=object)
    out[:] = [
        basis(3, 0, dtype=dtype),
        basis(3, 1, dtype=dtype),
        basis(3, 2, dtype=dtype),
    ]
    return out

_COHERENT_METHODS = ('operator', 'analytic')


def coherent(N, alpha, offset=0, method=None, *, dtype=None):
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

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state : qobj
        Qobj quantum object for coherent state

    Examples
    --------
    >>> coherent(5,0.25j) # doctest: +SKIP
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
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    if offset < 0:
        raise ValueError('Offset must be non-negative')

    if method is None:
        method = "operator" if offset == 0 else "analytic"

    if method == "operator":
        if offset != 0:
            raise ValueError(
                "The method 'operator' does not support offset != 0. Please"
                " select another method or set the offset to zero."
            )
        return (displace(N, alpha, dtype=dtype) * basis(N, 0)).to(dtype)

    elif method == "analytic":
        sqrtn = np.sqrt(np.arange(offset, offset+N, dtype=complex))
        sqrtn[0] = 1  # Get rid of divide by zero warning
        data = alpha / sqrtn
        if offset == 0:
            data[0] = np.exp(-abs(alpha)**2 / 2.0)
        else:
            s = np.prod(np.sqrt(np.arange(1, offset + 1)))  # sqrt factorial
            data[0] = np.exp(-abs(alpha)**2 * 0.5) * alpha**offset / s
        np.cumprod(data, out=sqrtn)  # Reuse sqrtn array
        return Qobj(sqrtn,
                    dims=[[N], [1]],
                    type='ket',
                    copy=False).to(dtype)
    raise TypeError(
        "The method option can only take values in " + repr(_COHERENT_METHODS)
    )


def coherent_dm(N, alpha, offset=0, method='operator', *, dtype=None):
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

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    dm : qobj
        Density matrix representation of coherent state.

    Examples
    --------
    >>> coherent_dm(3,0.25j) # doctest: +SKIP
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
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    return coherent(N, alpha, offset=offset, method=method, dtype=dtype).proj()


def fock_dm(dimensions, n=None, offset=None, *, dtype=None):
    """Density matrix representation of a Fock state

    Constructed via outer product of :func:`qutip.states.fock`.

    Parameters
    ----------
    dimensions : int or list of ints
        Number of Fock states in Hilbert space.  If a list, then the resultant
        object will be a tensor product over spaces with those dimensions.

    n : int or list of ints, optional (default 0 for all dimensions)
        Integer corresponding to desired number state, defaults to 0 for all
        dimensions if omitted.  The shape must match ``dimensions``, e.g. if
        ``dimensions`` is a list, then ``n`` must either be omitted or a list
        of equal length.

    offset : int or list of ints, optional (default 0 for all dimensions)
        The lowest number state that is included in the finite number state
        representation of the state in the relevant dimension.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    dm : qobj
        Density matrix representation of Fock state.

    Examples
    --------
     >>> fock_dm(3,1) # doctest: +SKIP
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
     Qobj data =
     [[ 0.+0.j  0.+0.j  0.+0.j]
      [ 0.+0.j  1.+0.j  0.+0.j]
      [ 0.+0.j  0.+0.j  0.+0.j]]

    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dia
    return basis(dimensions, n, offset=offset, dtype=dtype).proj()


def fock(dimensions, n=None, offset=None, *, dtype=None):
    """Bosonic Fock (number) state.

    Same as :func:`qutip.states.basis`.

    Parameters
    ----------
    dimensions : int or list of ints
        Number of Fock states in Hilbert space.  If a list, then the resultant
        object will be a tensor product over spaces with those dimensions.

    n : int or list of ints, optional (default 0 for all dimensions)
        Integer corresponding to desired number state, defaults to 0 for all
        dimensions if omitted.  The shape must match ``dimensions``, e.g. if
        ``dimensions`` is a list, then ``n`` must either be omitted or a list
        of equal length.

    offset : int or list of ints, optional (default 0 for all dimensions)
        The lowest number state that is included in the finite number state
        representation of the state in the relevant dimension.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
        Requested number state :math:`\\left|n\\right>`.

    Examples
    --------
    >>> fock(4,3) # doctest: +SKIP
    Quantum object: dims = [[4], [1]], shape = [4, 1], type = ket
    Qobj data =
    [[ 0.+0.j]
     [ 0.+0.j]
     [ 0.+0.j]
     [ 1.+0.j]]

    """
    return basis(dimensions, n, offset=offset, dtype=dtype)


def thermal_dm(N, n, method='operator', *, dtype=None):
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

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    dm : qobj
        Thermal state density matrix.

    Examples
    --------
    >>> thermal_dm(5, 1) # doctest: +SKIP
    Quantum object: dims = [[5], [5]], \
shape = [5, 5], type = oper, isHerm = True
    Qobj data =
    [[ 0.51612903  0.          0.          0.          0.        ]
     [ 0.          0.25806452  0.          0.          0.        ]
     [ 0.          0.          0.12903226  0.          0.        ]
     [ 0.          0.          0.          0.06451613  0.        ]
     [ 0.          0.          0.          0.          0.03225806]]


    >>> thermal_dm(5, 1, 'analytic') # doctest: +SKIP
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
    dtype = dtype or settings.core["default_dtype"] or _data.Dia
    if n == 0:
        return fock_dm(N, 0, dtype=dtype)
    else:
        i = np.arange(N)
        if method == 'operator':
            beta = np.log(1.0 / n + 1.0)
            diags = np.exp(-beta * i)
            diags = diags / np.sum(diags)
            # populates diagonal terms using truncated operator expression

        elif method == 'analytic':
            # populates diagonal terms using analytic values
            diags = (1.0 + n) ** (-1.0) * (n / (1.0 + n)) ** (i)
        else:
            raise ValueError("The method option can only take "
                             "values 'operator' or 'analytic'")
        out = qdiags(diags, 0, dims=[[N], [N]], shape=(N, N), dtype=dtype)
        out._isherm = True
        return out


def maximally_mixed_dm(N, *, dtype=None):
    """
    Returns the maximally mixed density matrix for a Hilbert space of
    dimension N.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    dm : qobj
        Thermal state density matrix.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dia
    if not isinstance(N, numbers.Integral) or N <= 0:
        raise ValueError("N must be integer N > 0")
    return Qobj(_data.identity[dtype](N, scale=1/N), dims=[[N], [N]],
                type='oper', isherm=True, isunitary=(N == 1), copy=False)


def ket2dm(Q):
    """
    Takes input ket or bra vector and returns density matrix formed by outer
    product.  This is completely identical to calling `Q.proj()`.

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
    >>> ket2dm(x) # doctest: +SKIP
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j]]

    """
    if Q.isket or Q.isbra:
        return Q.proj()
    raise TypeError("Input is not a ket or bra vector.")


def projection(N, n, m, offset=None, *, dtype=None):
    r"""
    The projection operator that projects state :math:`\lvert m\rangle` on
    state :math:`\lvert n\rangle`.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    n, m : float
        The number states in the projection.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the projector.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
         Requested projection operator.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.CSR
    return basis(N, n, offset=offset, dtype=dtype) @ \
           basis(N, m, offset=offset, dtype=dtype).dag()


def qstate(string, *, dtype=None):
    r"""Creates a tensor product for a set of qubits in either
    the 'up' :math:`\lvert0\rangle` or 'down' :math:`\lvert1\rangle` state.

    Parameters
    ----------
    string : str
        String containing 'u' or 'd' for each qubit (ex. 'ududd')

    Returns
    -------
    qstate : qobj
        Qobj for tensor product corresponding to input string.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Notes
    -----
    Look at ket and bra for more general functions
    creating multiparticle states.

    Examples
    --------
    >>> qstate('udu') # doctest: +SKIP
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
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    n = len(string)
    if n != (string.count('u') + string.count('d')):
        raise TypeError('String input to QSTATE must consist ' +
                        'of "u" and "d" elements only')
    return basis([2]*n, [1 if x == 'u' else 0 for x in string], dtype=dtype)


#
# different qubit notation dictionary
#
_qubit_dict = {
    'g': 0,  # ground state
    'e': 1,  # excited state
    'u': 0,  # spin up
    'd': 1,  # spin down
    'H': 0,  # horizontal polarization
    'V': 1,  # vertical polarization
}


def _character_to_qudit(x):
    """
    Converts a character representing a one-particle state into int.
    """
    return _qubit_dict[x] if x in _qubit_dict else int(x)


def ket(seq, dim=2, *, dtype=None):
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

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    ket : qobj

    Examples
    --------
    >>> ket("10") # doctest: +SKIP
    Quantum object: dims = [[2, 2], [1, 1]], shape = [4, 1], type = ket
    Qobj data =
    [[ 0.]
     [ 0.]
     [ 1.]
     [ 0.]]

    >>> ket("Hue") # doctest: +SKIP
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

    >>> ket("12", 3) # doctest: +SKIP
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

    >>> ket("31", [5, 2]) # doctest: +SKIP
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
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    ns = [_character_to_qudit(x) for x in seq]
    dim = [dim]*len(ns) if isinstance(dim, numbers.Integral) else dim
    return basis(dim, ns, dtype=dtype)


def bra(seq, dim=2, *, dtype=None):
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

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    bra : qobj

    Examples
    --------
    >>> bra("10") # doctest: +SKIP
    Quantum object: dims = [[1, 1], [2, 2]], shape = [1, 4], type = bra
    Qobj data =
    [[ 0.  0.  1.  0.]]

    >>> bra("Hue") # doctest: +SKIP
    Quantum object: dims = [[1, 1, 1], [2, 2, 2]], shape = [1, 8], type = bra
    Qobj data =
    [[ 0.  1.  0.  0.  0.  0.  0.  0.]]

    >>> bra("12", 3) # doctest: +SKIP
    Quantum object: dims = [[1, 1], [3, 3]], shape = [1, 9], type = bra
    Qobj data =
    [[ 0.  0.  0.  0.  0.  1.  0.  0.  0.]]


    >>> bra("31", [5, 2]) # doctest: +SKIP
    Quantum object: dims = [[1, 1], [5, 2]], shape = [1, 10], type = bra
    Qobj data =
    [[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]]
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    return ket(seq, dim=dim, dtype=dtype).dag()


def state_number_enumerate(dims, excitations=None):
    """
    An iterator that enumerates all the state number tuples (quantum numbers of
    the form (n1, n2, n3, ...)) for a system with dimensions given by dims.

    Example:

        >>> for state in state_number_enumerate([2,2]): # doctest: +SKIP
        >>>     print(state) # doctest: +SKIP
        ( 0  0 )
        ( 0  1 )
        ( 1  0 )
        ( 1  1 )

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.

    Returns
    -------
    state_number : tuple
        Successive state number tuples that can be used in loops and other
        iterations, using standard state enumeration *by definition*.

    """

    if excitations is None:
        # in this case, state numbers are a direct product
        yield from itertools.product(*(range(d) for d in dims))
        return

    # From here on, excitations is not None

    # General idea of algorithm: add excitations one by one in last mode (idx =
    # len(dims)-1), and carry over to the next index when the limit is reached.
    # Keep track of the number of excitations while doing so to avoid having to
    # do explicit sums over the states.
    state = (0,)*len(dims)
    nexc = 0
    while True:
        yield state
        idx = len(dims) - 1
        state = state[:idx] + (state[idx]+1,)
        nexc += 1
        while nexc > excitations or state[idx] >= dims[idx]:
            # remove all excitations in mode idx, add one in idx-1
            idx -= 1
            if idx < 0:
                return
            nexc -= state[idx+1] - 1
            state = state[:idx] + (state[idx]+1, 0) + state[idx+2:]


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
    return np.ravel_multi_index(state, dims)


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
    state : tuple
        The state number tuple corresponding to index `index` in standard
        enumeration ordering.

    """
    return np.unravel_index(index, dims)


def state_number_qobj(dims, state, *, dtype=None):
    """
    Return a Qobj representation of a quantum state specified by the state
    array `state`.

    Example:

        >>> state_number_qobj([2, 2, 2], [1, 0, 1]) # doctest: +SKIP
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

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state : :class:`qutip.Qobj`
        The state as a :class:`qutip.Qobj` instance.

    .. note::
        Deprecated in QuTiP 5.0, use :func:`basis` instead.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    warnings.warn("basis() is a drop-in replacement for this",
                  DeprecationWarning)
    return basis(dims, state, dtype=dtype)


# Excitation-number restricted (enr) states

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
    nstates, state2idx, idx2state: integer, dict, list
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a list containing the state tuples
        ordered by state indices. state2idx and idx2state are reverses of
        each other, i.e., state2idx[idx2state[idx]] = idx and
        idx2state[state2idx[state]] = state.
    """
    idx2state = list(state_number_enumerate(dims, excitations))
    state2idx = {state: idx for idx, state in enumerate(idx2state)}
    nstates = len(idx2state)

    return nstates, state2idx, idx2state


def enr_fock(dims, excitations, state, *, dtype=None):
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

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    ket : Qobj
        A Qobj instance that represent a Fock state in the exication-number-
        restricted state space defined by `dims` and `exciations`.

    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    nstates, state2idx, _ = enr_state_dictionaries(dims, excitations)
    try:
        data =_data.one_element[dtype]((nstates, 1),
                                       (state2idx[tuple(state)], 0), 1)
    except KeyError:
        msg = (
            "The state tuple " + str(tuple(state))
            + " is not in the restricted state space."
        )
        raise ValueError(msg) from None
    return Qobj(data, dims=[dims, [1]*len(dims)], type='ket', copy=False)


def enr_thermal_dm(dims, excitations, n, *, dtype=None):
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

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    dm : Qobj
        Thermal state density matrix.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.CSR
    nstates, _, idx2state = enr_state_dictionaries(dims, excitations)
    if not isinstance(n, (list, np.ndarray)):
        n = np.ones(len(dims)) * n
    else:
        n = np.asarray(n)

    diags = [np.prod((n / (n + 1)) ** np.array(state)) for state in idx2state]
    diags /= np.sum(diags)
    out = qdiags(diags, 0, dims=[dims, dims],
                 shape=(nstates, nstates), dtype=dtype)
    out._isherm = True
    return out


def phase_basis(N, m, phi0=0, *, dtype=None):
    """
    Basis vector for the mth phase of the Pegg-Barnett phase operator.

    Parameters
    ----------
    N : int
        Number of basis vectors in Hilbert space.

    m : int
        Integer corresponding to the mth discrete phase
        phi_m = phi0 + 2 * pi * m / N

    phi0 : float (default=0)
        Reference phase angle.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state : qobj
        Ket vector for mth Pegg-Barnett phase operator basis state.

    Notes
    -----
    The Pegg-Barnett basis states form a complete set over the truncated
    Hilbert space.

    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    phim = phi0 + (2.0 * np.pi * m) / N
    n = np.arange(N)[:, np.newaxis]
    data = np.exp(1.0j * n * phim) / np.sqrt(N)
    return Qobj(data, dims=[[N], [1]], type='ket', copy=False).to(dtype)


def zero_ket(N, dims=None, *, dtype=None):
    """
    Creates the zero ket vector with shape Nx1 and dimensions `dims`.

    Parameters
    ----------
    N : int
        Hilbert space dimensionality
    dims : list
        Optional dimensions if ket corresponds to
        a composite Hilbert space.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    zero_ket : qobj
        Zero ket on given Hilbert space.

    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    return Qobj(_data.zeros[dtype](N, 1), dims=dims, type='ket', copy=False)


def spin_state(j, m, type='ket', *, dtype=None):
    r"""Generates the spin state :math:`\lvert j, m\rangle`, i.e. the
    eigenstate of the spin-j Sz operator with eigenvalue m.

    Parameters
    ----------
    j : float
        The spin of the state ().

    m : int
        Eigenvalue of the spin-j Sz operator.

    type : string {'ket', 'bra', 'dm'}
        Type of state to generate.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state : qobj
        Qobj quantum object for spin state
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    J = 2*j + 1

    if type == 'ket':
        return basis(int(J), int(j - m), dtype=dtype)
    elif type == 'bra':
        return basis(int(J), int(j - m), dtype=dtype).dag()
    elif type == 'dm':
        return fock_dm(int(J), int(j - m), dtype=dtype)
    else:
        raise ValueError(f"Invalid value keyword argument type='{type}'")


def spin_coherent(j, theta, phi, type='ket', *, dtype=None):
    r"""Generate the coherent spin state :math:`\lvert \theta, \phi\rangle`.

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

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state : qobj
        Qobj quantum object for spin coherent state
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    if type not in ['ket', 'bra', 'dm']:
        raise ValueError("Invalid value keyword argument 'type'")
    Sp = jmat(j, '+')
    Sm = jmat(j, '-')
    psi = (0.5 * theta * np.exp(1j * phi) * Sm -
           0.5 * theta * np.exp(-1j * phi) * Sp).expm() * \
           spin_state(j, j)

    if type == 'bra':
        psi = psi.dag()
    elif type == 'dm':
        psi = ket2dm(psi)
    return psi.to(dtype)


_BELL_STATES = {
    '00': np.sqrt(0.5) * (basis([2, 2], [0, 0]) + basis([2, 2], [1, 1])),
    '01': np.sqrt(0.5) * (basis([2, 2], [0, 0]) - basis([2, 2], [1, 1])),
    '10': np.sqrt(0.5) * (basis([2, 2], [0, 1]) + basis([2, 2], [1, 0])),
    '11': np.sqrt(0.5) * (basis([2, 2], [0, 1]) - basis([2, 2], [1, 0])),
}

def bell_state(state='00', *, dtype=None):
    r"""
    Returns the selected Bell state:

    .. math::

        \begin{aligned}
        \lvert B_{00}\rangle &=
            \frac1{\sqrt2}(\lvert00\rangle+\lvert11\rangle)\\
        \lvert B_{01}\rangle &=
            \frac1{\sqrt2}(\lvert00\rangle-\lvert11\rangle)\\
        \lvert B_{10}\rangle &=
            \frac1{\sqrt2}(\lvert01\rangle+\lvert10\rangle)\\
        \lvert B_{11}\rangle &=
            \frac1{\sqrt2}(\lvert01\rangle-\lvert10\rangle)\\
        \end{aligned}


    Parameters
    ----------
    state : str ['00', '01', `10`, `11`]
        Which bell state to return

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    Bell_state : qobj
        Bell state
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    return _BELL_STATES[state].copy().to(dtype)


def singlet_state(*, dtype=None):
    r"""
    Returns the two particle singlet-state:

    .. math::

        \lvert S\rangle = \frac1{\sqrt2}(\lvert01\rangle-\lvert10\rangle)

    that is identical to the fourth bell state.

    Parameters
    ----------
    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    Bell_state : qobj
        :math:`\lvert B_{11}\rangle` Bell state
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    return bell_state('11').to(dtype)


def triplet_states(*, dtype=None):
    r"""
    Returns a list of the two particle triplet-states:

    .. math::

        \lvert T_1\rangle = \lvert11\rangle
        \lvert T_2\rangle = \frac1{\sqrt2}(\lvert01\rangle + \lvert10\rangle)
        \lvert T_3\rangle = \lvert00\rangle

    Parameters
    ----------
    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    trip_states : list
        2 particle triplet states
    """
    return [
        basis([2, 2], [1, 1], dtype=dtype),
        np.sqrt(0.5) * (basis([2, 2], [0, 1], dtype=dtype) +
                        basis([2, 2], [1, 0], dtype=dtype)),
        basis([2, 2], [0, 0], dtype=dtype),
    ]


def w_state(N=3, *, dtype=None):
    """
    Returns the N-qubit W-state:
        ``[ |100..0> + |010..0> + |001..0> + ... |000..1> ] / sqrt(n)``

    Parameters
    ----------
    N : int (default=3)
        Number of qubits in state

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    W : :obj:`~Qobj`
        N-qubit W-state
    """
    inds = np.zeros(N, dtype=int)
    inds[0] = 1
    state = basis([2]*N, list(inds), dtype=dtype)
    for kk in range(1, N):
        state += basis([2]*N, list(np.roll(inds, kk)), dtype=dtype)
    return np.sqrt(1 / N) * state


def ghz_state(N=3, *, dtype=None):
    """
    Returns the N-qubit GHZ-state:
        ``[ |00...00> + |11...11> ] / sqrt(2)``

    Parameters
    ----------
    N : int (default=3)
        Number of qubits in state

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    G : qobj
        N-qubit GHZ-state
    """
    return np.sqrt(0.5) * (basis([2]*N, [0]*N, dtype=dtype) +
                           basis([2]*N, [1]*N, dtype=dtype))
