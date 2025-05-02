"""
This module contains functions for generating Qobj representation of a variety
of commonly occuring quantum operators.
"""
# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = [
    'jmat', 'spin_Jx', 'spin_Jy', 'spin_Jz', 'spin_Jm', 'spin_Jp',
    'spin_J_set', 'sigmap', 'sigmam', 'sigmax', 'sigmay', 'sigmaz',
    'destroy', 'create', 'fdestroy', 'fcreate', 'qeye', 'identity',
    'position', 'momentum', 'num', 'squeeze', 'squeezing', 'displace',
    'commutator', 'qutrit_ops', 'qdiags', 'phase', 'qzero', 'charge',
    'tunneling', 'qft', 'qzero_like', 'qeye_like', 'swap',
]

import numpy as np
from typing import Literal, overload
from . import data as _data
from .qobj import Qobj
from .dimensions import Space
from .. import settings
from ..typing import DimensionLike, SpaceLike, LayerType

def qdiags(
    diagonals: np.typing.ArrayLike | list[np.typing.ArrayLike],
    offsets: int | list[int] = None,
    dims: DimensionLike = None,
    shape: tuple[int, int] = None,
    *,
    dtype: LayerType = None,
) -> Qobj:
    """
    Constructs an operator from an array of diagonals.

    Parameters
    ----------
    diagonals : array_like or sequence of array_like
        Array of elements to place along the selected diagonals.

    offsets : int or sequence of ints, optional
        Sequence for diagonals to be set:
            - k=0 main diagonal
            - k>0 kth upper diagonal
            - k<0 kth lower diagonal

    dims : list, optional
        Dimensions for operator

    shape : list, tuple, optional
        Shape of operator.  If omitted, a square operator large enough
        to contain the diagonals is generated.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
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
    dtype = _data._parse_default_dtype(dtype, "diagonal")
    offsets = [0] if offsets is None else offsets
    if not isinstance(offsets, list):
        offsets = [offsets]
    if len(offsets) == 1 and offsets[0] != 0:
        isherm = False
        isunitary = False
    elif offsets == [0]:
        isherm = np.all(np.abs(np.imag(diagonals)) <= settings.core["atol"])
        isunitary = np.all(np.abs(diagonals) - 1 <= settings.core["atol"])
    else:
        isherm = None
        isunitary = None
    data = _data.diag[dtype](diagonals, offsets, shape)
    return Qobj(
        data, copy=False, dtype=dtype,
        dims=dims, isherm=isherm, isunitary=isunitary
    )


@overload
def jmat(
    j: float,
    which: Literal[None],
    *,
    dtype: LayerType = None
) -> tuple[Qobj]: ...

@overload
def jmat(
    j: float,
    which: Literal["x", "y", "z", "+", "-"],
    *,
    dtype: LayerType = None
) -> Qobj: ...

def jmat(
    j: float,
    which: Literal["x", "y", "z", "+", "-", None] = None,
    *,
    dtype: LayerType = None
) -> Qobj | tuple[Qobj]:
    """Higher-order spin operators:

    Parameters
    ----------
    j : float
        Spin of operator

    which : str, optional
        Which operator to return 'x','y','z','+','-'.
        If not given, then output is ['x','y','z']

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
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
    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
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
        return Qobj(_jplus(j, dtype=dtype), dims=dims, dtype=dtype,
                    isherm=False, isunitary=False, copy=False)
    if which == '-':
        return Qobj(_jplus(j, dtype=dtype).adjoint(), dims=dims, dtype=dtype,
                    isherm=False, isunitary=False, copy=False)
    if which == 'x':
        A = _jplus(j, dtype=dtype)
        return Qobj(_data.add(A, A.adjoint()) * 0.5, dims=dims, dtype=dtype,
                    isherm=True, isunitary=False, copy=False)
    if which == 'y':
        A = _data.mul(_jplus(j, dtype=dtype), -0.5j)
        return Qobj(_data.add(A, A.adjoint()), dims=dims, dtype=dtype,
                    isherm=True, isunitary=False, copy=False)
    if which == 'z':
        return Qobj(_jz(j, dtype=dtype), dims=dims, dtype=dtype,
                    isherm=True, isunitary=False, copy=False)
    raise ValueError('Invalid spin operator: ' + which)


def _jplus(j, *, dtype=None):
    """
    Internal functions for generating the data representing the J-plus
    operator.
    """
    m = np.arange(j, -j - 1, -1, dtype=complex)
    data = np.sqrt(j * (j + 1) - m * (m + 1))[1:]
    return _data.diag[dtype](data, 1)


def _jz(j, *, dtype=None):
    """
    Internal functions for generating the data representing the J-z operator.
    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    N = int(2*j + 1)
    data = np.array([j-k for k in range(N)], dtype=complex)
    return _data.diag[dtype](data, 0)


#
# Spin j operators:
#
def spin_Jx(j: float, *, dtype: LayerType = None) -> Qobj:
    """Spin-j x operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'x', dtype=dtype)


def spin_Jy(j: float, *, dtype: LayerType = None) -> Qobj:
    """Spin-j y operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'y', dtype=dtype)


def spin_Jz(j: float, *, dtype: LayerType = None) -> Qobj:
    """Spin-j z operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'z', dtype=dtype)


def spin_Jm(j: float, *, dtype: LayerType = None) -> Qobj:
    """Spin-j annihilation operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '-', dtype=dtype)


def spin_Jp(j: float, *, dtype: LayerType = None) -> Qobj:
    """Spin-j creation operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '+', dtype=dtype)


def spin_J_set(j: float, *, dtype: LayerType = None) -> tuple[Qobj]:
    """Set of spin-j operators (x, y, z)

    Parameters
    ----------
    j : float
        Spin of operators

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    list : tuple of Qobj
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
_SIGMAX._isunitary = True
_SIGMAY = 2 * jmat(0.5, 'y')
_SIGMAY._isunitary = True
_SIGMAZ = 2 * jmat(0.5, 'z')
_SIGMAZ._isunitary = True


def sigmap(*, dtype: LayerType = None) -> Qobj:
    """Creation operator for Pauli spins.

    Parameters
    ----------
    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Examples
    --------
    >>> sigmap() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 0.  0.]]

    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    return _SIGMAP.to(dtype, True)


def sigmam(*, dtype: LayerType = None) -> Qobj:
    """Annihilation operator for Pauli spins.

    Parameters
    ----------
    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Examples
    --------
    >>> sigmam() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  0.]
     [ 1.  0.]]

    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    return _SIGMAM.to(dtype, True)


def sigmax(*, dtype: LayerType = None) -> Qobj:
    """Pauli spin 1/2 sigma-x operator

    Parameters
    ----------
    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Examples
    --------
    >>> sigmax() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 1.  0.]]

    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    return _SIGMAX.to(dtype, True)


def sigmay(*, dtype: LayerType = None) -> Qobj:
    """Pauli spin 1/2 sigma-y operator.

    Parameters
    ----------
    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Examples
    --------
    >>> sigmay() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.-1.j]
     [ 0.+1.j  0.+0.j]]

    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    return _SIGMAY.to(dtype, True)


def sigmaz(*, dtype: LayerType = None) -> Qobj:
    """Pauli spin 1/2 sigma-z operator.

    Parameters
    ----------
    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Examples
    --------
    >>> sigmaz() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]

    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    return _SIGMAZ.to(dtype, True)


def destroy(N: int, offset: int = 0, *, dtype: LayerType = None) -> Qobj:
    """
    Destruction (lowering) operator.

    Parameters
    ----------
    N : int
        Number of basis states in the Hilbert space.

    offset : int, default: 0
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
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
    dtype = _data._parse_default_dtype(dtype, "diagonal")
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset+1, N+offset, dtype=complex))
    return qdiags(data, 1, dtype=dtype)


def create(N: int, offset: int = 0, *, dtype: LayerType = None) -> Qobj:
    """
    Creation (raising) operator.

    Parameters
    ----------
    N : int
        Number of basis states in the Hilbert space.

    offset : int, default: 0
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Qobj for raising operator.

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
    dtype = _data._parse_default_dtype(dtype, "diagonal")
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset+1, N+offset, dtype=complex))
    return qdiags(data, -1, dtype=dtype)


def fdestroy(n_sites: int, site, dtype: LayerType = None) -> Qobj:
    """
    Fermionic destruction operator.
    We use the Jordan-Wigner transformation,
    making use of the Jordan-Wigner ZZ..Z strings,
    to construct this as follows:

    .. math::

        a_j = \\sigma_z^{\\otimes j} \\otimes
        (\\frac{\\sigma_x + i \\sigma_y}{2})
        \\otimes I^{\\otimes N-j-1}

    Parameters
    ----------
    n_sites : int
        Number of sites in Fock space.

    site : int
        The site in Fock space to add a fermion to.
        Corresponds to j in the above JW transform.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Qobj for destruction operator.

    Examples
    --------
    >>> fdestroy(2) # doctest: +SKIP
    Quantum object: dims=[[2 2], [2 2]], shape=(4, 4), \
    type='oper', isherm=False
    Qobj data =
    [[0. 0. 1. 0.]
    [0. 0. 0. 1.]
    [0. 0. 0. 0.]
    [0. 0. 0. 0.]]
    """
    return _f_op(n_sites, site, 'destruction', dtype=dtype)


def fcreate(n_sites: int, site, dtype: LayerType = None) -> Qobj:
    """
    Fermionic creation operator.
    We use the Jordan-Wigner transformation,
    making use of the Jordan-Wigner ZZ..Z strings,
    to construct this as follows:

    .. math::

        a_j = \\sigma_z^{\\otimes j}
        \\otimes (\\frac{\\sigma_x - i \\sigma_y}{2})
        \\otimes I^{\\otimes N-j-1}


    Parameters
    ----------
    n_sites : int
        Number of sites in Fock space.

    site : int
        The site in Fock space to add a fermion to.
        Corresponds to j in the above JW transform.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Qobj for raising operator.

    Examples
    --------
    >>> fcreate(2) # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), \
    type = oper, isherm = False
    Qobj data =
    [[0. 0. 0. 0.]
    [0. 0. 0. 0.]
    [1. 0. 0. 0.]
    [0. 1. 0. 0.]]
    """
    return _f_op(n_sites, site, 'creation', dtype=dtype)


def _f_op(n_sites, site, action, dtype: LayerType = None,):
    """ Makes fermionic creation and destruction operators.
    We use the Jordan-Wigner transformation,
    making use of the Jordan-Wigner ZZ..Z strings,
    to construct this as follows:

    .. math::

        a_j = \\sigma_z^{\\otimes j}
        \\otimes (frac{sigma_x \\pm i sigma_y}{2})
        \\otimes I^{\\otimes N-j-1}

    Parameters
    ----------
    action : str
        The type of operator to build.
        Can only be 'creation' or 'destruction'

    n_sites : int
        Number of sites in Fock space.

    site : int
        The site in Fock space to create/destroy a fermion on.
        Corresponds to j in the above JW transform.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Qobj for destruction operator.
    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    # get `tensor` and sigma z objects
    from .tensor import tensor
    s_z = 2 * jmat(0.5, 'z', dtype=dtype)

    # sanity check
    if site < 0:
        raise ValueError(f'The specified site {site} cannot be \
                         less than 0.')
    elif 0 >= n_sites:
        raise ValueError(f'The specified number of sites {n_sites} \
                         cannot be equal to or less than 0.')
    elif site >= n_sites:
        raise ValueError(f'The specified site {site} is not in \
                         the range of {n_sites} sites.')

    # figure out which operator to build
    if action.lower() == 'creation':
        operator = create(2, dtype=dtype)
    elif action.lower() == 'destruction':
        operator = destroy(2, dtype=dtype)
    else:
        raise TypeError("Unknown operator '%s'. `action` must be \
                        either 'creation' or 'destruction.'" % action)

    eye = identity(2, dtype=dtype)
    opers = [s_z] * site + [operator] + [eye] * (n_sites - site - 1)
    out = tensor(opers).to(dtype)
    out.isherm = False
    out._isunitary = False
    return out


def qzero(
    dimensions: SpaceLike,
    dims_right: SpaceLike = None,
    *,
    dtype: LayerType = None
) -> Qobj:
    """
    Zero operator.

    Parameters
    ----------
    dimensions : int, list of int, list of list of int, Space
        Number of basis states in the Hilbert space. If provided as a list of
        ints, then the dimension is the product over this list, but the
        ``dims`` property of the new Qobj are set to this list.  This can
        produce either `oper` or `super` depending on the passed `dimensions`.

    dims_right : int, list of int, list of list of int, Space, optional
        Number of basis states in the right Hilbert space when the operator is
        rectangular.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    qzero : qobj
        Zero operator Qobj.

    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    dims_left = Space(dimensions)
    size_left = dims_left.size
    if dims_right is None:
        dims_right = dims_left
        size_right = size_left
    else:
        dims_right = Space(dims_right)
        size_right = dims_right.size
    dims = [dims_left, dims_right]
    # A sparse matrix with no data is equal to a zero matrix.
    return Qobj(_data.zeros[dtype](size_left, size_right), dims=dims,
                isherm=True, isunitary=False, copy=False, dtype=dtype)


def qzero_like(qobj: Qobj) -> Qobj:
    """
    Zero operator of the same dims and type as the reference.

    Parameters
    ----------
    qobj : Qobj, QobjEvo
        Reference quantum object to copy the dims from.

    Returns
    -------
    qzero : qobj
        Zero operator Qobj.

    """

    return Qobj(
        _data.zeros[qobj.dtype](*qobj.shape), dims=qobj._dims,
        isherm=True, isunitary=False, copy=False, dtype=qobj.dtype
    )


def qeye(dimensions: SpaceLike, *, dtype: LayerType = None) -> Qobj:
    """
    Identity operator.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int), Space
        Number of basis states in the Hilbert space. If provided as a list of
        ints, then the dimension is the product over this list, but the
        ``dims`` property of the new Qobj are set to this list.  This can
        produce either `oper` or `super` depending on the passed `dimensions`.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
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
    dtype = _data._parse_default_dtype(dtype, "diagonal")
    dimensions = Space(dimensions)
    return Qobj(_data.identity[dtype](dimensions.size), dims=[dimensions]*2,
                isherm=True, isunitary=True, copy=False, dtype=dtype)


# Name alias.
identity = qeye


def qeye_like(qobj: Qobj) -> Qobj:
    """
    Identity operator with the same dims and type as the reference quantum
    object.

    Parameters
    ----------
    qobj : Qobj, QobjEvo
        Reference quantum object to copy the dims from.

    Returns
    -------
    oper : qobj
        Identity operator Qobj.

    """
    if qobj.shape[0] != qobj.shape[1]:
        raise ValueError(
            "Can't create an identity matrix like a non square matrix."
        )
    return Qobj(
        _data.identity[qobj.dtype](qobj.shape[0]), dims=qobj._dims,
        isherm=True, isunitary=True, copy=False, dtype=qobj.dtype
    )


def position(N: int, offset: int = 0, *, dtype: LayerType = None) -> Qobj:
    """
    Position operator :math:`x = 1 / sqrt(2) * (a + a.dag())`

    Parameters
    ----------
    N : int
        Number of basis states in the Hilbert space.

    offset : int, default: 0
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Position operator as Qobj.
    """
    dtype = _data._parse_default_dtype(dtype, "diagonal")
    a = destroy(N, offset=offset, dtype=dtype)
    position = np.sqrt(0.5) * (a + a.dag())
    position.isherm = True
    position._isunitary = False
    return position.to(dtype)


def momentum(N: int, offset: int = 0, *, dtype: LayerType = None) -> Qobj:
    """
    Momentum operator p=-1j/sqrt(2)*(a-a.dag())

    Parameters
    ----------
    N : int
        Number of basis states in the Hilbert space.

    offset : int, default: 0
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Momentum operator as Qobj.
    """
    dtype = _data._parse_default_dtype(dtype, "diagonal")
    a = destroy(N, offset=offset, dtype=dtype)
    momentum = -1j * np.sqrt(0.5) * (a - a.dag())
    momentum.isherm = True
    momentum._isunitary = False
    return momentum.to(dtype)


def num(N: int, offset: int = 0, *, dtype: LayerType = None) -> Qobj:
    """
    Quantum object for number operator.

    Parameters
    ----------
    N : int
        Number of basis states in the Hilbert space.

    offset : int, default: 0
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
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
    dtype = _data._parse_default_dtype(dtype, "diagonal")
    data = np.arange(offset, offset + N, dtype=complex)
    return qdiags(data, 0, dtype=dtype)


def squeeze(
    N: int,
    z: float,
    offset: int = 0,
    *,
    dtype: LayerType = None,
) -> Qobj:
    """Single-mode squeezing operator.

    Parameters
    ----------
    N : int
        Dimension of hilbert space.

    z : float/complex
        Squeezing parameter.

    offset : int, default: 0
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : :class:`.Qobj`
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
    dtype = _data._parse_default_dtype(dtype, "dense")
    asq = destroy(N, offset=offset, dtype=dtype) ** 2
    op = 0.5*np.conj(z)*asq - 0.5*z*asq.dag()
    out = op.expm(dtype=dtype)
    out.isherm = (N == 2) or (z == 0.)
    out._isunitary = True
    return out


def squeezing(a1: Qobj, a2: Qobj, z: float) -> Qobj:
    """Generalized squeezing operator.

    .. math::

        S(z) = \\exp\\left(\\frac{1}{2}\\left(z^*a_1a_2
        - za_1^\\dagger a_2^\\dagger\\right)\\right)

    Parameters
    ----------
    a1 : :class:`.Qobj`
        Operator 1.

    a2 : :class:`.Qobj`
        Operator 2.

    z : float/complex
        Squeezing parameter.

    Returns
    -------
    oper : :class:`.Qobj`
        Squeezing operator.

    """
    b = 0.5 * (np.conj(z)*(a1 @ a2) - z*(a1.dag() @ a2.dag()))
    return b.expm()


def displace(
    N: int,
    alpha: float,
    offset: int = 0,
    *,
    dtype: LayerType = None,
) -> Qobj:
    """Single-mode displacement operator.

    Parameters
    ----------
    N : int
        Number of basis states in the Hilbert space.

    alpha : float/complex
        Displacement amplitude.

    offset : int, default: 0
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Displacement operator.

    Examples
    --------
    >>> displace(4,0.25) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.96923323+0.j -0.24230859+0.j  0.04282883+0.j -0.00626025+0.j]
     [ 0.24230859+0.j  0.90866411+0.j -0.33183303+0.j  0.07418172+0.j]
     [ 0.04282883+0.j  0.33183303+0.j  0.84809499+0.j -0.41083747+0.j]
     [ 0.00626025+0.j  0.07418172+0.j  0.41083747+0.j  0.90866411+0.j]]

    """
    dtype = _data._parse_default_dtype(dtype, "dense")
    a = destroy(N, offset=offset)
    out = (alpha * a.dag() - np.conj(alpha) * a).expm(dtype=dtype)
    out.isherm = (alpha == 0.)
    out._isunitary = True
    return out


def commutator(
    A: Qobj,
    B: Qobj,
    kind: Literal["normal", "anti"] = "normal"
) -> Qobj:
    """
    Return the commutator of kind `kind` (normal, anti) of the
    two operators A and B.

    Parameters
    ----------
    A, B : :obj:`Qobj`, :obj:`QobjEvo`
        The operators to compute the commutator of.

    kind: str {"normal", "anti"}, default: "anti"
        Which kind of commutator to compute.
    """
    if kind == 'normal':
        return A @ B - B @ A

    elif kind == 'anti':
        return A @ B + B @ A

    else:
        raise TypeError("Unknown commutator kind '%s'" % kind)


def qutrit_ops(*, dtype: LayerType = None) -> list[Qobj]:
    """
    Operators for a three level system (qutrit).

    Parameters
    ----------
    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    opers: array
        `array` of qutrit operators.

    """
    from .states import qutrit_basis

    dtype = _data._parse_default_dtype(dtype, "sparse")
    out = []
    basis = qutrit_basis()
    for i in range(3):
        op = (basis[i] @ basis[i].dag()).to(dtype)
        op.isherm = True
        op._isunitary = False
        out.append(op)
    for i in range(3):
        op = (basis[i] @ basis[(i+1)%3].dag()).to(dtype)
        op.isherm = False
        op._isunitary = False
        out.append(op)
    return out


def phase(N: int, phi0: float = 0, *, dtype: LayerType = None) -> Qobj:
    """
    Single-mode Pegg-Barnett phase operator.

    Parameters
    ----------
    N : int
        Number of basis states in the Hilbert space.

    phi0 : float, default: 0
        Reference phase.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Phase operator with respect to reference phase.

    Notes
    -----
    The Pegg-Barnett phase operator is Hermitian on a truncated Hilbert space.

    """
    dtype = _data._parse_default_dtype(dtype, "dense")
    phim = phi0 + (2 * np.pi * np.arange(N)) / N  # discrete phase angles
    n = np.arange(N)[:, np.newaxis]
    states = np.array([np.sqrt(kk) / np.sqrt(N) * np.exp(1j * n * kk)
                       for kk in phim])
    ops = np.sum([np.outer(st, st.conj()) for st in states], axis=0)
    return Qobj(
        ops,
        isherm=True,
        isunitary=False,
        dims=[[N], [N]],
        copy=False
    ).to(dtype)


def charge(
    Nmax: int,
    Nmin: int = None,
    frac: float = 1,
    *,
    dtype: LayerType = None
) -> Qobj:
    """
    Generate the diagonal charge operator over charge states
    from Nmin to Nmax.

    Parameters
    ----------
    Nmax : int
        Maximum charge state to consider.

    Nmin : int, default: -Nmax
        Lowest charge state to consider.

    frac : float, default: 1
        Specify fractional charge if needed.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    C : Qobj
        Charge operator over [Nmin, Nmax].

    Notes
    -----
    .. versionadded:: 3.2

    """
    dtype = _data._parse_default_dtype(dtype, "diagonal")
    if Nmin is None:
        Nmin = -Nmax
    diag = frac * np.arange(Nmin, Nmax+1, dtype=float)
    out = qdiags(diag, 0, dtype=dtype)
    out._isunitary = (len(diag) <= 2) and np.all(np.abs(diag) == 1.)
    return out


def tunneling(N: int, m: int = 1, *, dtype: LayerType = None) -> Qobj:
    r"""
    Tunneling operator with elements of the form
    :math:`\\sum |N><N+m| + |N+m><N|`.

    Parameters
    ----------
    N : int
        Number of basis states in the Hilbert space.

    m : int, default: 1
        Number of excitations in tunneling event.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    T : Qobj
        Tunneling operator.
    """
    diags = [np.ones(N-m, dtype=int), np.ones(N-m, dtype=int)]
    T = qdiags(diags, [m, -m], dtype=dtype)
    T.isherm = True
    T._isunitary = (m * 2 == N)
    return T


def qft(dimensions: SpaceLike, *, dtype: LayerType = None) -> Qobj:
    """
    Quantum Fourier Transform operator.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Number of basis states in the Hilbert space. If provided as a list of
        ints, then the dimension is the product over this list, but the
        ``dims`` property of the new Qobj are set to this list.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    QFT: qobj
        Quantum Fourier transform operator.

    """
    dtype = _data._parse_default_dtype(dtype, "dense")
    dimensions = Space(dimensions)
    dims = [dimensions]*2
    N2 = dimensions.size

    phase = 2.0j * np.pi / N2
    arr = np.arange(N2)
    L, M = np.meshgrid(arr, arr)
    data = np.exp(phase * (L * M)) / np.sqrt(N2)
    return Qobj(data, isherm=False, isunitary=True, dims=dims).to(dtype)


def swap(N: int, M: int, *, dtype: LayerType = None) -> Qobj:
    """
    Operator that exchanges the order of tensored spaces:

        swap(N, M) @ tensor(ketN, ketM) == tensor(ketM, ketN)

    parameters
    ----------
    N : int
        Number of basis states in the first Hilbert space.

    M : int
        Number of basis states in the second Hilbert space.
    """
    dtype = _data._parse_default_dtype(dtype, "sparse")

    if N == 1 and M == 1:
        return qeye([1, 1], dtype=dtype)

    data = np.ones(N * M)
    rows = np.arange(N * M + 1)  # last entry is nnz
    cols = np.ravel(M * np.arange(N)[None, :] + np.arange(M)[:, None])
    return Qobj(
        _data.CSR((data, cols, rows), (N * M, N * M)),
        dims=[[M, N], [N, M]],
        isherm=(N == M),
        isunitary=True,
    ).to(dtype)
