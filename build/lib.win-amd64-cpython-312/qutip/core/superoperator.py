__all__ = [
    'liouvillian', 'lindblad_dissipator', 'operator_to_vector',
    'vector_to_operator', 'stack_columns', 'unstack_columns', 'stacked_index',
    'unstacked_index', 'spost', 'spre', 'sprepost', 'reshuffle',
]

import functools
from typing import TypeVar, overload
import numpy as np

from .qobj import Qobj
from .cy.qobjevo import QobjEvo
from . import data as _data
from .dimensions import Compound, SuperSpace, Space


def _map_over_compound_operators(f):
    """
    Convert a function which takes Qobj into one that can also take compound
    operators like QobjEvo, and applies itself over all the components.
    """
    @functools.wraps(f)
    def out(qobj):
        # To avoid circular dependencies
        from .cy.qobjevo import QobjEvo
        if isinstance(qobj, QobjEvo):
            return qobj.linear_map(f, _skip_check=True)
        if not isinstance(qobj, Qobj):
            raise TypeError("expected a quantum object")
        return f(qobj)
    return out


@overload
def liouvillian(
    H: Qobj,
    c_ops: list[Qobj],
    data_only: bool,
    chi: list[float]
) -> Qobj: ...

@overload
def liouvillian(
    H: Qobj | QobjEvo,
    c_ops: list[Qobj | QobjEvo],
    data_only: bool,
    chi: list[float]
) -> QobjEvo: ...

def liouvillian(
    H: Qobj | QobjEvo = None,
    c_ops: list[Qobj | QobjEvo] = None,
    data_only: bool = False,
    chi: list[float] = None,
) -> Qobj | QobjEvo:
    """Assembles the Liouvillian superoperator from a Hamiltonian
    and a ``list`` of collapse operators.

    Parameters
    ----------
    H : Qobj or QobjEvo, optional
        System Hamiltonian or Hamiltonian component of a Liouvillian.
        Considered `0` if not given.

    c_ops : array_like of Qobj or QobjEvo, optional
        A ``list`` or ``array`` of collapse operators.

    data_only : bool, default: False
        Return the data object instead of a Qobj

    chi : array_like of float, optional
        In some systems it is possible to determine the statistical moments
        (mean, variance, etc) of the probability distributions of occupation of
        various states by numerically evaluating the derivatives of the steady
        state occupation probability as a function of artificial phase
        parameters ``chi`` which are included in the
        :func:`lindblad_dissipator` for each collapse operator. See the
        documentation of :func:`lindblad_dissipator` for references and further
        details. This parameter is deprecated and may be removed in QuTiP 5.

    Returns
    -------
    L : Qobj or QobjEvo
        Liouvillian superoperator.

    """
    # To avoid circular dependencies
    from .cy.qobjevo import QobjEvo
    if (
        data_only
        and (isinstance(H, QobjEvo)
             or any(isinstance(op, QobjEvo) for op in c_ops))
    ):
        raise ValueError("Cannot return the data object when computing the"
                         " liouvillian with QobjEvo")

    c_ops = c_ops or []
    if isinstance(c_ops, (Qobj, QobjEvo)):
        c_ops = [c_ops]
    if chi and len(chi) != len(c_ops):
        raise ValueError('chi must be a list with same length as c_ops')
    chi = chi or [0] * len(c_ops)

    if H is None:
        # No Hamiltonian, add the lindblad_dissipator of c_ops:
        if not c_ops:
            raise ValueError("The liouvillian need an Hamiltonian"
                             " and/or c_ops")
        out = sum(lindblad_dissipator(c_op, chi=chi_)
                  for c_op, chi_ in zip(c_ops, chi))
        return out.data if data_only else out
    elif not H.isoper:
        raise TypeError("Invalid type for Hamiltonian.")

    if isinstance(H, QobjEvo) or any(isinstance(op, QobjEvo) for op in c_ops):
        # With QobjEvo, faster computation using Data is not used
        L = -1.0j * (spre(H) - spost(H))
        L += sum(lindblad_dissipator(c_op, chi=chi_)
                 for c_op, chi_ in zip(c_ops, chi))
        return L
    spI = _data.identity_like(H.data)
    data = _data.mul(_data.kron(spI, H.data), -1j)
    data = _data.add(data, _data.kron_transpose(H.data, spI),
                     scale=1j)

    for c_op, chi_ in zip(c_ops, chi):
        c = c_op.data
        cd = c.adjoint()
        cdc = _data.matmul(cd, c)
        data = _data.add(data, _data.kron(c.conj(), c), np.exp(1j*chi_))
        data = _data.add(data, _data.kron(spI, cdc), -0.5)
        data = _data.add(data, _data.kron_transpose(cdc, spI), -0.5)

    if data_only:
        return data
    else:
        return Qobj(data,
                    dims=[H._dims, H._dims],
                    superrep='super',
                    copy=False)


@overload
def lindblad_dissipator(
    a: Qobj,
    b: Qobj,
    data_only: bool,
    chi: list[float]
) -> Qobj: ...

@overload
def lindblad_dissipator(
    a: Qobj | QobjEvo,
    b: Qobj | QobjEvo,
    data_only: bool,
    chi: list[float]
) -> QobjEvo: ...

def lindblad_dissipator(
    a: Qobj | QobjEvo,
    b: Qobj | QobjEvo = None,
    data_only: bool = False,
    chi: list[float] = None,
) -> Qobj | QobjEvo:
    """
    Lindblad dissipator (generalized) for a single pair of collapse operators
    (a, b), or for a single collapse operator (a) when b is not specified:

    .. math::

        \\mathcal{D}[a,b]\\rho = a \\rho b^\\dagger -
        \\frac{1}{2}a^\\dagger b\\rho - \\frac{1}{2}\\rho a^\\dagger b

    Parameters
    ----------
    a : Qobj or QobjEvo
        Left part of collapse operator.

    b : Qobj or QobjEvo, optional
        Right part of collapse operator. If not specified, b defaults to a.

    chi : float, optional
        In some systems it is possible to determine the statistical moments
        (mean, variance, etc) of the probability distribution of the occupation
        numbers of states by numerically evaluating the derivatives of the
        steady state occupation probability as a function of an artificial
        phase parameter ``chi`` which multiplies the ``a \\rho a^dagger`` term
        of the dissipator by ``e ^ (i * chi)``. The factor ``e ^ (i * chi)`` is
        introduced via the generating function of the statistical moments. For
        examples of the technique, see `Full counting statistics of
        nano-electromechanical systems
        <https://arxiv.org/abs/cond-mat/0410322>`_ and `Photon-mediated
        electron transport in hybrid circuit-QED
        <https://arxiv.org/abs/1303.7449>`_. This parameter is deprecated and
        may be removed in QuTiP 5.

    data_only : bool, default: False
        Return the data object instead of a Qobj

    Returns
    -------
    D : qobj, QobjEvo
        Lindblad dissipator superoperator.
    """
    # To avoid circular dependencies
    from .cy.qobjevo import QobjEvo
    if data_only and (isinstance(a, QobjEvo) or isinstance(b, QobjEvo)):
        raise ValueError("Cannot return the data object when computing the"
                         " collapse of a QobjEvo")
    if b is None:
        b = a
    ad_b = a.dag() * b
    if chi:
        D = (
            spre(a) * spost(b.dag()) * np.exp(1j * chi)
            - 0.5 * spre(ad_b)
            - 0.5 * spost(ad_b)
        )
    else:
        D = spre(a) * spost(b.dag()) - 0.5 * spre(ad_b) - 0.5 * spost(ad_b)

    return D.data if data_only else D


@_map_over_compound_operators
def operator_to_vector(op: Qobj) -> Qobj:
    """
    Create a vector representation given a quantum operator in matrix form.
    The passed object should have a ``Qobj.type`` of 'oper' or 'super'; this
    function is not designed for general-purpose matrix reshaping.

    Parameters
    ----------
    op : Qobj or QobjEvo
        Quantum operator in matrix form.  This must have a type of 'oper' or
        'super'.

    Returns
    -------
    Qobj or QobjEvo
        The same object, but re-cast into a column-stacked-vector form of type
        'operator-ket'.  The output is the same type as the passed object.
    """
    if op.type in ['super', 'operator-ket', 'operator-bra']:
        raise TypeError("Cannot convert object already "
                        "in super representation")
    return Qobj(stack_columns(op.data),
                dims=[op.dims, [1]],
                superrep="super",
                copy=False)


@_map_over_compound_operators
def vector_to_operator(op: Qobj) -> Qobj:
    """
    Create a matrix representation given a quantum operator in vector form.
    The passed object should have a ``Qobj.type`` of 'operator-ket'; this
    function is not designed for general-purpose matrix reshaping.

    Parameters
    ----------
    op : Qobj or QobjEvo
        Quantum operator in column-stacked-vector form.  This must have a type
        of 'operator-ket'.

    Returns
    -------
    Qobj or QobjEvo
        The same object, but re-cast into "standard" operator form.  The output
        is the same type as the passed object.
    """
    if not op.isoperket:
        raise TypeError("only defined for operator-kets")
    if op.superrep != "super":
        raise TypeError("only defined for operator-kets in super format")
    dims = op.dims[0]
    return Qobj(unstack_columns(op.data, (np.prod(dims[0]), np.prod(dims[1]))),
                dims=dims,
                copy=False)


QobjOrArray = TypeVar("QobjOrArray", Qobj, np.ndarray)


def stack_columns(matrix: QobjOrArray) -> QobjOrArray:
    """
    Stack the columns in a data-layer type, useful for converting an operator
    into a superoperator representation.
    """
    if not isinstance(matrix, (_data.Data, np.ndarray)):
        raise TypeError(
            "input " + repr(type(matrix)) + " is not data-layer type"
        )
    if isinstance(matrix, np.ndarray):
        return matrix.ravel('F')[:, None]
    return _data.column_stack(matrix)


def unstack_columns(
    vector: QobjOrArray,
    shape: tuple[int, int] = None,
) -> QobjOrArray:
    """
    Unstack the columns in a data-layer type back into a 2D shape, useful for
    converting an operator in vector form back into a regular operator.  If
    `shape` is not passed, the output operator will be assumed to be square.
    """
    if not isinstance(vector, (_data.Data, np.ndarray)):
        raise TypeError(
            "input " + repr(type(vector)) + " is not data-layer type"
        )
    if (
        (isinstance(vector, _data.Data) and vector.shape[1] != 1)
        or (isinstance(vector, np.ndarray)
            and ((vector.ndim == 2 and vector.shape[1] != 1)
                 or vector.ndim > 2))
    ):
        raise TypeError("input is not a single column")
    if shape is None:
        n = int(np.sqrt(vector.shape[0]))
        if n * n != vector.shape[0]:
            raise ValueError(
                "input cannot be made square, but no specific shape given"
            )
        shape = (n, n)
    if isinstance(vector, np.ndarray):
        return vector.reshape(shape, order='F')
    return _data.column_unstack(vector, shape[0])


def unstacked_index(size, index):
    """
    Convert an index into a column-stacked square operator with `size` rows and
    columns, into a pair of indices into the unstacked operator.
    """
    return index % size, index // size


def stacked_index(size, row, col):
    """
    Convert a pair of indices into a square operator of `size` into a single
    index into the column-stacked version of the operator.
    """
    return row + size*col


AnyQobj = TypeVar("AnyQobj", Qobj, QobjEvo)


@_map_over_compound_operators
def spost(A: AnyQobj) -> AnyQobj:
    """
    Superoperator formed from post-multiplication by operator A

    Parameters
    ----------
    A : Qobj or QobjEvo
        Quantum operator for post multiplication.

    Returns
    -------
    super : Qobj or QobjEvo
        Superoperator formed from input qauntum object.
    """
    if not A.isoper:
        raise TypeError('Input is not a quantum operator')
    data = _data.kron_transpose(A.data, _data.identity_like(A.data))
    return Qobj(data,
                dims=[A._dims, A._dims],
                superrep='super',
                isherm=A._isherm,
                copy=False)


@_map_over_compound_operators
def spre(A: AnyQobj) -> AnyQobj:
    """Superoperator formed from pre-multiplication by operator A.

    Parameters
    ----------
    A : Qobj or QobjEvo
        Quantum operator for pre-multiplication.

    Returns
    -------
    super :Qobj or QobjEvo
        Superoperator formed from input quantum object.
    """
    if not A.isoper:
        raise TypeError('Input is not a quantum operator')
    data = _data.kron(_data.identity_like(A.data), A.data)
    return Qobj(data,
                dims=[A._dims, A._dims],
                superrep='super',
                isherm=A._isherm,
                copy=False)


def _drop_projected_dims(dims):
    """
    Eliminate subsystems that has been collapsed to only one state due to
    a projection.
    """
    return [d for d in dims if d != 1]


@overload
def sprepost(A: Qobj, B: Qobj) -> Qobj: ...

@overload
def sprepost(A: Qobj | QobjEvo, B: Qobj | QobjEvo) -> QobjEvo: ...

def sprepost(A, B):
    """
    Superoperator formed from pre-multiplication by A and post-multiplication
    by B.

    Parameters
    ----------
    A : Qobj or QobjEvo
        Quantum operator for pre-multiplication.

    B : Qobj or QobjEvo
        Quantum operator for post-multiplication.

    Returns
    -------
    super : Qobj or QobjEvo
        Superoperator formed from input quantum objects.
    """
    # To avoid circular dependencies
    from .cy.qobjevo import QobjEvo
    if (isinstance(A, QobjEvo) or isinstance(B, QobjEvo)):
        return spre(A) * spost(B)
    dims = [[_drop_projected_dims(A.dims[0]),
             _drop_projected_dims(B.dims[1])],
            [_drop_projected_dims(A.dims[1]),
             _drop_projected_dims(B.dims[0])]]
    return Qobj(_data.kron_transpose(B.data, A.data),
                dims=dims,
                superrep='super',
                isherm=A._isherm and B._isherm,
                copy=False)


def _to_super_of_tensor(q_oper):
    """
    Transform a superoperator composed of multiple space into a superoperator
    over a composite spaces.
    """
    msg = "Reshuffling is only supported for square operators."
    if not q_oper._dims.issuper:
        raise TypeError("Reshuffling is only supported on type='super' "
                        "or type='operator-ket'.")
    if q_oper.isoper and not q_oper._dims.issquare:
        raise NotImplementedError(msg)

    dims = q_oper._dims[0]
    if isinstance(dims, SuperSpace):
        return q_oper.copy()

    perm_idxs = [[], []]

    if isinstance(dims, Compound):
        shift = 0
        for space in dims.spaces:
            if not isinstance(space, SuperSpace) or not space.oper.issquare:
                raise NotImplementedError(msg)
            space_dims = space.oper.to_
            if type(space_dims) is Space:
                perm_idxs[0] += [shift]
                perm_idxs[1] += [shift + 1]
                shift += 2
            elif isinstance(space_dims, Compound):
                N = len(space_dims.spaces)
                perm_idxs[0] += [shift + i for i in range(N)]
                perm_idxs[1] += [shift + N + i for i in range(N)]
                shift += 2 * N
            else:
                # ENR space or other complex spaces
                raise NotImplementedError("Reshuffling with non standard space"
                                          "is not supported.")

    return q_oper.permute(perm_idxs)


def _to_tensor_of_super(q_oper):
    """
    Transform a superoperator composed of multiple space into a tensor of
    superoperator on each spaces.
    """
    msg = "Reshuffling is only supported for square operators."
    if not q_oper._dims[0].issuper:
        raise TypeError("Reshuffling is only supported on type='super' "
                        "or type='operator-ket'.")

    dims = q_oper._dims[0]
    perm_idxs = []

    if isinstance(dims, Compound):
        shift = 0
        for space in dims.spaces:
            if not isinstance(space, SuperSpace) or not space.oper.issquare:
                raise TypeError(msg)
            space_dims = space.oper.to_
            if type(space_dims) is Space:
                perm_idxs += [[shift], [shift + 1]]
                shift += 2
            elif isinstance(space_dims, Compound):
                N = len(space_dims.spaces)
                idxs = range(0, N * 2, 2)
                perm_idxs += [[i + shift] for i in idxs]
                perm_idxs += [[i + shift + 1] for i in idxs]
                shift += N * 2
            else:
                # ENR space or other complex spaces
                raise NotImplementedError("Reshuffling with non standard space"
                                          "is not supported.")
    elif isinstance(dims, SuperSpace):
        if isinstance(dims.oper.to_, Compound):
            step = len(dims.oper.to_.spaces)
            perm_idxs = sum([[[i], [i+step]] for i in range(step)], [])
        else:
            return q_oper

    return q_oper.permute(perm_idxs)


def reshuffle(q_oper: Qobj) -> Qobj:
    """
    Column-reshuffles a super operator or a operator-ket Qobj.
    """
    if q_oper.type not in ["super", "operator-ket"]:
        raise TypeError("Reshuffling is only supported on type='super' "
                        "or type='operator-ket'.")

    if isinstance(q_oper._dims[0], Compound):
        return _to_super_of_tensor(q_oper)
    else:
        return _to_tensor_of_super(q_oper)
