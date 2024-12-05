# -*- coding: utf-8 -*-
#
# This module was initially contributed by Ben Criger.
#
"""
This module implements transformations between superoperator representations,
including supermatrix, Kraus, Choi and Chi (process) matrix formalisms.
"""

__all__ = [
    'kraus_to_choi', 'kraus_to_super',
    'to_choi', 'to_chi', 'to_super', 'to_kraus', 'to_stinespring',
]

import itertools
import numbers

import numpy as np
import scipy.linalg

from . import data as _data
from .superoperator import stack_columns, unstack_columns, sprepost
from .tensor import tensor
from .dimensions import flatten
from .qobj import Qobj
from .operators import identity, sigmax, sigmay, sigmaz
from .states import basis


# TODO: revisit when creation routines have dispatching.
_SINGLE_QUBIT_PAULI_BASIS = (
    identity(2).to(_data.CSR),
    sigmax().to(_data.CSR),
    sigmay().to(_data.CSR),
    sigmaz().to(_data.CSR),
)


def _superpauli_basis(nq=1):
    dims = [[[2] * nq] * 2] * 2
    nnz = 8**nq
    data = _data.csr.empty(4**nq, 4**nq, nnz)
    sci = data.as_scipy(full=True)
    ptr, ptr_inc = 0, 2**nq
    # Construct the Pauli basis by vertically stacking rows in sparse format.
    # The CSR format is much more efficient at handling row-stacking, so we
    # actually have to do a little dance through adjoint/transpose to get it
    # into the right format.
    for i, paulis in enumerate(itertools.product(_SINGLE_QUBIT_PAULI_BASIS,
                                                 repeat=nq)):
        basis = paulis[0].data
        for pauli in paulis[1:]:
            basis = _data.kron_csr(basis, pauli.data)
        basis_ket_sci = _data.column_stack_csr(basis).transpose().as_scipy()
        sci.data[ptr : ptr+ptr_inc] = basis_ket_sci.data
        sci.indices[ptr : ptr+ptr_inc] = basis_ket_sci.indices
        sci.indptr[i] = ptr
        ptr += ptr_inc
    sci.indptr[-1] = nnz
    return Qobj(data.adjoint(),
                dims=dims,
                superrep='super',
                isherm=False,
                isunitary=False,
                copy=False)


def _int_log_two(x):
    return int(x).bit_length() - 1


def _is_power_of_two(x):
    return isinstance(x, numbers.Integral) and x == 2**_int_log_two(x)


def _nq(dims):
    dim = np.prod(dims[0][0])
    nq = _int_log_two(dim)
    if 2 ** nq != dim:
        raise ValueError("{} is not an integer power of 2.".format(dim))
    return nq


def isqubitdims(dims: list[list[int]] | list[list[list[int]]]) -> bool:
    """
    Checks whether all entries in a dims list are integer powers of 2.

    Parameters
    ----------
    dims : nested list of ints
        Dimensions to be checked.

    Returns
    -------
    isqubitdims : bool
        True if and only if every member of the flattened dims
        list is an integer power of 2.
    """
    return all(_is_power_of_two(dim) for dim in flatten(dims))


def _to_superpauli(q_oper):
    """
    Convert a superoperator to the Pauli basis (assuming qubit dimensions).

    This is an internal function, as QuTiP does not currently have
    a way to mark that superoperators are represented in the Pauli
    basis as opposed to the column-stacking basis; a Pauli-basis
    ``type='super'`` would thus break other conversion functions.
    """
    # Ensure we start with a column-stacking-basis superoperator.
    sqobj = to_super(q_oper)
    if not isqubitdims(sqobj.dims):
        raise ValueError("Pauli basis is only defined for qubits.")
    nq = _int_log_two(sqobj.shape[0]) // 2
    B = _superpauli_basis(nq) * 2**(-0.5 * nq)
    # To do this, we have to hack a bit and force the dims to match,
    # since the superpauli_basis function makes different assumptions
    # about indices than we need here.
    B.dims = sqobj.dims
    return B.dag() @ sqobj @ B


def _choi_to_kraus(q_oper, tol=1e-9):
    """
    Takes a Choi matrix and returns a list of Kraus operators.
    TODO: Create a new class structure for quantum channels, perhaps as a
    strict sub-class of Qobj.
    """
    vals, vecs = q_oper.eigenstates()
    dims = [q_oper.dims[0][1], q_oper.dims[0][0]]
    shape = (np.prod(q_oper.dims[0][1]), np.prod(q_oper.dims[0][0]))
    return [Qobj(_data.mul(unstack_columns(vec.data, shape=shape), np.sqrt(val)),
                 dims=dims, copy='False')
            for val, vec in zip(vals, vecs) if abs(val) >= tol]


# Individual conversions from Kraus operators are public because the output
# list of Kraus operators is not itself a quantum object.

def kraus_to_choi(kraus_ops: list[Qobj]) -> Qobj:
    r"""
    Convert a list of Kraus operators into Choi representation of the channel.

    Essentially, kraus operators are a decomposition of a Choi matrix,
    and its reconstruction from them should go as
    :math:`E = \sum_{i} |K_i\rangle\rangle \langle\langle K_i|`,
    where we use vector representation of Kraus operators.

    Parameters
    ----------
    kraus_ops : list[Qobj]
        The list of Kraus operators to be converted to Choi representation.

    Returns
    -------
    choi : Qobj
        A quantum object representing the same map as ``kraus_ops``, such that
        ``choi.superrep == "choi"``.
    """
    len_op = np.prod(kraus_ops[0].shape)
    # If Kraus ops have dims [M, N] in qutip notation (act on [N, N] density
    # matrix and produce [M, M] d.m.), Choi matrix Hilbert space will
    # be [[M, N], [M, N]] because Choi Hilbert space
    # is (output space) x (input space).
    choi_dims = [kraus_ops[0].dims] * 2
    # transform a list of Qobj matrices list[sum_ij k_ij |i><j|]
    # into an array of array vectors sum_ij k_ij |i, j>> = sum_I k_I |I>>
    kraus_vectors = np.asarray([
        np.reshape(kraus_op.full(), len_op, order="F")
        for kraus_op in kraus_ops
    ])
    # sum_{I} |k_I|^2 |I>><<I|
    choi_array = np.tensordot(
        kraus_vectors, kraus_vectors.conj(), axes=([0], [0])
    )
    return Qobj(choi_array, choi_dims, superrep="choi", copy=False)


def kraus_to_super(kraus_list: list[Qobj], sparse=False) -> Qobj:
    """
    Convert a list of Kraus operators to a superoperator.

    Parameters
    ----------
    kraus_list : list of Qobj
        The list of Kraus super operators to convert.
    sparse: bool
        Prevents dense intermediates if true.
    """
    if sparse:
        return sum(sprepost(k, k.dag()) for k in kraus_list)
    else:
        return to_super(kraus_to_choi(kraus_list))


def _super_tofrom_choi(q_oper):
    """
    We exploit that the basis transformation between Choi and supermatrix
    representations squares to the identity, so that if we munge Qobj.type,
    we can use the same function.
    """
    if not q_oper.issuper:
        raise ValueError("needs to be a superoperator")
    if q_oper.superrep not in ('super', 'choi'):
        raise ValueError("operator is not in super or choi format")
    data = q_oper.data.to_array()
    dims = q_oper.dims
    new_dims = [[dims[1][1], dims[0][1]], [dims[1][0], dims[0][0]]]
    d0 = np.prod(flatten(new_dims[0]))
    d1 = np.prod(flatten(new_dims[1]))
    s0 = np.prod(dims[0][0])
    s1 = np.prod(dims[1][1])
    data = (
        data.reshape([s0, s1, s0, s1]).transpose(3, 1, 2, 0).reshape([d0, d1])
    )
    return Qobj(data,
                dims=new_dims,
                superrep='super' if q_oper.superrep == 'choi' else 'choi',
                copy=False)


def _choi_to_chi(q_oper):
    """
    Converts a Choi matrix to a Chi matrix in the Pauli basis.

    NOTE: this is only supported for qubits right now. Need to extend to
    Heisenberg-Weyl for other subsystem dimensions.
    """
    nq = _nq(q_oper.dims)
    B = _superpauli_basis(nq).data
    return Qobj(_data.matmul(_data.matmul(B.adjoint(), q_oper.data), B),
                dims=q_oper.dims,
                superrep='chi',
                copy=False)


def _chi_to_choi(q_oper):
    """
    Converts a Chi matrix to a Choi matrix.

    NOTE: this is only supported for qubits right now. Need to extend to
    Heisenberg-Weyl for other subsystem dimensions.
    """
    nq = _nq(q_oper.dims)
    B = _superpauli_basis(nq).data
    # The Chi matrix has tr(chi) == d², so we need to divide out
    # by that to get back to the Choi form.
    return Qobj(_data.mul(
                    _data.matmul(_data.matmul(B, q_oper.data), B.adjoint()),
                    1 / q_oper.shape[0]
                ),
                dims=q_oper.dims,
                superrep='choi',
                copy=False)


def _svd_u_to_kraus(U, S, d, dK, indims, outdims):
    """
    Given a partial isometry U and a vector of square-roots of singular values
    S obtained from a SVD, produces the Kraus operators represented by U.

    Returns
    -------
    Ks : list of Qobj
        Quantum objects represnting each of the Kraus operators.
    """
    # We use U * S since S is 1-index, such that this is equivalent to
    # U . diag(S), but easier to write down.
    data = np.array(U * S).reshape((d, d, dK), order='F').transpose((2, 0, 1))
    return [
        Qobj(x,
             dims=[outdims, indims],
             copy=False)
        for x in data
    ]


def _generalized_kraus(q_oper, threshold=1e-10):
    # TODO: document!
    # TODO: use this to generalize to_kraus to the case where U != V.
    #       This is critical for non-CP maps, as appear in (for example)
    #       diamond norm differences between two CP maps.
    if not q_oper.issuper or q_oper.superrep != "choi":
        raise ValueError("".join([
            "Expected a Choi matrix, got a ", repr(q_oper.type),
            " (superrep ", repr(q_oper.superrep), ")."
        ]))

    # Remember the shape of the underlying space,
    # as we'll need this to make Kraus operators later.
    dL, dR = int(np.sqrt(q_oper.shape[0])), int(np.sqrt(q_oper.shape[1]))
    # Also remember the dims breakout.
    out_dims, in_dims = q_oper.dims
    out_left, out_right = out_dims
    in_left, in_right = in_dims

    # Find the SVD.
    U, S, V = scipy.linalg.svd(q_oper.full())

    # Truncate away the zero singular values, up to a threshold.
    nonzero_idxs = S > threshold
    dK = nonzero_idxs.sum()
    U = np.array(U)[:, nonzero_idxs]
    # We also want S to be a single index array, which np.matrix
    # doesn't allow for. This is stripped by calling array() on it.
    S = np.sqrt(np.array(S)[nonzero_idxs])
    # Since NumPy returns V and not V+, we need to take the dagger
    # to get back to quantum info notation for Stinespring pairs.
    V = np.array(V.conj().T)[:, nonzero_idxs]

    # Next, we convert each of U and V into Kraus operators.
    # Finally, we want the Kraus index to be left-most so that we
    # can map over it when making Qobjs.
    # FIXME: does not preserve dims!
    kU = _svd_u_to_kraus(U, S, dL, dK, out_right, out_left)
    kV = _svd_u_to_kraus(V, S, dL, dK, in_right, in_left)

    return kU, kV


def _choi_to_stinespring(q_oper, threshold=1e-10):
    # TODO: document!
    kU, kV = _generalized_kraus(q_oper, threshold=threshold)

    assert len(kU) == len(kV)
    dK = len(kU)
    dL = kU[0].shape[0]
    dR = kV[0].shape[1]
    # Also remember the dims breakout.
    out_dims, in_dims = q_oper.dims
    out_left, out_right = out_dims
    in_left, in_right = in_dims

    A = Qobj(_data.zeros(dK * dL, dL),
             dims=[out_left + [dK], out_right + [1]],
             isherm=True,
             isunitary=False,
             copy=False)
    B = Qobj(_data.zeros(dK * dR, dR),
             dims=[in_left + [dK], in_right + [1]],
             isherm=True,
             isunitary=False,
             copy=False)

    for idx_kraus, (KL, KR) in enumerate(zip(kU, kV)):
        A += tensor(KL, basis(dK, idx_kraus))
        B += tensor(KR, basis(dK, idx_kraus))

    # There is no input (right) Kraus index, so strip that off.
    A.dims = [out_left + [dK], out_right]
    B.dims = [in_left + [dK], in_right]

    return A, B


def to_choi(q_oper: Qobj) -> Qobj:
    """
    Converts a Qobj representing a quantum map to the Choi representation,
    such that the trace of the returned operator is equal to the dimension
    of the system.

    Parameters
    ----------
    q_oper : Qobj
        Superoperator to be converted to Choi representation. If
        ``q_oper`` is ``type="oper"``, then it is taken to act by conjugation,
        such that ``to_choi(A) == to_choi(sprepost(A, A.dag()))``.

    Returns
    -------
    choi : Qobj
        A quantum object representing the same map as ``q_oper``, such that
        ``choi.superrep == "choi"``.

    Raises
    ------
    TypeError:
        If the given quantum object is not a map, or cannot be converted to
        Choi representation.
    """
    if q_oper.type == 'super':
        if q_oper.superrep == 'choi':
            return q_oper
        if q_oper.superrep == 'super':
            return _super_tofrom_choi(q_oper)
        if q_oper.superrep == 'chi':
            return _chi_to_choi(q_oper)
        else:
            raise TypeError(q_oper.superrep)
    elif q_oper.type == 'oper':
        return _super_tofrom_choi(sprepost(q_oper, q_oper.dag()))
    else:
        raise TypeError(
            "Conversion of Qobj with type = {0.type} "
            "and superrep = {0.choi} to Choi not supported.".format(q_oper)
        )


def to_chi(q_oper: Qobj) -> Qobj:
    """
    Converts a Qobj representing a quantum map to a representation as a chi
    (process) matrix in the Pauli basis, such that the trace of the returned
    operator is equal to the dimension of the system.

    Parameters
    ----------
    q_oper : Qobj
        Superoperator to be converted to Chi representation. If
        ``q_oper`` is ``type="oper"``, then it is taken to act by conjugation,
        such that ``to_chi(A) == to_chi(sprepost(A, A.dag()))``.

    Returns
    -------
    chi : Qobj
        A quantum object representing the same map as ``q_oper``, such that
        ``chi.superrep == "chi"``.

    Raises
    ------
    TypeError:
        If the given quantum object is not a map, or cannot be converted
        to Chi representation.
    """
    if q_oper.type == 'super':
        if q_oper.superrep == 'chi':
            return q_oper
        elif q_oper.superrep == 'choi':
            return _choi_to_chi(q_oper)
        elif q_oper.superrep == 'super':
            return _choi_to_chi(to_choi(q_oper))
        else:
            raise TypeError(q_oper.superrep)
    elif q_oper.type == 'oper':
        return to_chi(sprepost(q_oper, q_oper.dag()))
    else:
        raise TypeError(
            "Conversion of Qobj with type = {0.type} "
            "and superrep = {0.choi} to Choi not supported.".format(q_oper)
        )


def to_super(q_oper: Qobj) -> Qobj:
    """
    Converts a Qobj representing a quantum map to the supermatrix (Liouville)
    representation.

    Parameters
    ----------
    q_oper : Qobj
        Superoperator to be converted to supermatrix representation. If
        ``q_oper`` is ``type="oper"``, then it is taken to act by conjugation,
        such that ``to_super(A) == sprepost(A, A.dag())``.

    Returns
    -------
    superop : Qobj
        A quantum object representing the same map as ``q_oper``, such that
        ``superop.superrep == "super"``.

    Raises
    ------
    TypeError
        If the given quantum object is not a map, or cannot be converted
        to supermatrix representation.
    """
    if q_oper.type == 'super':
        if q_oper.superrep == "super":
            return q_oper
        elif q_oper.superrep == 'choi':
            return _super_tofrom_choi(q_oper)
        elif q_oper.superrep == 'chi':
            return to_super(to_choi(q_oper))
        else:
            raise ValueError(
                "Unrecognized superrep '{}'.".format(q_oper.superrep))
    elif q_oper.type == 'oper':  # Assume unitary
        return sprepost(q_oper, q_oper.dag())
    else:
        raise TypeError(
            "Conversion of Qobj with type = {0.type} "
            "and superrep = {0.superrep} to supermatrix not "
            "supported.".format(q_oper)
        )


def to_kraus(q_oper: Qobj, tol: float=1e-9) -> list[Qobj]:
    """
    Converts a Qobj representing a quantum map to a list of quantum objects,
    each representing an operator in the Kraus decomposition of the given map.

    Parameters
    ----------
    q_oper : Qobj
        Superoperator to be converted to Kraus representation. If
        ``q_oper`` is ``type="oper"``, then it is taken to act by conjugation,
        such that ``to_kraus(A) == to_kraus(sprepost(A, A.dag())) == [A]``.

    tol : Float, default: 1e-9
        Optional threshold parameter for eigenvalues/Kraus ops to be discarded.

    Returns
    -------
    kraus_ops : list of Qobj
        A list of quantum objects, each representing a Kraus operator in the
        decomposition of ``q_oper``.

    Raises
    ------
    TypeError: if the given quantum object is not a map, or cannot be
        decomposed into Kraus operators.
    """
    if q_oper.issuper:
        if q_oper.superrep != 'choi':
            q_oper = to_choi(q_oper)
        return _choi_to_kraus(q_oper, tol)
    elif q_oper.isoper:  # Assume unitary
        return [q_oper]
    raise TypeError(
        "Conversion of Qobj with type={0.type} "
        "and superrep={0.superrep} to Kraus decomposition not "
        "supported.".format(q_oper)
    )


def to_stinespring(q_oper: Qobj, threshold: float=1e-10) -> tuple[Qobj, Qobj]:
    r"""
    Converts a Qobj representing a quantum map :math:`\Lambda` to a pair of
    partial isometries ``A`` and ``B`` such that
    :math:`\Lambda(X) = \Tr_2(A X B^\dagger)` for all inputs ``X``, where the
    partial trace is taken over a a new index on the output dimensions of
    ``A`` and ``B``.

    For completely positive inputs, ``A`` will always equal ``B`` up to
    precision errors.

    Parameters
    ----------
    q_oper : Qobj
        Superoperator to be converted to a Stinespring pair.

    threshold : float, default: 1e-10
        Threshold parameter for eigenvalues/Kraus ops to be discarded.

    Returns
    -------
    A, B : Qobj
        Quantum objects representing each of the Stinespring matrices for the
        input Qobj.
    """
    return _choi_to_stinespring(to_choi(q_oper), threshold)
