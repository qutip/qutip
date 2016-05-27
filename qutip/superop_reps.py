# -*- coding: utf-8 -*-
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
#
# This module was initially contributed by Ben Criger.
#
"""
This module implements transformations between superoperator representations,
including supermatrix, Kraus, Choi and Chi (process) matrix formalisms.
"""

__all__ = ['super_to_choi', 'choi_to_super', 'choi_to_kraus', 'kraus_to_choi',
           'kraus_to_super', 'choi_to_chi', 'chi_to_choi', 'to_choi',
           'to_chi', 'to_super', 'to_kraus', 'to_stinespring'
           ]

# Python Standard Library
from itertools import starmap, product

# NumPy/SciPy
from numpy.core.multiarray import array, zeros
from numpy.core.shape_base import hstack
from numpy.matrixlib.defmatrix import matrix
from numpy import sqrt, floor, log2
from numpy import dot
from scipy.linalg import eig, svd
# Needed to avoid conflict with itertools.product.
import numpy as np

# Other QuTiP functions and classes
from qutip.superoperator import vec2mat, spre, spost, operator_to_vector
from qutip.operators import identity, sigmax, sigmay, sigmaz
from qutip.tensor import tensor, flatten
from qutip.qobj import Qobj
from qutip.states import basis


# SPECIFIC SUPEROPERATORS -----------------------------------------------------

def _dep_super(pe):
    """
    Returns the superoperator corresponding to qubit depolarization for a
    given parameter pe.

    TODO: if this is going into production (hopefully it isn't) then check
    CPTP, expand to arbitrary dimensional systems, etc.
    """
    return Qobj(dims=[[[2], [2]], [[2], [2]]],
                inpt=array([[1. - pe / 2., 0., 0., pe / 2.],
                            [0., 1. - pe, 0., 0.],
                            [0., 0., 1. - pe, 0.],
                            [pe / 2., 0., 0., 1. - pe / 2.]]))


def _dep_choi(pe):
    """
    Returns the choi matrix corresponding to qubit depolarization for a
    given parameter pe.

    TODO: if this is going into production (hopefully it isn't) then check
    CPTP, expand to arbitrary dimensional systems, etc.
    """
    return Qobj(dims=[[[2], [2]], [[2], [2]]],
                inpt=array([[1. - pe / 2., 0., 0., 1. - pe],
                            [0., pe / 2., 0., 0.],
                            [0., 0., pe / 2., 0.],
                            [1. - pe, 0., 0., 1. - pe / 2.]]),
                superrep='choi')


# CHANGE OF BASIS FUNCTIONS ---------------------------------------------------
# These functions find change of basis matrices, and are useful in converting
# between (for instance) Choi and chi matrices. At some point, these should
# probably be moved out to another module.

_SINGLE_QUBIT_PAULI_BASIS = (identity(2), sigmax(), sigmay(), sigmaz())


def _pauli_basis(nq=1):
    # NOTE: This is slow as can be.
    # TODO: Make this sparse. CSR format was causing problems for the [idx, :]
    #       slicing below.
    B = zeros((4 ** nq, 4 ** nq), dtype=complex)
    dims = [[[2] * nq] * 2] * 2

    for idx, op in enumerate(starmap(tensor,
                                     product(_SINGLE_QUBIT_PAULI_BASIS,
                                             repeat=nq))):
        B[:, idx] = operator_to_vector(op).dag().data.todense()

    return Qobj(B, dims=dims)


# PRIVATE CONVERSION FUNCTIONS ------------------------------------------------
# These functions handle the main work of converting between representations,
# and are exposed below by other functions that add postconditions about types.
#
# TODO: handle type='kraus' as a three-index Qobj, rather than as a list?

def _super_tofrom_choi(q_oper):
    """
    We exploit that the basis transformation between Choi and supermatrix
    representations squares to the identity, so that if we munge Qobj.type,
    we can use the same function.

    Since this function doesn't respect :attr:`Qobj.type`, we mark it as
    private; only those functions which wrap this in a way so as to preserve
    type should be called externally.
    """
    data = q_oper.data.toarray()
    sqrt_shape = int(sqrt(data.shape[0]))
    return Qobj(dims=q_oper.dims,
                inpt=data.reshape([sqrt_shape] * 4).
                transpose(3, 1, 2, 0).reshape(q_oper.data.shape))

def _isqubitdims(dims):
    """Checks whether all entries in a dims list are integer powers of 2.

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
    return all([
        2**floor(log2(dim)) == dim
        for dim in flatten(dims)
    ])


def _super_to_superpauli(q_oper):
    """
    Converts a superoperator in the column-stacking basis to
    the Pauli basis (assuming qubit dimensions).

    This is an internal function, as QuTiP does not currently have
    a way to mark that superoperators are represented in the Pauli
    basis as opposed to the column-stacking basis; a Pauli-basis
    ``type='super'`` would thus break other conversion functions.
    """
    # Ensure we start with a column-stacking-basis superoperator.
    sqobj = to_super(q_oper)
    if not _isqubitdims(sqobj.dims):
        raise ValueError("Pauli basis is only defined for qubits.")
    nq = int(log2(sqobj.shape[0]) / 2)
    B = _pauli_basis(nq) / sqrt(2**nq)
    # To do this, we have to hack a bit and force the dims to match,
    # since the _pauli_basis function makes different assumptions
    # about indices than we need here.
    B.dims = sqobj.dims
    return (B.dag() * sqobj * B)


def super_to_choi(q_oper):
    # TODO: deprecate and make private in favor of to_choi,
    # which looks at Qobj.type to determine the right conversion function.
    """
    Takes a superoperator to a Choi matrix
    TODO: Sanitize input, incorporate as method on Qobj if type=='super'
    """
    q_oper = _super_tofrom_choi(q_oper)
    q_oper.superrep = 'choi'
    return q_oper


def choi_to_super(q_oper):
    # TODO: deprecate and make private in favor of to_super,
    # which looks at Qobj.type to determine the right conversion function.
    """
    Takes a Choi matrix to a superoperator
    TODO: Sanitize input, Abstract-ify application of channels to states
    """
    q_oper = super_to_choi(q_oper)
    q_oper.superrep = 'super'
    return q_oper


def choi_to_kraus(q_oper):
    """
    Takes a Choi matrix and returns a list of Kraus operators.
    TODO: Create a new class structure for quantum channels, perhaps as a
    strict sub-class of Qobj.
    """
    vals, vecs = eig(q_oper.data.todense())
    vecs = [array(_) for _ in zip(*vecs)]
    return [Qobj(inpt=sqrt(val)*vec2mat(vec)) for val, vec in zip(vals, vecs)]


def kraus_to_choi(kraus_list):
    """
    Takes a list of Kraus operators and returns the Choi matrix for the channel
    represented by the Kraus operators in `kraus_list`
    """
    kraus_mat_list = list(map(lambda x: matrix(x.data.todense()), kraus_list))
    op_len = len(kraus_mat_list[0])
    op_rng = range(op_len)
    choi_blocks = array([[sum([op[:, c_ix] * array([op.H[r_ix, :]])
                               for op in kraus_mat_list])
                          for r_ix in op_rng]
                         for c_ix in op_rng])
    return Qobj(inpt=hstack(hstack(choi_blocks)),
                dims=[kraus_list[0].dims, kraus_list[0].dims], type='super',
                superrep='choi')


def kraus_to_super(kraus_list):
    """
    Converts a list of Kraus operators and returns a super operator.
    """
    return choi_to_super(kraus_to_choi(kraus_list))


def _nq(dims):
    dim = np.product(dims[0][0])
    nq = int(log2(dim))
    if 2 ** nq != dim:
        raise ValueError("{} is not an integer power of 2.".format(dim))
    return nq


def choi_to_chi(q_oper):
    """
    Converts a Choi matrix to a Chi matrix in the Pauli basis.

    NOTE: this is only supported for qubits right now. Need to extend to
    Heisenberg-Weyl for other subsystem dimensions.
    """
    nq = _nq(q_oper.dims)
    B = _pauli_basis(nq)
    # Force the basis change to match the dimensions of
    # the input.
    B.dims = q_oper.dims
    B.superrep = 'choi'

    return Qobj(B.dag() * q_oper * B, superrep='chi')


def chi_to_choi(q_oper):
    """
    Converts a Choi matrix to a Chi matrix in the Pauli basis.

    NOTE: this is only supported for qubits right now. Need to extend to
    Heisenberg-Weyl for other subsystem dimensions.
    """
    nq = _nq(q_oper.dims)
    B = _pauli_basis(nq)
    # Force the basis change to match the dimensions of
    # the input.
    B.dims = q_oper.dims

    # We normally should not multiply objects of different
    # superreps, so Qobj warns about that. Here, however, we're actively
    # converting between, so the superrep of B is irrelevant.
    # To suppress warnings, we pretend that B is also a chi.
    B.superrep = 'chi'

    # The Chi matrix has tr(chi) == d², so we need to divide out
    # by that to get back to the Choi form.
    return Qobj((B * q_oper * B.dag()) / q_oper.shape[0], superrep='choi')

def _svd_u_to_kraus(U, S, d, dK, indims, outdims):
    """
    Given a partial isometry U and a vector of square-roots of singular values S
    obtained from an SVD, produces the Kraus operators represented by U.

    Returns
    -------
    Ks : list of Qobj
        Quantum objects represnting each of the Kraus operators.
    """
    # We use U * S since S is 1-index, such that this is equivalent to
    # U . diag(S), but easier to write down.
    Ks = list(map(Qobj, array(U * S).reshape((d, d, dK), order='F').transpose((2, 0, 1))))
    for K in Ks:
        K.dims = [outdims, indims]
    return Ks


def _generalized_kraus(q_oper, thresh=1e-10):
    # TODO: document!
    # TODO: use this to generalize to_kraus to the case where U != V.
    #       This is critical for non-CP maps, as appear in (for example)
    #       diamond norm differences between two CP maps.
    if q_oper.type != "super" or q_oper.superrep != "choi":
        raise ValueError("Expected a Choi matrix, got a {} (superrep {}).".format(q_oper.type, q_oper.superrep))
    
    # Remember the shape of the underlying space,
    # as we'll need this to make Kraus operators later.
    dL, dR = map(int, map(sqrt, q_oper.shape))
    # Also remember the dims breakout.
    out_dims, in_dims = q_oper.dims
    out_left, out_right = out_dims
    in_left, in_right = in_dims

    # Find the SVD.
    U, S, V = svd(q_oper.data.todense())

    # Truncate away the zero singular values, up to a threshold.
    nonzero_idxs = S > thresh
    dK = nonzero_idxs.sum()
    U = array(U)[:, nonzero_idxs]
    # We also want S to be a single index array, which np.matrix
    # doesn't allow for. This is stripped by calling array() on it.
    S = sqrt(array(S)[nonzero_idxs])
    # Since NumPy returns V and not V+, we need to take the dagger
    # to get back to quantum info notation for Stinespring pairs.
    V = array(V.conj().T)[:, nonzero_idxs]

    # Next, we convert each of U and V into Kraus operators.
    # Finally, we want the Kraus index to be left-most so that we
    # can map over it when making Qobjs.
    # FIXME: does not preserve dims!
    kU = _svd_u_to_kraus(U, S, dL, dK, out_right, out_left)
    kV = _svd_u_to_kraus(V, S, dL, dK, in_right, in_left)

    return kU, kV


def choi_to_stinespring(q_oper, thresh=1e-10):
    # TODO: document!
    kU, kV = _generalized_kraus(q_oper, thresh=thresh)

    assert(len(kU) == len(kV))
    dK = len(kU)
    dL = kU[0].shape[0]
    dR = kV[0].shape[1]
    # Also remember the dims breakout.
    out_dims, in_dims = q_oper.dims
    out_left, out_right = out_dims
    in_left, in_right = in_dims

    A = Qobj(zeros((dK * dL, dL)), dims=[out_left + [dK], out_right + [1]])
    B = Qobj(zeros((dK * dR, dR)), dims=[in_left + [dK], in_right + [1]])

    for idx_kraus, (KL, KR) in enumerate(zip(kU, kV)):
        A += tensor(KL, basis(dK, idx_kraus))
        B += tensor(KR, basis(dK, idx_kraus))
        
    # There is no input (right) Kraus index, so strip that off.
    del A.dims[1][-1]
    del B.dims[1][-1]

    return A, B

# PUBLIC CONVERSION FUNCTIONS -------------------------------------------------
# These functions handle superoperator conversions in a way that preserves the
# correctness of Qobj.type, and in a way that automatically branches based on
# the input Qobj.type.

def to_choi(q_oper):
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
    TypeError: if the given quantum object is not a map, or cannot be converted
        to Choi representation.
    """
    if q_oper.type == 'super':
        if q_oper.superrep == 'choi':
            return q_oper
        if q_oper.superrep == 'super':
            return super_to_choi(q_oper)
        if q_oper.superrep == 'chi':
            return chi_to_choi(q_oper)
        else:
            raise TypeError(q_oper.superrep)
    elif q_oper.type == 'oper':
        return super_to_choi(spre(q_oper) * spost(q_oper.dag()))
    else:
        raise TypeError(
            "Conversion of Qobj with type = {0.type} "
            "and superrep = {0.choi} to Choi not supported.".format(q_oper)
        )


def to_chi(q_oper):
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
    TypeError: if the given quantum object is not a map, or cannot be converted
        to Chi representation.
    """
    if q_oper.type == 'super':
        # Case 1: Already done.
        if q_oper.superrep == 'chi':
            return q_oper
        # Case 2: Can directly convert.
        elif q_oper.superrep == 'choi':
            return choi_to_chi(q_oper)
        # Case 3: Need to go through Choi.
        elif q_oper.superrep == 'super':
            return to_chi(to_choi(q_oper))
        else:
            raise TypeError(q_oper.superrep)
    elif q_oper.type == 'oper':
        return to_chi(spre(q_oper) * spost(q_oper.dag()))
    else:
        raise TypeError(
            "Conversion of Qobj with type = {0.type} "
            "and superrep = {0.choi} to Choi not supported.".format(q_oper)
        )


def to_super(q_oper):
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
        # Case 1: Already done.
        if q_oper.superrep == "super":
            return q_oper
        # Case 2: Can directly convert.
        elif q_oper.superrep == 'choi':
            return choi_to_super(q_oper)
        # Case 3: Need to go through Choi.
        elif q_oper.superrep == 'chi':
            return to_super(to_choi(q_oper))
        # Case 4: Something went wrong.
        else:
            raise ValueError(
                "Unrecognized superrep '{}'.".format(q_oper.superrep))
    elif q_oper.type == 'oper':  # Assume unitary
        return spre(q_oper) * spost(q_oper.dag())
    else:
        raise TypeError(
            "Conversion of Qobj with type = {0.type} "
            "and superrep = {0.superrep} to supermatrix not "
            "supported.".format(q_oper)
        )


def to_kraus(q_oper):
    """
    Converts a Qobj representing a quantum map to a list of quantum objects,
    each representing an operator in the Kraus decomposition of the given map.

    Parameters
    ----------
    q_oper : Qobj
        Superoperator to be converted to Kraus representation. If
        ``q_oper`` is ``type="oper"``, then it is taken to act by conjugation,
        such that ``to_kraus(A) == to_kraus(sprepost(A, A.dag())) == [A]``.

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
    if q_oper.type == 'super':
        if q_oper.superrep in ("super", "chi"):
            return to_kraus(to_choi(q_oper))
        elif q_oper.superrep == 'choi':
            return choi_to_kraus(q_oper)
    elif q_oper.type == 'oper':  # Assume unitary
        return [q_oper]
    else:
        raise TypeError(
            "Conversion of Qobj with type = {0.type} "
            "and superrep = {0.superrep} to Kraus decomposition not "
            "supported.".format(q_oper)
        )

def to_stinespring(q_oper):
    r"""
    Converts a Qobj representing a quantum map $\Lambda$ to a pair of partial isometries
    $A$ and $B$ such that $\Lambda(X) = \Tr_2(A X B^\dagger)$ for all inputs $X$, where
    the partial trace is taken over a a new index on the output dimensions of $A$ and $B$.

    For completely positive inputs, $A$ will always equal $B$ up to precision errors.

    Parameters
    ----------
    q_oper : Qobj
        Superoperator to be converted to a Stinespring pair.

    Returns
    -------
    A, B : Qobj
        Quantum objects representing each of the Stinespring matrices for the input Qobj.
    """
    return choi_to_stinespring(to_choi(q_oper))
