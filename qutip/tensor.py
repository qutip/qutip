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
Module for the creation of composite quantum objects via the tensor product.
"""

__all__ = [
    'tensor', 'super_tensor', 'composite', 'tensor_swap', 'tensor_contract'
]

import numpy as np
import scipy.sparse as sp
from qutip.cy.spmath import zcsr_kron
from qutip.qobj import Qobj
from qutip.permute import reshuffle
from qutip.superoperator import operator_to_vector
from qutip.dimensions import (
    flatten, enumerate_flat, unflatten, deep_remove,
    dims_to_tensor_shape, dims_idxs_to_tensor_idxs
)

import qutip.settings
import qutip.superop_reps  # Avoid circular dependency here.


def tensor(*args):
    """Calculates the tensor product of input operators.

    Parameters
    ----------
    args : array_like
        ``list`` or ``array`` of quantum objects for tensor product.

    Returns
    -------
    obj : qobj
        A composite quantum object.

    Examples
    --------
    >>> tensor([sigmax(), sigmax()])
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]]
    """

    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        # this is the case when tensor is called on the form:
        # tensor([q1, q2, q3, ...])
        qlist = args[0]

    elif len(args) == 1 and isinstance(args[0], Qobj):
        # tensor is called with a single Qobj as an argument, do nothing
        return args[0]

    else:
        # this is the case when tensor is called on the form:
        # tensor(q1, q2, q3, ...)
        qlist = args

    if not all([isinstance(q, Qobj) for q in qlist]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")

    out = Qobj()

    if qlist[0].issuper:
        out.superrep = qlist[0].superrep
        if not all([q.superrep == out.superrep for q in qlist]):
            raise TypeError("In tensor products of superroperators, all must" +
                            "have the same representation")

    out.isherm = True
    for n, q in enumerate(qlist):
        if n == 0:
            out.data = q.data
            out.dims = q.dims
        else:
            out.data  = zcsr_kron(out.data, q.data)
            
            out.dims = [out.dims[0] + q.dims[0], out.dims[1] + q.dims[1]]

        out.isherm = out.isherm and q.isherm

    if not out.isherm:
        out._isherm = None

    return out.tidyup() if qutip.settings.auto_tidyup else out


def super_tensor(*args):
    """Calculates the tensor product of input superoperators, by tensoring
    together the underlying Hilbert spaces on which each vectorized operator
    acts.

    Parameters
    ----------
    args : array_like
        ``list`` or ``array`` of quantum objects with ``type="super"``.

    Returns
    -------
    obj : qobj
        A composite quantum object.

    """
    if isinstance(args[0], list):
        args = args[0]

    # Check if we're tensoring vectors or superoperators.
    if all(arg.issuper for arg in args):
        if not all(arg.superrep == "super" for arg in args):
            raise TypeError(
                "super_tensor on type='super' is only implemented for "
                "superrep='super'."
            )

        # Reshuffle the superoperators.
        shuffled_ops = list(map(reshuffle, args))

        # Tensor the result.
        shuffled_tensor = tensor(shuffled_ops)

        # Unshuffle and return.
        out = reshuffle(shuffled_tensor)
        out.superrep = args[0].superrep
        return out

    elif all(arg.isoperket for arg in args):

        # Reshuffle the superoperators.
        shuffled_ops = list(map(reshuffle, args))

        # Tensor the result.
        shuffled_tensor = tensor(shuffled_ops)

        # Unshuffle and return.
        out = reshuffle(shuffled_tensor)
        return out

    elif all(arg.isoperbra for arg in args):
        return super_tensor(*(arg.dag() for arg in args)).dag()

    else:
        raise TypeError(
            "All arguments must be the same type, "
            "either super, operator-ket or operator-bra."
        )


def _isoperlike(q):
    return q.isoper or q.issuper


def _isketlike(q):
    return q.isket or q.isoperket


def _isbralike(q):
    return q.isbra or q.isoperbra


def composite(*args):
    """
    Given two or more operators, kets or bras, returns the Qobj
    corresponding to a composite system over each argument.
    For ordinary operators and vectors, this is the tensor product,
    while for superoperators and vectorized operators, this is
    the column-reshuffled tensor product.

    If a mix of Qobjs supported on Hilbert and Liouville spaces
    are passed in, the former are promoted. Ordinary operators
    are assumed to be unitaries, and are promoted using ``to_super``,
    while kets and bras are promoted by taking their projectors and
    using ``operator_to_vector(ket2dm(arg))``.
    """
    # First step will be to ensure everything is a Qobj at all.
    if not all(isinstance(arg, Qobj) for arg in args):
        raise TypeError("All arguments must be Qobjs.")

    # Next, figure out if we have something oper-like (isoper or issuper),
    # or something ket-like (isket or isoperket). Bra-like we'll deal with
    # by turning things into ket-likes and back.
    if all(map(_isoperlike, args)):
        # OK, we have oper/supers.
        if any(arg.issuper for arg in args):
            # Note that to_super does nothing to things
            # that are already type=super, while it will
            # promote unitaries to superunitaries.
            return super_tensor(*map(qutip.superop_reps.to_super, args))

        else:
            # Everything's just an oper, so ordinary tensor products work.
            return tensor(*args)

    elif all(map(_isketlike, args)):
        # Ket-likes.
        if any(arg.isoperket for arg in args):
            # We have a vectorized operator, we we may need to promote
            # something.
            return super_tensor(*(
                arg if arg.isoperket
                else operator_to_vector(qutip.states.ket2dm(arg))
                for arg in args
            ))

        else:
            # Everything's ordinary, so we can use the tensor product here.
            return tensor(*args)

    elif all(map(_isbralike, args)):
        # Turn into ket-likes and recurse.
        return composite(*(arg.dag() for arg in args)).dag()

    else:
        raise TypeError("Unsupported Qobj types [{}].".format(
            ", ".join(arg.type for arg in args)
        ))


def _tensor_contract_single(arr, i, j):
    """
    Contracts a dense tensor along a single index pair.
    """
    if arr.shape[i] != arr.shape[j]:
        raise ValueError("Cannot contract over indices of different length.")
    idxs = np.arange(arr.shape[i])
    sl = tuple(slice(None, None, None)
               if idx not in (i, j) else idxs for idx in range(arr.ndim))
    contract_at = i if j == i + 1 else 0
    return np.sum(arr[sl], axis=contract_at)


def _tensor_contract_dense(arr, *pairs):
    """
    Contracts a dense tensor along one or more index pairs,
    keeping track of how the indices are relabeled by the removal
    of other indices.
    """
    axis_idxs = list(range(arr.ndim))
    for pair in pairs:
        # axis_idxs.index effectively evaluates the mapping from
        # original index labels to the labels after contraction.
        arr = _tensor_contract_single(arr, *map(axis_idxs.index, pair))
        list(map(axis_idxs.remove, pair))
    return arr


def tensor_swap(q_oper, *pairs):
    """Transposes one or more pairs of indices of a Qobj.
    Note that this uses dense representations and thus
    should *not* be used for very large Qobjs.

    Parameters
    ----------

    pairs : tuple
        One or more tuples ``(i, j)`` indicating that the
        ``i`` and ``j`` dimensions of the original qobj
        should be swapped.

    Returns
    -------

    sqobj : Qobj
        The original Qobj with all named index pairs swapped with each other
    """
    dims = q_oper.dims
    tensor_pairs = dims_idxs_to_tensor_idxs(dims, pairs)

    data = q_oper.data.toarray()

    # Reshape into tensor indices
    data = data.reshape(dims_to_tensor_shape(dims))

    # Now permute the dims list so we know how to get back.
    flat_dims = flatten(dims)
    perm = list(range(len(flat_dims)))
    for i, j in pairs:
        flat_dims[i], flat_dims[j] = flat_dims[j], flat_dims[i]
    for i, j in tensor_pairs:
        perm[i], perm[j] = perm[j], perm[i]
    dims = unflatten(flat_dims, enumerate_flat(dims))

    # Next, permute the actual indices of the dense tensor.
    data = data.transpose(perm)

    # Reshape back, using the left and right of dims.
    data = data.reshape(list(map(np.prod, dims)))

    return Qobj(inpt=data, dims=dims, superrep=q_oper.superrep)


def tensor_contract(qobj, *pairs):
    """Contracts a qobj along one or more index pairs.
    Note that this uses dense representations and thus
    should *not* be used for very large Qobjs.

    Parameters
    ----------

    pairs : tuple
        One or more tuples ``(i, j)`` indicating that the
        ``i`` and ``j`` dimensions of the original qobj
        should be contracted.

    Returns
    -------

    cqobj : Qobj
        The original Qobj with all named index pairs contracted
        away.

    """
    # Record and label the original dims.
    dims = qobj.dims
    dims_idxs = enumerate_flat(dims)
    tensor_dims = dims_to_tensor_shape(dims)

    # Convert to dense first, since sparse won't support the reshaping we need.
    qtens = qobj.data.toarray()

    # Reshape by the flattened dims.
    qtens = qtens.reshape(tensor_dims)

    # Contract out the indices from the flattened object.
    # Note that we need to feed pairs through dims_idxs_to_tensor_idxs
    # to ensure that we are contracting the right indices.
    qtens = _tensor_contract_dense(qtens, *dims_idxs_to_tensor_idxs(dims, pairs))

    # Remove the contracted indexes from dims so we know how to
    # reshape back.
    # This concerns dims, and not the tensor indices, so we need
    # to make sure to use the original dims indices and not the ones
    # generated by dims_to_* functions.
    contracted_idxs = deep_remove(dims_idxs, *flatten(list(map(list, pairs))))
    contracted_dims = unflatten(flatten(dims), contracted_idxs)

    # We don't need to check for tensor idxs versus dims idxs here,
    # as column- versus row-stacking will never move an index for the
    # vectorized operator spaces all the way from the left to the right.
    l_mtx_dims, r_mtx_dims = map(np.product, map(flatten, contracted_dims))

    # Reshape back into a 2D matrix.
    qmtx = qtens.reshape((l_mtx_dims, r_mtx_dims))

    # Return back as a qobj.
    return Qobj(qmtx, dims=contracted_dims, superrep=qobj.superrep)

import qutip.states
