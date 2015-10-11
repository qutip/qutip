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

__all__ = ['tensor', 'super_tensor', 'composite', 'tensor_contract']

import numpy as np
import scipy.sparse as sp

from qutip.qobj import Qobj
from qutip.permute import reshuffle
from qutip.superoperator import operator_to_vector

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
            out.data = sp.kron(out.data, q.data, format='csr')
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


def flatten(l):
    """Flattens a list of lists to the first level.

    Given a list containing a mix of scalars and lists,
    flattens down to a list of the scalars within the original
    list.

    Examples
    --------

    >>> print(flatten([[[0], 1], 2]))
    [0, 1, 2]

    """
    if not isinstance(l, list):
        return [l]
    else:
        return sum(map(flatten, l), [])


def _enumerate_flat(l, idx=0):
    if not isinstance(l, list):
        # Found a scalar, so return and increment.
        return idx, idx + 1
    else:
        # Found a list, so append all the scalars
        # from it and recurse to keep the increment
        # correct.
        acc = []
        for elem in l:
            labels, idx = _enumerate_flat(elem, idx)
            acc.append(labels)
        return acc, idx


def enumerate_flat(l):
    """Labels the indices at which scalars occur in a flattened list.

    Given a list containing a mix of scalars and lists,
    returns a list of the same structure, where each scalar
    has been replaced by an index into the flattened list.

    Examples
    --------

    >>> print(enumerate_flat([[[10], [20, 30]], 40]))
    [[[0], [1, 2]], 3]

    """
    return _enumerate_flat(l)[0]


def deep_remove(l, *what):
    """Removes scalars from all levels of a nested list.

    Given a list containing a mix of scalars and lists,
    returns a list of the same structure, but where one or
    more scalars have been removed.

    Examples
    --------

    >>> print(deep_remove([[[[0, 1, 2]], [3, 4], [5], [6, 7]]], 0, 5))
    [[[[1, 2]], [3, 4], [], [6, 7]]]

    """
    if isinstance(l, list):
        # Make a shallow copy at this level.
        l = l[:]
        for to_remove in what:
            if to_remove in l:
                l.remove(to_remove)
            else:
                l = list(map(lambda elem: deep_remove(elem, to_remove), l))
    return l


def unflatten(l, idxs):
    """Unflattens a list by a given structure.

    Given a list of scalars and a deep list of indices
    as produced by `flatten`, returns an "unflattened"
    form of the list. This perfectly inverts `flatten`.

    Examples
    --------

    >>> l = [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]]
    >>> idxs = enumerate_flat(l)
    >>> print(unflatten(flatten(l)), idxs) == l
    True

    """
    acc = []
    for idx in idxs:
        if isinstance(idx, list):
            acc.append(unflatten(l, idx))
        else:
            acc.append(l[idx])
    return acc


def _tensor_contract_single(arr, i, j):
    """
    Contracts a dense tensor along a single index pair.
    """
    if arr.shape[i] != arr.shape[j]:
        raise ValueError("Cannot contract over indices of different length.")
    idxs = np.arange(arr.shape[i])
    sl = tuple(slice(None, None, None)
               if idx not in (i, j) else idxs for idx in range(arr.ndim))
    return np.sum(arr[sl], axis=0)


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
    flat_dims = flatten(dims)

    # Convert to dense first, since sparse won't support the reshaping we need.
    qtens = qobj.data.toarray()

    # Reshape by the flattened dims.
    qtens = qtens.reshape(flat_dims)

    # Contract out the indices from the flattened object.
    qtens = _tensor_contract_dense(qtens, *pairs)

    # Remove the contracted indexes from dims so we know how to
    # reshape back.
    contracted_idxs = deep_remove(dims_idxs, *flatten(list(map(list, pairs))))
    contracted_dims = unflatten(flat_dims, contracted_idxs)

    l_mtx_dims, r_mtx_dims = map(np.product, contracted_dims)

    # Reshape back into a 2D matrix.
    qmtx = qtens.reshape((l_mtx_dims, r_mtx_dims))

    # Return back as a qobj.
    return Qobj(qmtx, dims=contracted_dims, superrep=qobj.superrep)

import qutip.states
