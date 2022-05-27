"""
Module for the creation of composite quantum objects via the tensor product.
"""

__all__ = [
    'tensor', 'super_tensor', 'composite', 'tensor_swap', 'tensor_contract',
    'expand_operator'
]

import numpy as np
from functools import partial
from .operators import qeye
from .qobj import Qobj
from .superoperator import operator_to_vector, reshuffle
from .dimensions import (
    flatten, enumerate_flat, unflatten, deep_remove, dims_to_tensor_shape,
    dims_idxs_to_tensor_idxs
)
from . import data as _data


class _reverse_partial_tensor:
    """ Picklable lambda op: tensor(op, right) """
    def __init__(self, right):
        self.right = right

    def __call__(self, op):
        return tensor(op, self.right)


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
    >>> tensor([sigmax(), sigmax()]) # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]]
    """
    from .cy.qobjevo import QobjEvo
    if not args:
        raise TypeError("Requires at least one input argument")
    if len(args) == 1 and isinstance(args[0], (Qobj, QobjEvo)):
        return args[0].copy()
    if len(args) == 1:
        try:
            args = tuple(args[0])
        except TypeError:
            raise TypeError("requires Qobj or QobjEvo operands") from None
    if not all(isinstance(q, (Qobj, QobjEvo)) for q in args):
        raise TypeError("requires Qobj or QobjEvo operands")
    if any(isinstance(q, QobjEvo) for q in args):
        # First make tensor from pairs only
        if len(args) >= 3:
            return tensor(args[0], tensor(args[1:]))

        left, right = args
        if isinstance(left, Qobj):
            return right.linear_map(partial(tensor, left))
        if isinstance(right, Qobj):
            return left.linear_map(_reverse_partial_tensor(right))
        left_t = left.linear_map(_reverse_partial_tensor(qeye(right.dims[0])))
        right_t = right.linear_map(partial(tensor, qeye(left.dims[1])))
        return left_t @ right_t

    if not all(q.superrep == args[0].superrep for q in args[1:]):
        raise TypeError("".join([
            "In tensor products of superroperators,",
            " all must have the same representation"
        ]))
    type = args[0].type
    isherm = args[0]._isherm
    isunitary = args[0]._isunitary
    out_data = args[0].data
    dims_l = [d for arg in args for d in arg.dims[0]]
    dims_r = [d for arg in args for d in arg.dims[1]]
    for arg in args[1:]:
        out_data = _data.kron(out_data, arg.data)
        # If both _are_ Hermitian and/or unitary, then so is the output, but if
        # both _aren't_, then output still can be.
        isherm = (isherm and arg._isherm) or None
        isunitary = (isunitary and arg._isunitary) or None
        if arg.type != type:
            type = None
    return Qobj(out_data,
                dims=[dims_l, dims_r],
                type=type,
                isherm=isherm,
                isunitary=isunitary,
                superrep=args[0].superrep,
                copy=False)


def super_tensor(*args):
    """
    Calculate the tensor product of input superoperators, by tensoring together
    the underlying Hilbert spaces on which each vectorized operator acts.

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
    if all(arg.isoperket for arg in args):

        # Reshuffle the superoperators.
        shuffled_ops = list(map(reshuffle, args))

        # Tensor the result.
        shuffled_tensor = tensor(shuffled_ops)

        # Unshuffle and return.
        out = reshuffle(shuffled_tensor)
        return out

    if all(arg.isoperbra for arg in args):
        return super_tensor(*(arg.dag() for arg in args)).dag()
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
    import qutip.core.superop_reps
    # First step will be to ensure everything is a Qobj at all.
    if not all(isinstance(arg, Qobj) for arg in args):
        raise TypeError("All arguments must be Qobjs.")

    # Next, figure out if we have something oper-like (isoper or issuper),
    # or something ket-like (isket or isoperket). Bra-like we'll deal with
    # by turning things into ket-likes and back.
    if all(map(_isoperlike, args)):
        if any(arg.issuper for arg in args):
            # to_super will promote 'oper' and leave 'super' untouched
            return super_tensor(*map(qutip.core.superop_reps.to_super, args))
        return tensor(*args)
    if all(map(_isketlike, args)):
        if any(arg.isoperket for arg in args):
            return super_tensor(*(
                arg if arg.isoperket else operator_to_vector(arg.proj())
                for arg in args
            ))
        return tensor(*args)
    if all(map(_isbralike, args)):
        # Turn into ket-likes and recurse.
        return composite(*(arg.dag() for arg in args)).dag()
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
    sl = tuple(slice(None, None, None) if idx not in (i, j) else idxs
               for idx in range(arr.ndim))
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
        # axis_idxs.index effectively evaluates the mapping from original index
        # labels to the labels after contraction.
        arr = _tensor_contract_single(arr, *map(axis_idxs.index, pair))
        axis_idxs.remove(pair[0])
        axis_idxs.remove(pair[1])
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
    data = q_oper.full()
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
    return Qobj(data, dims=dims, superrep=q_oper.superrep, copy=False)


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
    qtens = qobj.data.to_array()

    # Reshape by the flattened dims.
    qtens = qtens.reshape(tensor_dims)

    # Contract out the indices from the flattened object.
    # Note that we need to feed pairs through dims_idxs_to_tensor_idxs
    # to ensure that we are contracting the right indices.
    qtens = _tensor_contract_dense(qtens,
                                   *dims_idxs_to_tensor_idxs(dims, pairs))

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
    return Qobj(qmtx, dims=contracted_dims, superrep=qobj.superrep, copy=False)


def _check_oper_dims(oper, dims=None, targets=None):
    """
    Check if the given operator is valid.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        The quantum object to be checked.
    dims : list, optional
        A list of integer for the dimension of each composite system.
        e.g ``[2, 2, 2, 2, 2]`` for 5 qubits system.
    targets : int or list of int, optional
        The indices of subspace that are acted on.
    """
    # if operator matches N
    if not isinstance(oper, Qobj) or oper.dims[0] != oper.dims[1]:
        raise ValueError(
            "The operator is not an "
            "Qobj with the same input and output dimensions.")
    # if operator dims matches the target dims
    if dims is not None and targets is not None:
        targ_dims = [dims[t] for t in targets]
        if oper.dims[0] != targ_dims:
            raise ValueError(
                "The operator dims {} do not match "
                "the target dims {}.".format(
                    oper.dims[0], targ_dims))


def _targets_to_list(targets, oper=None, N=None):
    """
    transform targets to a list and check validity.

    Parameters
    ----------
    targets : int or list of int
        The indices of subspace that are acted on.
    oper : :class:`qutip.Qobj`, optional
        An operator, the type of the :class:`qutip.Qobj`
        has to be an operator
        and the dimension matches the tensored qubit Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    N : int, optional
        The number of subspace in the system.
    """
    # if targets is a list of integer
    if targets is None:
        targets = list(range(len(oper.dims[0])))
    if not hasattr(targets, '__iter__'):
        targets = [targets]
    if not all([isinstance(t, int) for t in targets]):
        raise TypeError(
            "targets should be "
            "an integer or a list of integer")
    # if targets has correct length
    if oper is not None:
        req_num = len(oper.dims[0])
        if len(targets) != req_num:
            raise ValueError(
                "The given operator needs {} "
                "target qutbis, "
                "but {} given.".format(
                    req_num, len(targets)))
    # if targets is smaller than N
    if N is not None:
        if not all([t < N for t in targets]):
            raise ValueError("Targets must be smaller than N={}.".format(N))
    return targets


def expand_operator(oper, targets, dims):
    """
    Expand an operator to one that acts on a system with desired dimensions.
    e.g.
    ```
    expand_operator(oper, 2, [2, 3, 4, 5]) ==
        tensor(qeye(2), qeye(3), oper, qeye(5))
    expand_operator(tensor(oper1, oper2), [2, 0], [2, 3, 4, 5]) ==
        tensor(oper2, qeye(3), oper1, qeye(5))
    ```

    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        An operator that act on the subsystem, has to be an operator and the
        dimension matches the tensored dims Hilbert space
        e.g. oper.dims = ``[[2, 3], [2, 3]]``
    targets : int or list of int
        The indices of subspace that are acted on.
    dims : list
        A list of integer for the dimension of each composite system.
        E.g ``[2, 3, 2, 3, 4]``.

    Returns
    -------
    expanded_oper : :class:`qutip.Qobj`
        The expanded operator acting on a system with desired dimension.
    """
    from .operators import identity
    N = len(dims)
    targets = _targets_to_list(targets, oper=oper, N=N)
    _check_oper_dims(oper, dims=dims, targets=targets)

    # Generate the correct order for permutation,
    # eg. if N = 5, targets = [3,0], the order is [1,2,3,0,4].
    # If the operator is cnot,
    # this order means that the 3rd qubit controls the 0th qubit.
    new_order = [0] * N
    for i, t in enumerate(targets):
        new_order[t] = i
    # allocate the rest qutbits (not targets) to the empty
    # position in new_order
    rest_pos = [q for q in list(range(N)) if q not in targets]
    rest_qubits = list(range(len(targets), N))
    for i, ind in enumerate(rest_pos):
        new_order[ind] = rest_qubits[i]
    id_list = [identity(dims[i]) for i in rest_pos]
    return tensor([oper] + id_list).permute(new_order)
