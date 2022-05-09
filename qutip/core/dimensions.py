"""
Internal use module for manipulating dims specifications.
"""

# Everything should be explicitly imported, not made available by default.
__all__ = ['expand_operator']

import numpy as np
from operator import getitem
from functools import partial
from .tensor import tensor
from .operators import identity


def is_scalar(dims):
    """
    Returns True if a dims specification is effectively
    a scalar (has dimension 1).
    """
    return np.prod(flatten(dims)) == 1


def is_vector(dims):
    return (
        isinstance(dims, list) and
        isinstance(dims[0], (int, np.integer))
    )


def is_vectorized_oper(dims):
    return (
        isinstance(dims, list) and
        isinstance(dims[0], list)
    )


def type_from_dims(dims, enforce_square=False):
    bra_like, ket_like = map(is_scalar, dims)

    if bra_like:
        if is_vector(dims[1]):
            return 'bra'
        elif is_vectorized_oper(dims[1]):
            return 'operator-bra'

    if ket_like:
        if is_vector(dims[0]):
            return 'ket'
        elif is_vectorized_oper(dims[0]):
            return 'operator-ket'

    elif is_vector(dims[0]) and (dims[0] == dims[1] or not enforce_square):
        return 'oper'

    elif (
        is_vectorized_oper(dims[0])
        and (
            not enforce_square
            or (dims[0] == dims[1] and dims[0][0] == dims[1][0])
        )
    ):
        return 'super'
    return 'other'


def flatten(l):
    """Flattens a list of lists to the first level.

    Given a list containing a mix of scalars and lists,
    flattens down to a list of the scalars within the original
    list.

    Examples
    --------

    >>> flatten([[[0], 1], 2]) # doctest: +SKIP
    [0, 1, 2]

    """
    if not isinstance(l, list):
        return [l]
    else:
        return sum(map(flatten, l), [])


def deep_remove(l, *what):
    """Removes scalars from all levels of a nested list.

    Given a list containing a mix of scalars and lists,
    returns a list of the same structure, but where one or
    more scalars have been removed.

    Examples
    --------

    >>> deep_remove([[[[0, 1, 2]], [3, 4], [5], [6, 7]]], 0, 5) # doctest: +SKIP
    [[[[1, 2]], [3, 4], [], [6, 7]]]

    """
    if isinstance(l, list):
        # Make a shallow copy at this level.
        l = l[:]
        for to_remove in what:
            if to_remove in l:
                l.remove(to_remove)
            else:
                l = [deep_remove(elem, to_remove) for elem in l]
    return l


def unflatten(l, idxs):
    """Unflattens a list by a given structure.

    Given a list of scalars and a deep list of indices
    as produced by `flatten`, returns an "unflattened"
    form of the list. This perfectly inverts `flatten`.

    Examples
    --------

    >>> l = [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]] # doctest: +SKIP
    >>> idxs = enumerate_flat(l) # doctest: +SKIP
    >>> unflatten(flatten(l), idxs) == l # doctest: +SKIP
    True

    """
    acc = []
    for idx in idxs:
        if isinstance(idx, list):
            acc.append(unflatten(l, idx))
        else:
            acc.append(l[idx])
    return acc


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


def _collapse_composite_index(dims):
    """
    Given the dimensions specification for a composite index
    (e.g.: [2, 3] for the right index of a ket with dims [[1], [2, 3]]),
    returns a dimensions specification for an index of the same shape,
    but collapsed to a single "leg." In the previous example, [2, 3]
    would collapse to [6].
    """
    return [np.prod(dims)]


def _collapse_dims_to_level(dims, level=1):
    """
    Recursively collapses all indices in a dimensions specification
    appearing at a given level, such that the returned dimensions
    specification does not represent any composite systems.
    """
    if level == 0:
        return _collapse_composite_index(dims)
    return [_collapse_dims_to_level(index, level=level - 1) for index in dims]


def collapse_dims_oper(dims):
    """
    Given the dimensions specifications for a ket-, bra- or oper-type
    Qobj, returns a dimensions specification describing the same shape
    by collapsing all composite systems. For instance, the bra-type
    dimensions specification ``[[2, 3], [1]]`` collapses to
    ``[[6], [1]]``.

    Parameters
    ----------

    dims : list of lists of ints
        Dimensions specifications to be collapsed.

    Returns
    -------

    collapsed_dims : list of lists of ints
        Collapsed dimensions specification describing the same shape
        such that ``len(collapsed_dims[0]) == len(collapsed_dims[1]) == 1``.
    """
    return _collapse_dims_to_level(dims, 1)


def collapse_dims_super(dims):
    """
    Given the dimensions specifications for an operator-ket-, operator-bra- or
    super-type Qobj, returns a dimensions specification describing the same shape
    by collapsing all composite systems. For instance, the super-type
    dimensions specification ``[[[2, 3], [2, 3]], [[2, 3], [2, 3]]]`` collapses to
    ``[[[6], [6]], [[6], [6]]]``.

    Parameters
    ----------

    dims : list of lists of ints
        Dimensions specifications to be collapsed.

    Returns
    -------

    collapsed_dims : list of lists of ints
        Collapsed dimensions specification describing the same shape
        such that ``len(collapsed_dims[i][j]) == 1`` for ``i`` and ``j``
        in ``range(2)``.
    """
    return _collapse_dims_to_level(dims, 2)


def enumerate_flat(l):
    """Labels the indices at which scalars occur in a flattened list.

    Given a list containing a mix of scalars and lists,
    returns a list of the same structure, where each scalar
    has been replaced by an index into the flattened list.

    Examples
    --------

    >>> print(enumerate_flat([[[10], [20, 30]], 40])) # doctest: +SKIP
    [[[0], [1, 2]], 3]

    """
    return _enumerate_flat(l)[0]


def deep_map(fn, collection, over=(tuple, list)):
    if isinstance(collection, over):
        return type(collection)(deep_map(fn, el, over) for el in collection)
    else:
        return fn(collection)


def dims_to_tensor_perm(dims):
    """
    Given the dims of a Qobj instance, returns a list representing
    a permutation from the flattening of that dims specification to
    the corresponding tensor indices.

    Parameters
    ----------

    dims : list
        Dimensions specification for a Qobj.

    Returns
    -------

    perm : list
        A list such that ``data[flatten(dims)[idx]]`` gives the
        index of the tensor ``data`` corresponding to the ``idx``th
        dimension of ``dims``.
    """
    # We figure out the type of the dims specification,
    # relaxing the requirement that operators be square.
    # This means that dims_type need not coincide with
    # Qobj.type, but that works fine for our purposes here.
    dims_type = type_from_dims(dims, enforce_square=False)
    perm = enumerate_flat(dims)
    if dims_type in ('oper', 'ket', 'bra'):
        return flatten(perm)

    # If the type is other, we need to figure out if the
    # dims is superlike on its outputs and inputs
    # This is the case if the dims type for left or right
    # are, respectively, oper-like.
    if dims_type == 'other':
        raise NotImplementedError("Not yet implemented for type='other'.")

    # If we're still here, the story is more complicated. We'll
    # follow the strategy of creating a permutation by using
    # enumerate_flat then transforming the result to swap
    # input and output indices of vectorized matrices, then flattening
    # the result. We'll then rebuild indices using this permutation.
    if dims_type in ('operator-ket', 'super'):
        # Swap the input and output spaces of the right part of
        # perm.
        perm[1] = list(reversed(perm[1]))
    if dims_type in ('operator-bra', 'super'):
        # Ditto, but for the left indices.
        perm[0] = list(reversed(perm[0]))
    return flatten(perm)


def dims_to_tensor_shape(dims):
    """
    Given the dims of a Qobj instance, returns the shape of the
    corresponding tensor. This helps, for instance, resolve the
    column-stacking convention for superoperators.

    Parameters
    ----------

    dims : list
        Dimensions specification for a Qobj.

    Returns
    -------

    tensor_shape : tuple
        NumPy shape of the corresponding tensor.
    """
    perm = dims_to_tensor_perm(dims)
    dims = flatten(dims)
    return tuple(map(partial(getitem, dims), perm))


def dims_idxs_to_tensor_idxs(dims, indices):
    """
    Given the dims of a Qobj instance, and some indices into
    dims, returns the corresponding tensor indices. This helps
    resolve, for instance, that column-stacking for superoperators,
    oper-ket and oper-bra implies that the input and output tensor
    indices are reversed from their order in dims.

    Parameters
    ----------

    dims : list
        Dimensions specification for a Qobj.

    indices : int, list or tuple
        Indices to convert to tensor indices. Can be specified
        as a single index, or as a collection of indices.
        In the latter case, this can be nested arbitrarily
        deep. For instance, [0, [0, (2, 3)]].

    Returns
    -------

    tens_indices : int, list or tuple
        Container of the same structure as indices containing
        the tensor indices for each element of indices.
    """
    perm = dims_to_tensor_perm(dims)
    return deep_map(partial(getitem, perm), indices)


def _check_qubits_oper(oper, dims=None, targets=None):
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
        The indices of qubits that are acted on.
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
        The indices of qubits that are acted on.
    oper : :class:`qutip.Qobj`, optional
        An operator acts on qubits, the type of the :class:`qutip.Qobj`
        has to be an operator
        and the dimension matches the tensored qubit Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    N : int, optional
        The number of qubits in the system.
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
    Expand a qubits operator to one that acts on a N-qubit system.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        An operator acts on qubits, the type of the :class:`qutip.Qobj`
        has to be an operator and the dimension matches the tensored qubit
        Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    targets : int or list of int
        The indices of qubits that are acted on.
    dims : list, optional
        A list of integer for the dimension of each composite system.
        E.g ``[2, 2, 2, 3, 3]``.

    Returns
    -------
    expanded_oper : :class:`qutip.Qobj`
        The expanded qubits operator acting on a system with N qubits.
    """
    N = len(dims)
    targets = _targets_to_list(targets, oper=oper, N=N)
    _check_qubits_oper(oper, dims=dims, targets=targets)

    # Generate the correct order for qubits permutation,
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
