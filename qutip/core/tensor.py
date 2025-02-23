"""
Module for the creation of composite quantum objects via the tensor product.
"""

__all__ = [
    'tensor', 'super_tensor', 'composite', 'tensor_swap', 'tensor_contract',
    'expand_operator'
]

import numpy as np
from functools import partial
from typing import overload

from .operators import qeye
from .qobj import Qobj
from .cy.qobjevo import QobjEvo
from .superoperator import operator_to_vector, reshuffle
from .dimensions import (
    flatten, enumerate_flat, unflatten, deep_remove, dims_to_tensor_shape,
    dims_idxs_to_tensor_idxs
)
from . import data as _data
from .. import settings
from ..typing import LayerType


class _reverse_partial_tensor:
    """Picklable lambda op: tensor(op, right)"""
    def __init__(self, right):
        self.right = right

    def __call__(self, op):
        return tensor(op, self.right)


@overload
def tensor(*args: Qobj) -> Qobj: ...
 
@overload
def tensor(*args: Qobj | QobjEvo) -> QobjEvo: ...

def tensor(*args: Qobj | QobjEvo) -> Qobj | QobjEvo:
    """
    Calculates the tensor product of input operators.

    Parameters
    ----------
    args : array_like
        ``list`` or ``array`` of quantum objects for tensor product.

    Returns
    -------
    obj : Qobj | QobjEvo
        A composite quantum object.

    Examples
    --------
    >>> tensor([sigmax(), sigmax()])  # doctest: +SKIP
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

    if len(args) == 1 and isinstance(args[0], (Qobj, QobjEvo)):
        return args[0].copy()

    try:
        args = tuple(args[0])
    except TypeError:
        raise TypeError("Input must be a list or tuple of Qobj or QobjEvo operands")

    if not all(isinstance(q, (Qobj, QobjEvo)) for q in args):
        raise TypeError("All arguments must be Qobj or QobjEvo.")

    if any(isinstance(q, QobjEvo) for q in args):
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
        raise TypeError("In tensor products of superoperators, all must have the same representation.")

    isherm = args[0]._isherm
    isunitary = args[0]._isunitary
    out_data = args[0].data
    dims_l = [args[0]._dims[0]]
    dims_r = [args[0]._dims[1]]

    for arg in args[1:]:
        out_data = _data.kron(out_data, arg.data)
        isherm = (isherm and arg._isherm) or None
        isunitary = (isunitary and arg._isunitary) or None
        dims_l.append(arg._dims[0])
        dims_r.append(arg._dims[1])

    return Qobj(
        out_data,
        dims=[dims_l, dims_r],
        isherm=isherm,
        isunitary=isunitary,
        copy=False
    )


@overload
def super_tensor(*args: Qobj) -> Qobj: ...

@overload
def super_tensor(*args: Qobj | QobjEvo) -> QobjEvo: ...

def super_tensor(*args: Qobj | QobjEvo) -> Qobj | QobjEvo:
    """
    Calculate the tensor product of input superoperators.

    Parameters
    ----------
    args : array_like
        ``list`` or ``array`` of quantum objects with ``type="super"```.

    Returns
    -------
    obj : Qobj | QobjEvo
        A composite quantum object.
    """
    if not all(arg.issuper for arg in args):
        raise TypeError("All arguments must be superoperators.")

    if not all(arg.superrep == "super" for arg in args):
        raise TypeError("All superoperators must have the 'super' representation.")

    shuffled_ops = list(map(reshuffle, args))
    shuffled_tensor = tensor(shuffled_ops)
    out = reshuffle(shuffled_tensor)
    out.superrep = args[0].superrep
    return out


def composite(*args: Qobj | QobjEvo) -> Qobj | QobjEvo:
    """
    Create a composite quantum object from a mix of operators, kets, bras, and superoperators.

    Parameters
    ----------
    args : array_like
        list or array of quantum objects.

    Returns
    -------
    obj : Qobj | QobjEvo
        A composite quantum object.
    """
    if not all(isinstance(arg, Qobj) for arg in args):
        raise TypeError("All arguments must be Qobjs.")

    if all(arg.isoper or arg.issuper for arg in args):
        if any(arg.issuper for arg in args):
            return super_tensor(*map(lambda x: x.to_super(), args))
        return tensor(*args)

    if all(arg.isket or arg.isoperket for arg in args):
        return super_tensor(*[
            arg if arg.isoperket else operator_to_vector(arg.proj())
            for arg in args
        ])

    if all(arg.isbra or arg.isoperbra for arg in args):
        return composite(*(arg.dag() for arg in args)).dag()

    raise TypeError("Unsupported Qobj types.")


def tensor_swap(q_oper: Qobj, *pairs: tuple[int, int]) -> Qobj:
    """
    Transposes one or more pairs of indices of a Qobj.

    Parameters
    ----------
    q_oper : Qobj
        Operator to swap dimensions.

    pairs : tuple[int, int]
        Pairs of indices to swap.

    Returns
    -------
    sqobj : Qobj
        The original Qobj with swapped dimensions.
    """
    dims = q_oper.dims
    tensor_pairs = dims_idxs_to_tensor_idxs(dims, pairs)
    data = q_oper.full().reshape(dims_to_tensor_shape(dims))

    flat_dims = flatten(dims)
    perm = list(range(len(flat_dims)))
    for i, j in pairs:
        flat_dims[i], flat_dims[j] = flat_dims[j], flat_dims[i]
    for i, j in tensor_pairs:
        perm[i], perm[j] = perm[j], perm[i]

    dims = unflatten(flat_dims, enumerate_flat(dims))
    data = data.transpose(perm).reshape(list(map(np.prod, dims)))
    return Qobj(data, dims=dims, superrep=q_oper.superrep, copy=False)


def tensor_contract(qobj: Qobj, *pairs: tuple[int, int]) -> Qobj:
    """
    Contracts a Qobj along one or more index pairs.

    Parameters
    ----------
    qobj : Qobj
        Operator to contract subspaces on.

    pairs : tuple[int, int]
        Pairs of indices to contract.

    Returns
    -------
    cqobj : Qobj
        The original Qobj with contracted dimensions.
    """
    dims = qobj.dims
    tensor_dims = dims_to_tensor_shape(dims)
    qtens = qobj.data.to_array().reshape(tensor_dims)

    qtens = _tensor_contract_dense(qtens, *dims_idxs_to_tensor_idxs(dims, pairs))
    contracted_dims = unflatten(
        deep_remove(flatten(dims), *flatten(list(map(list, pairs)))),
        enumerate_flat(dims)
    )

    l_mtx_dims, r_mtx_dims = map(np.prod, map(flatten, contracted_dims))
    qmtx = qtens.reshape((l_mtx_dims, r_mtx_dims))
    return Qobj(qmtx, dims=contracted_dims, superrep=qobj.superrep, copy=False)


def expand_operator(
    oper: Qobj | QobjEvo,
    dims: list[int],
    targets: int | list[int],
    dtype: LayerType = None
) -> Qobj | QobjEvo:
    """
    Expand an operator to act on a system with desired dimensions.

    Parameters
    ----------
    oper : Qobj | QobjEvo
        An operator acting on the subsystem.

    dims : list[int]
        Dimensions of each composite system.

    targets : int | list[int]
        Indices of subspace that are acted on.

    dtype : LayerType, optional
        Data type of the output Qobj.

    Returns
    -------
    expanded_oper : Qobj | QobjEvo
        The expanded operator.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.CSR
    oper = oper.to(dtype)
    N = len(dims)
    targets = _targets_to_list(targets, oper=oper, N=N)
    _check_oper_dims(oper, dims=dims, targets=targets)

    new_order = [0] * N
    for i, t in enumerate(targets):
        new_order[t] = i
    rest_pos = [q for q in range(N) if q not in targets]
    rest_qubits = list(range(len(targets), N))
    for i, ind in enumerate(rest_pos):
        new_order[ind] = rest_qubits[i]

    id_list = [qeye(dims[i]) for i in rest_pos]
    return tensor([oper] + id_list).permute(new_order).to(dtype)


def _targets_to_list(
    targets: int | list[int],
    oper: Qobj | None = None,
    N: int | None = None
) -> list[int]:
    """
    Transform targets to a list and check validity.

    Parameters
    ----------
    targets : int | list[int]
        Indices of subspace that are acted on.

    oper : Qobj, optional
        The operator being expanded.

    N : int, optional
        Total number of subsystems.

    Returns
    -------
    list[int]
        Validated list of target indices.
    """
    if targets is None:
        targets = list(range(len(oper.dims[0]) if oper else []))
    if not hasattr(targets, '__iter__'):
        targets = [targets]
    if not all(isinstance(t, int) for t in targets):
        raise TypeError("Targets must be integers.")
    if oper and len(targets) != len(oper.dims[0]):
        raise ValueError("Number of targets does not match operator dimensions.")
    if N and any(t >= N for t in targets):
        raise ValueError(f"Targets must be less than N={N}.")
    return targets


def _check_oper_dims(oper: Qobj, dims: list[int], targets: list[int]) -> None:
    """
    Validate the operator dimensions against the target dimensions.

    Parameters
    ----------
    oper : Qobj
        The operator being checked.

    dims : list[int]
        Dimensions of the composite system.

    targets : list[int]
        Indices of the subsystems being targeted.
    """
    if not isinstance(oper, Qobj) or oper.dims[0] != oper.dims[1]:
        raise ValueError("Operator must have the same input and output dimensions.")
    target_dims = [dims[t] for t in targets]
    if oper.dims[0] != target_dims:
        raise ValueError(f"Operator dims {oper.dims[0]} do not match target dims {target_dims}.")
