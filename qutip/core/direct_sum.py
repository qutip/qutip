from .cy.qobjevo import QobjEvo
from .dimensions import Dimensions, Field, SumSpace
from .operators import qzero_like
from .qobj import Qobj
from .superoperator import operator_to_vector
from . import data as _data

from numbers import Number
from typing import overload, Union

import numpy as np

__all__ = ['direct_sum', 'component']


QobjLike = Union[Number, Qobj, QobjEvo]

def _qobj_data(qobj: Number | Qobj) -> np.ndarray | _data.Data:
    return qobj.data if isinstance(qobj, Qobj) else np.array([[qobj]])

def _qobj_dims(qobj: QobjLike) -> Dimensions:
    return (
        qobj._dims if isinstance(qobj, (Qobj, QobjEvo))
        else Dimensions(Field(), Field())
    )

def _is_like_ket(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'ket']
        ))

def _is_like_bra(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'bra']
        ))

def _is_like_oper(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'oper', 'ket', 'bra']
        ))

def _is_like_operator_ket(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'oper', 'operator-ket']
        ))

def _is_like_operator_bra(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'oper', 'operator-bra']
        ))

def _is_like_super(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'super',
                              'operator-ket', 'operator-bra']
        ))

@overload
def direct_sum(
    qobjs: list[Qobj | float] | list[list[Qobj | float]]
) -> Qobj:
    ...

@overload
def direct_sum(
    qobjs: list[QobjEvo | Qobj | float] | list[list[QobjEvo | Qobj | float]]
) -> QobjEvo:
    ...

def direct_sum(qobjs):
    if len(qobjs) == 0:
        raise ValueError("No Qobjs provided for direct sum.")

    linear = isinstance(qobjs[0], QobjLike)
    if not linear and len(qobjs[0]) == 0:
        raise ValueError("No Qobjs provided for direct sum.")
    if not linear and not all(len(row) == len(qobjs[0]) for row in qobjs):
        raise ValueError("Matrix of Qobjs in direct sum must be square.")

    if linear and all(_is_like_ket(qobj) for qobj in qobjs):
        qobjs = [[qobj] for qobj in qobjs]
    elif linear and all(_is_like_operator_ket(qobj) for qobj in qobjs):
        # for convenience, we call operator_to_vector on operators provided
        # in matrix form
        def _ensure_vector(qobj):
            if isinstance(qobj, (Qobj, QobjEvo)) and qobj.type == 'oper':
                return operator_to_vector(qobj)
            return qobj
        qobjs = [[_ensure_vector(qobj)] for qobj in qobjs]
    elif linear and all(_is_like_bra(qobj) for qobj in qobjs):
        qobjs = [[qobj for qobj in qobjs]]
    elif linear and all(_is_like_operator_bra(qobj) for qobj in qobjs):
        # for convenience, we call operator_to_vector on operators provided
        # in matrix form
        def _ensure_vector(qobj):
            if isinstance(qobj, (Qobj, QobjEvo)) and qobj.type == 'oper':
                return operator_to_vector(qobj).dag()
            return qobj
        qobjs = [[_ensure_vector(qobj) for qobj in qobjs]]
    elif linear:
        raise ValueError("Invalid combination of Qobj types"
                         " for direct sum.")
    else:
        allsuper = all(_is_like_super(qobj) for row in qobjs for qobj in row)
        alloper = all(_is_like_oper(qobj) for row in qobjs for qobj in row)
        if not (allsuper or alloper):
            raise ValueError("Invalid combination of Qobj types"
                             " for direct sum.")

    from_dim = [_qobj_dims(qobj).from_ for qobj in qobjs[0]]
    to_dim = [_qobj_dims(row[0]).to_ for row in qobjs]
    dims_match = all(
        _qobj_dims(qobj).from_ == from_dim[col_index]
        and _qobj_dims(qobj).to_ == to_dim[row_index]
        for row_index, row in enumerate(qobjs)
        for col_index, qobj in enumerate(row)
    )
    if not dims_match:
        raise ValueError("Mismatching dimensions in direct sum.")
    out_dims = Dimensions(SumSpace(*from_dim), SumSpace(*to_dim))

    # Handle QobjEvos. We have to pull them out and handle them separately.
    qobjevos = []
    for row_index, row in enumerate(qobjs):
        for col_index, qobj in enumerate(row):
            if isinstance(qobj, QobjEvo):
                qobjs[row_index][col_index] = qzero_like(qobj)

                dim_before = sum(from_dim[i].size for i in range(col_index))
                dim_after = sum(from_dim[i].size
                                for i in range(col_index + 1, len(from_dim)))
                dim_above = sum(to_dim[i].size for i in range(row_index))
                dim_below = sum(to_dim[i].size
                                for i in range(row_index + 1, len(to_dim)))
                zeropad_args = (dim_before, dim_after, dim_above, dim_below)
                blow_up = qobj.linear_map(
                    lambda x: Qobj(_data.zeropad(x.data, *zeropad_args),
                                   dims=out_dims, copy=False)
                )
                qobjevos.append(blow_up)

    out_data = _data.concat_data(
        [[_qobj_data(qobj) for qobj in row] for row in qobjs],
        _skip_checks=True
    )
    result = Qobj(out_data, dims=out_dims, copy=False)
    return sum(qobjevos, start=result)


def _check_index(given, min, max):
    if not (min <= given < max):
        raise ValueError(f"Index ({given}) out of bounds ({min}, {max-1})"
                          " for component of direct sum.")

@overload
def component(qobj: Qobj, *index: int) -> Qobj:
    ...

@overload
def component(qobj: QobjEvo, *index: int) -> Qobj | QobjEvo:
    ...

def component(qobj, *index):
    if isinstance(qobj, QobjEvo):
        result = qobj.linear_map(lambda x: component(x, *index))
        result.compress()
        return result(0) if result.isconstant else result

    is_to_sum = isinstance(qobj._dims.to_, SumSpace)
    is_from_sum = isinstance(qobj._dims.from_, SumSpace)
    if not is_to_sum and not is_from_sum:
        if index == [0] or index == [0, 0]:
            return qobj
        raise ValueError("Qobj is not a direct sum.")

    if is_to_sum and is_from_sum:
        if len(index) != 2:
            raise ValueError("Invalid number of indices provided for component"
                             " of direct sum (two indices required).")
        to_index, from_index = index
        _check_index(to_index, 0, len(qobj._dims.to_.spaces))
        _check_index(from_index, 0, len(qobj._dims.from_.spaces))
        to_data_start = qobj._dims.to_._space_cumdims[to_index]
        to_data_stop = qobj._dims.to_._space_cumdims[to_index + 1]
        from_data_start = qobj._dims.from_._space_cumdims[from_index]
        from_data_stop = qobj._dims.from_._space_cumdims[from_index + 1]

        out_data = _data.slice(qobj.data,
                               to_data_start, to_data_stop,
                               from_data_start, from_data_stop)
        return Qobj(out_data,
                    dims=Dimensions(qobj._dims.from_.spaces[from_index],
                                    qobj._dims.to_.spaces[to_index]),
                    copy=False)

    if len(index) == 2:
        to_index, from_index = index
        if not is_to_sum:
            _check_index(to_index, 0, 1)
            index = [from_index]
        else:
            _check_index(from_index, 0, 1)
            index = [to_index]

    if len(index) != 1:
        raise ValueError("Invalid number of indices provided for component"
                         " of direct sum (one or two indices required).")

    if is_to_sum:
        to_index, = index
        _check_index(to_index, 0, len(qobj._dims.to_.spaces))
        to_data_start = qobj._dims.to_._space_cumdims[to_index]
        to_data_stop = qobj._dims.to_._space_cumdims[to_index + 1]
        from_data_start = 0
        from_data_stop = qobj._dims.from_.size

        out_data = _data.slice(qobj.data,
                               to_data_start, to_data_stop,
                               from_data_start, from_data_stop)
        result = Qobj(out_data,
                      dims=Dimensions(qobj._dims.from_,
                                      qobj._dims.to_.spaces[to_index]),
                      copy=False)
        return result

    from_index, = index
    _check_index(from_index, 0, len(qobj._dims.from_.spaces))
    to_data_start = 0
    to_data_stop = qobj._dims.to_.size
    from_data_start = qobj._dims.from_._space_cumdims[from_index]
    from_data_stop = qobj._dims.from_._space_cumdims[from_index + 1]

    out_data = _data.slice(qobj.data,
                           to_data_start, to_data_stop,
                           from_data_start, from_data_stop)
    result = Qobj(out_data,
                  dims=Dimensions(qobj._dims.from_.spaces[from_index],
                                  qobj._dims.to_),
                  copy=False)
    return result