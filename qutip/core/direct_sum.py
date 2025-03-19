from .dimensions import Dimensions, Field, SumSpace
from .qobj import Qobj
from .superoperator import operator_to_vector, vector_to_operator
from . import data as _data
from .. import settings

from numbers import Number

import numpy as np

__all__ = ['direct_sum', 'component']


def _is_like_scalar(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type == 'scalar'
        ))

def _is_like_qobj(qobj):
    return isinstance(qobj, Number) or isinstance(qobj, Qobj)

def _qobj_data(qobj):
    return qobj.data if isinstance(qobj, Qobj) else np.array([[qobj]])

def _qobj_dims(qobj):
    return (
        qobj._dims if isinstance(qobj, Qobj) else Dimensions(Field(), Field())
    )

def _is_like_ket(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'ket']
        ))

def _is_like_bra(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'bra']
        ))

def _is_like_oper(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'oper', 'ket', 'bra']
        ))

def _is_like_operator_ket(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'oper', 'operator-ket']
        ))

def _is_like_operator_bra(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'oper', 'operator-bra']
        ))

def _is_like_super(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'super',
                              'operator-ket', 'operator-bra']
        ))

def direct_sum(qobjs: list[Qobj | float] | list[list[Qobj | float]]) -> Qobj:
    if len(qobjs) == 0:
        raise ValueError("No Qobjs provided for direct sum.")

    linear = _is_like_qobj(qobjs[0])
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
            if isinstance(qobj, Qobj) and qobj.type == 'oper':
                return operator_to_vector(qobj)
            return qobj
        qobjs = [[_ensure_vector(qobj)] for qobj in qobjs]
    elif linear and all(_is_like_bra(qobj) for qobj in qobjs):
        qobjs = [[qobj for qobj in qobjs]]
    elif linear and all(_is_like_operator_bra(qobj) for qobj in qobjs):
        # for convenience, we call operator_to_vector on operators provided
        # in matrix form
        def _ensure_vector(qobj):
            if isinstance(qobj, Qobj) and qobj.type == 'oper':
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
        raise ValueError("Mismatching dimensions in direct sum of operators.")

    out_data = _data.concat_data(
        [[_qobj_data(qobj) for qobj in row] for row in qobjs],
        _skip_checks=True
    )
    return Qobj(out_data,
                dims=Dimensions(SumSpace(*from_dim), SumSpace(*to_dim)),
                copy=False)


def _check_index(given, min, max):
    if not (min <= given < max):
        raise ValueError(f"Index ({given}) out of bounds ({min}, {max-1})"
                          " for component of direct sum.")

def component(qobj: Qobj, *index: int) -> Qobj:
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