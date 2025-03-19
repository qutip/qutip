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
        if not settings.core['auto_tidyup_dims']:
            scalar_dim = _qobj_dims(qobjs[0]).from_
            scalar_dims_match = all(
                _qobj_dims(qobj).from_ == scalar_dim for qobj in qobjs
            )
            if not scalar_dims_match:
                raise ValueError("Scalar dimensions do not match in"
                                 " direct sum of vectors.")
        else:
            scalar_dim = Field()
        vector_dim = SumSpace([_qobj_dims(qobj).to_ for qobj in qobjs])

        out_data = _data.concat_data([[_qobj_data(qobj)] for qobj in qobjs],
                                     _skip_checks=True)
        return Qobj(out_data,
                    dims=Dimensions(scalar_dim, vector_dim),
                    copy=False)

    if linear and all(_is_like_bra(qobj) for qobj in qobjs):
        if not settings.core['auto_tidyup_dims']:
            scalar_dim = _qobj_dims(qobjs[0]).to_
            scalar_dims_match = all(
                _qobj_dims(qobj).to_ == scalar_dim for qobj in qobjs
            )
            if not scalar_dims_match:
                raise ValueError("Scalar dimensions do not match in"
                                 " direct sum of covectors.")
        else:
            scalar_dim = Field()
        vector_dim = SumSpace([_qobj_dims(qobj).from_ for qobj in qobjs])

        out_data = _data.concat_data([[_qobj_data(qobj) for qobj in qobjs]],
                                     _skip_checks=True)
        return Qobj(out_data,
                    dims=Dimensions(vector_dim, scalar_dim),
                    copy=False)

    if linear and all(_is_like_operator_ket(qobj) for qobj in qobjs):
        oper_as_opket = [False] * len(qobjs)
        for index, qobj in enumerate(qobjs):
            if isinstance(qobj, Qobj) and qobj.type == 'oper':
                oper_as_opket[index] = True
                qobjs[index] = operator_to_vector(qobj)

        if not settings.core['auto_tidyup_dims']:
            scalar_dim = _qobj_dims(qobjs[0]).from_
            scalar_dims_match = all(
                _qobj_dims(qobj).from_ == scalar_dim for qobj in qobjs
            )
            if not scalar_dims_match:
                raise ValueError("Scalar dimensions do not match in"
                                 " direct sum of operator-kets.")
        else:
            scalar_dim = Field()
        vector_dim = SumSpace([_qobj_dims(qobj).to_ for qobj in qobjs],
                              oper_as_opket)

        out_data = _data.concat_data([[_qobj_data(qobj)] for qobj in qobjs],
                                     _skip_checks=True)
        return Qobj(out_data,
                    dims=Dimensions(scalar_dim, vector_dim),
                    copy=False)

    if linear and all(_is_like_operator_bra(qobj) for qobj in qobjs):
        oper_as_opket = [False] * len(qobjs)
        for index, qobj in enumerate(qobjs):
            if isinstance(qobj, Qobj) and qobj.type == 'oper':
                oper_as_opket[index] = True
                qobjs[index] = operator_to_vector(qobj).dag()

        if not settings.core['auto_tidyup_dims']:
            scalar_dim = _qobj_dims(qobjs[0]).to_
            scalar_dims_match = all(
                _qobj_dims(qobj).to_ == scalar_dim for qobj in qobjs
            )
            if not scalar_dims_match:
                raise ValueError("Scalar dimensions do not match in"
                                 " direct sum of covectors.")
        else:
            scalar_dim = Field()
        vector_dim = SumSpace([_qobj_dims(qobj).from_ for qobj in qobjs],
                              oper_as_opket)

        out_data = _data.concat_data([[_qobj_data(qobj) for qobj in qobjs]],
                                     _skip_checks=True)
        return Qobj(out_data,
                    dims=Dimensions(vector_dim, scalar_dim),
                    copy=False)

    allsuper = all(_is_like_super(qobj) for row in qobjs for qobj in row)
    alloper = all(_is_like_oper(qobj) for row in qobjs for qobj in row)
    if allsuper and any(not _is_like_scalar(qobj) and qobj.superrep != 'super'
                        for row in qobjs for qobj in row):
        raise ValueError("Direct sums only accept superoperators"
                         " in super representation.")

    if not linear and (allsuper or alloper):
        from_dim = [_qobj_dims(qobj).from_ for qobj in qobjs[0]]
        to_dim = [_qobj_dims(row[0]).to_ for row in qobjs]
        dims_match = all(
            _qobj_dims(qobj).from_ == from_dim[col_index]
            and _qobj_dims(qobj).to_ == to_dim[row_index]
            for row_index, row in enumerate(qobjs)
            for col_index, qobj in enumerate(row)
        )
        if not dims_match:
            raise ValueError("Mismatching dimensions in"
                             " direct sum of operators.")

        out_data = _data.concat_data(
            [[_qobj_data(qobj) for qobj in row] for row in qobjs],
            _skip_checks=True
        )
        return Qobj(out_data,
                    dims=Dimensions(SumSpace(from_dim), SumSpace(to_dim)),
                    copy=False)

    raise ValueError("Invalid combination of Qobj types for direct sum.")


def component(qobj: Qobj, *index: list[int]) -> Qobj:
    is_to_sum = isinstance(qobj._dims.to_, SumSpace)
    is_from_sum = isinstance(qobj._dims.from_, SumSpace)
    if not is_to_sum and not is_from_sum:
        raise ValueError("Qobj is not a direct sum.")

    if is_to_sum and is_from_sum:
        if len(index) != 2:
            raise ValueError("Invalid number of indices provided for component"
                             " of direct sum (two indices required).")
        to_index, from_index = index
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

    if len(index) != 1:
        raise ValueError("Invalid number of indices provided for component"
                         " of direct sum (one index required).")

    if is_to_sum:
        to_index, = index
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
        if qobj._dims.to_._oper_as_opket[to_index]:
            return vector_to_operator(result)
        return result

    from_index, = index
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
    if qobj._dims.from_._oper_as_opket[from_index]:
        return vector_to_operator(result.dag())
    return result