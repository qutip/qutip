from .cy.qobjevo import QobjEvo
from .dimensions import Dimensions, Field, SumSpace
from .operators import qzero_like
from .qobj import Qobj
from .superoperator import operator_to_vector
from . import data as _data

from ..typing import DimensionLike, LayerType
from .. import settings

from numbers import Number
from typing import overload, Union

import numpy as np


__all__ = ['direct_sum', 'sparse_direct_sum', 'component', 'set_component']


QobjLike = Union[Number, Qobj, QobjEvo]


def _qobj_data(qobj: Number | Qobj, scalar_dtype: LayerType) -> _data.Data:
    if qobj is None:
        return None
    if isinstance(qobj, Number):
        return _data.identity[scalar_dtype](1, scale=qobj)
    return qobj.data


def _qobj_dims(qobj: QobjLike) -> Dimensions:
    return (
        qobj._dims if isinstance(qobj, (Qobj, QobjEvo))
        else Dimensions(Field(), Field())
    )


def _is_like_ket(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number) or (
            isinstance(qobj, (Qobj, QobjEvo)) and
            qobj.type in ['scalar', 'ket']
        ))


def _is_like_bra(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number) or (
            isinstance(qobj, (Qobj, QobjEvo)) and
            qobj.type in ['scalar', 'bra']
        ))


def _is_like_oper(qobj: QobjLike) -> bool:
    return (
        qobj is None or
        isinstance(qobj, Number) or (
            isinstance(qobj, (Qobj, QobjEvo)) and
            qobj.type in ['scalar', 'oper', 'ket', 'bra']
        ))


def _is_like_operator_ket(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number) or (
            isinstance(qobj, (Qobj, QobjEvo)) and
            qobj.type in ['scalar', 'oper', 'operator-ket']
        ))


def _is_like_operator_bra(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number) or (
            isinstance(qobj, (Qobj, QobjEvo)) and
            qobj.type in ['scalar', 'oper', 'operator-bra']
        ))


def _is_like_super(qobj: QobjLike) -> bool:
    return (
        qobj is None or
        isinstance(qobj, Number) or (
            isinstance(qobj, (Qobj, QobjEvo)) and
            qobj.type in ['scalar', 'super', 'operator-ket', 'operator-bra']
        ))


@overload
def direct_sum(
    qobjs: list[Qobj | float] | list[list[Qobj | float]],
    dtype: LayerType = None
) -> Qobj:
    ...


@overload
def direct_sum(
    qobjs: list[QobjEvo | Qobj | float] | list[list[QobjEvo | Qobj | float]],
    dtype: LayerType = None
) -> QobjEvo:
    ...


def direct_sum(qobjs, dtype=None):
    """
    Takes a list or matrix of Qobjs and makes them into a single Qobj with
    block-matrix elements.
    """
    if settings.core["default_dtype_scope"] == "full":
        dtype = _data._parse_default_dtype(dtype, "sparse")
    else:
        dtype = _data._parse_default_dtype(dtype or "core", "sparse")

    qobjs = _process_arguments(qobjs)
    out_dims, block_widths, block_heights = _determine_dimensions(qobjs)

    # Get data from the Qobjs
    data_array = np.empty((len(block_heights), len(block_widths)),
                          dtype=_data.Data)
    # For QobjEvos, we have to pull them out and handle them separately.
    qobjevos = []
    to_data_start = 0
    for to_index, row in enumerate(qobjs):
        from_data_start = 0
        for from_index, qobj in enumerate(row):
            if isinstance(qobj, QobjEvo):
                # don't include the qobjevo with the regular qobjs ...
                data_array[to_index][from_index] = None

                # ... but embed it in big zero matrix and add to qobjevos
                blow_up = qobj.linear_map(
                    lambda x: Qobj(
                        _data.insert(
                            _data.zeros[dtype](*out_dims.shape), x.data,
                            to_data_start, from_data_start, dtype=dtype),
                        dims=out_dims, copy=False),
                    _skip_check=True)
                qobjevos.append(blow_up)
            else:
                # regular qobj
                data_array[to_index][from_index] = _qobj_data(qobj, dtype)
            from_data_start += block_widths[from_index]
        to_data_start += block_heights[to_index]

    out_data = _data.concat(
        data_array, block_widths, block_heights, dtype=dtype
    )
    result = Qobj(out_data, dims=out_dims, copy=False)
    return sum(qobjevos, start=result)


def _process_arguments(qobjs):
    """
    Ensures that the qobjs have compatible types, including converting
    operators to operator-ket / operator-bra if necessary.
    If a list of Qobj is provided, converts it to a matrix (list of lists).
    """

    if len(qobjs) == 0:
        raise ValueError("No Qobjs provided for direct sum.")

    linear = isinstance(qobjs[0], QobjLike)
    if not linear and len(qobjs[0]) == 0:
        raise ValueError("No Qobjs provided for direct sum.")
    if not linear and not all(len(row) == len(qobjs[0]) for row in qobjs):
        raise ValueError("Matrix of Qobjs in direct sum must be square.")

    if linear and all(_is_like_ket(qobj) for qobj in qobjs):
        return [[qobj] for qobj in qobjs]

    if linear and all(_is_like_operator_ket(qobj) for qobj in qobjs):
        # for convenience, we call operator_to_vector on operators provided
        # in matrix form
        def _ensure_vector(qobj):
            if isinstance(qobj, (Qobj, QobjEvo)) and qobj.type == 'oper':
                return operator_to_vector(qobj)
            return qobj
        return [[_ensure_vector(qobj)] for qobj in qobjs]

    if linear and all(_is_like_bra(qobj) for qobj in qobjs):
        return [[qobj for qobj in qobjs]]

    if linear and all(_is_like_operator_bra(qobj) for qobj in qobjs):
        # for convenience, we call operator_to_vector on operators provided
        # in matrix form
        def _ensure_vector(qobj):
            if isinstance(qobj, (Qobj, QobjEvo)) and qobj.type == 'oper':
                return operator_to_vector(qobj).dag()
            return qobj
        return [[_ensure_vector(qobj) for qobj in qobjs]]

    if linear:
        raise ValueError("Invalid combination of Qobj types for direct sum.")

    allsuper = all(_is_like_super(qobj) for row in qobjs for qobj in row)
    alloper = all(_is_like_oper(qobj) for row in qobjs for qobj in row)
    if not (allsuper or alloper):
        raise ValueError(
            "Invalid combination of Qobj types for direct sum.")
    return qobjs


def _determine_dimensions(qobjs):
    num_columns = len(qobjs[0])
    num_rows = len(qobjs)

    from_spaces = [None] * num_columns
    to_spaces = [None] * num_rows
    block_widths = np.empty(num_columns, dtype=_data.base.idxint_dtype)
    block_heights = np.empty(num_rows, dtype=_data.base.idxint_dtype)

    for row in range(num_rows):
        for column in range(num_columns):
            if qobjs[row][column] is None:
                continue

            to_space = _qobj_dims(qobjs[row][column]).to_
            if to_spaces[row] is None:
                to_spaces[row] = to_space
                block_heights[row] = to_space.size
            elif to_spaces[row] != to_space:
                raise ValueError(
                    "Direct sum: inconsistent dimensions in row"
                    f" {row + 1}. Expected {to_spaces[row].as_list()},"
                    f" got {to_space.as_list()}.")

            from_space = _qobj_dims(qobjs[row][column]).from_
            if from_spaces[column] is None:
                from_spaces[column] = from_space
                block_widths[column] = from_space.size
            elif from_spaces[column] != from_space:
                raise ValueError(
                    "Direct sum: inconsistent dimensions in column"
                    f" {column + 1}. Expected {from_spaces[column].as_list()},"
                    f" got {from_space.as_list()}.")

    for row, space in enumerate(to_spaces):
        if space is None:
            raise ValueError(f"Direct sum: empty row {row + 1}.")
    for column, space in enumerate(from_spaces):
        if space is None:
            raise ValueError(f"Direct sum: empty column {column + 1}.")

    out_dims = Dimensions(SumSpace(*from_spaces), SumSpace(*to_spaces))
    return out_dims, block_widths, block_heights


@overload
def sparse_direct_sum(
    qobjs: dict[tuple[int, int], Qobj],
    sum_dimensions: DimensionLike,
    dtype: LayerType = None
) -> Qobj:
    ...


@overload
def sparse_direct_sum(
    qobjs: dict[tuple[int, int], Qobj | QobjEvo],
    sum_dimensions: DimensionLike,
    dtype: LayerType = None
) -> QobjEvo:
    ...


def sparse_direct_sum(qobjs, sum_dimensions, dtype=None):
    if settings.core["default_dtype_scope"] == "full":
        dtype = _data._parse_default_dtype(dtype, "sparse")
    else:
        dtype = _data._parse_default_dtype(dtype or "core", "sparse")

    sum_dimensions = Dimensions(sum_dimensions)
    if isinstance(sum_dimensions.from_, SumSpace):
        from_spaces = sum_dimensions.from_.spaces
        block_widths = np.array(sum_dimensions.from_._space_dims,
                                dtype=_data.base.idxint_dtype)
        block_cumwidths = sum_dimensions.from_._space_cumdims
    else:
        from_spaces = [sum_dimensions.from_]
        block_widths = np.array([sum_dimensions.from_.size],
                                dtype=_data.base.idxint_dtype)
        block_cumwidths = [0, sum_dimensions.from_.size]
    if isinstance(sum_dimensions.to_, SumSpace):
        to_spaces = sum_dimensions.to_.spaces
        block_heights = np.array(sum_dimensions.to_._space_dims,
                                 dtype=_data.base.idxint_dtype)
        block_cumheights = sum_dimensions.to_._space_cumdims
    else:
        to_spaces = [sum_dimensions.to_]
        block_heights = np.array([sum_dimensions.to_.size],
                                 dtype=_data.base.idxint_dtype)
        block_cumheights = [0, sum_dimensions.to_.size]

    num_non_evo = sum(isinstance(qobj, Qobj) for qobj in qobjs.values())
    qobjevos = []

    block_rows = np.empty((num_non_evo,), dtype=_data.base.idxint_dtype)
    block_cols = np.empty((num_non_evo,), dtype=_data.base.idxint_dtype)
    blocks = np.empty((num_non_evo,), dtype=_data.Data)

    places = list(qobjs.keys())
    places.sort()

    i = 0
    for row, column in places:
        qobj = qobjs[(row, column)]
        _check_bounds(row, 0, len(to_spaces))
        _check_bounds(column, 0, len(from_spaces))
        if (qobj._dims.to_ != to_spaces[row] or
                qobj._dims.from_ != from_spaces[column]):
            raise ValueError("Direct sum: dimension mismatch for component at"
                             f" ({row}, {column}).")

        if isinstance(qobj, Qobj):
            block_rows[i] = row
            block_cols[i] = column
            blocks[i] = _qobj_data(qobj, dtype)
            i += 1
            continue

        # Handle QobjEvo (same as in direct_sum)
        blow_up = qobj.linear_map(
            lambda x: Qobj(
                _data.insert(_data.zeros[dtype](*sum_dimensions.shape), x.data,
                             block_cumheights[row], block_cumwidths[column],
                             dtype=dtype),
                dims=sum_dimensions, copy=False),
            _skip_check=True)
        qobjevos.append(blow_up)

    out_data = _data.spconcat(block_rows, block_cols, blocks,
                              block_widths, block_heights, dtype=dtype)
    result = Qobj(out_data, dims=sum_dimensions, copy=False)
    return sum(qobjevos, start=result)


@overload
def component(sum_qobj: Qobj, *index: int) -> Qobj:
    ...


@overload
def component(sum_qobj: QobjEvo, *index: int) -> Qobj | QobjEvo:
    ...


def component(sum_qobj, *index):
    """
    Extracts component at index from qobj which is a direct sum.
    """
    if isinstance(sum_qobj, QobjEvo):
        result = sum_qobj.linear_map(lambda x: component(x, *index),
                                     _skip_check=True)
        result.compress()
        return result(0) if result.isconstant else result

    to_index, from_index = _check_component_index(sum_qobj._dims, index)
    (component_to, to_data_start, to_data_stop,
     component_from, from_data_start, from_data_stop) =\
        _component_info(sum_qobj._dims, to_index, from_index)

    out_data = _data.slice(sum_qobj.data,
                           to_data_start, to_data_stop,
                           from_data_start, from_data_stop)
    return Qobj(
        out_data, dims=Dimensions(component_from, component_to), copy=False
    )


@overload
def set_component(
    sum_qobj: Qobj,
    component: Qobj,
    *index: int,
    dtype: LayerType = None
) -> Qobj:
    ...


@overload
def set_component(
    sum_qobj: Qobj | QobjEvo,
    component: Qobj | QobjEvo,
    *index: int,
    dtype: LayerType = None
) -> QobjEvo:
    ...


def set_component(sum_qobj, component, *index, dtype=None):
    """
    Sets the component of the direct sum qobjs at the given index.
    """

    if settings.core["default_dtype_scope"] == "full":
        dtype = dtype or settings.core["default_dtype"] or sum_qobj.dtype
    else:
        dtype = dtype or sum_qobj.dtype

    to_index, from_index = _check_component_index(sum_qobj._dims, index)
    (component_to, to_data_start, _, component_from, from_data_start, _) =\
        _component_info(sum_qobj._dims, to_index, from_index)

    if component is not None and (
        _qobj_dims(component).to_ != component_to
        or _qobj_dims(component).from_ != component_from
    ):
        expected = Dimensions(component_from, component_to)
        raise ValueError("Canot set component of direct sum: dimension"
                         f" mismatch. Expected: {expected.as_list()},"
                         f" got: {_qobj_dims(component).as_list()}.")

    if not(isinstance(sum_qobj, QobjEvo) or isinstance(component, QobjEvo)):
        # get the easy case out of the way
        # all other cases will be broken down to this
        component_data = (
            _qobj_data(component, dtype) or
            _data.zeros[dtype](component_to.size, component_from.size)
        )
        out_data = _data.insert(sum_qobj.data, component_data,
                                to_data_start, from_data_start, dtype=dtype)
        return Qobj(out_data, dims=sum_qobj._dims, copy=False)

    # if QobjEvo is involved, the result is the sum of two parts:
    # - the sum_qobj with the component zeroed
    # - the new component embedded into a large zero matrix
    # all "set_component" below act on simple Qobj

    if isinstance(sum_qobj, QobjEvo):
        zeroed = sum_qobj.linear_map(
            lambda x: set_component(x, qzero_like(component), *index,
                                    dtype=dtype),
            _skip_check=True)
        zeroed.compress()
        zeroed = zeroed(0) if zeroed.isconstant else zeroed
    else:
        zeroed = set_component(sum_qobj, qzero_like(component), *index,
                               dtype=dtype)

    zeroes_like_sum = Qobj(
        _data.zeros[component.dtype](*sum_qobj._dims.shape),
        dims=sum_qobj._dims, copy=False
    )
    if isinstance(component, QobjEvo):
        blow_up = component.linear_map(
            lambda x: set_component(zeroes_like_sum, x, *index,
                                    dtype=dtype),
            _skip_check=True)
    else:
        blow_up = set_component(zeroes_like_sum, component, *index,
                                dtype=dtype)

    return zeroed + blow_up


def _check_bounds(given, min, max):
    if not (min <= given < max):
        raise ValueError(f"Index ({given}) out of bounds ({min}, {max-1})"
                         " for component of direct sum.")


def _check_component_index(sum_dims, index):
    is_to_sum = (
        isinstance(sum_dims.to_, SumSpace) and len(sum_dims.to_.spaces) > 1
    )
    is_from_sum = (
        isinstance(sum_dims.from_, SumSpace) and len(sum_dims.from_.spaces) > 1
    )
    if not is_to_sum and not is_from_sum:
        raise ValueError("Qobj is not a direct sum.")

    if is_to_sum and is_from_sum:
        if len(index) != 2:
            raise ValueError("Invalid number of indices provided for component"
                             " of direct sum (two indices required).")
    else:
        if len(index) == 1 and is_to_sum:
            index = (index[0], 0)
        elif len(index) == 1 and is_from_sum:
            index = (0, index[0])
        if len(index) != 2:
            raise ValueError("Invalid number of indices provided for component"
                             " of direct sum (one or two indices required).")

    return index

def _component_info(sum_dims, to_index, from_index):
    if isinstance(sum_dims.to_, SumSpace):
        _check_bounds(to_index, 0, len(sum_dims.to_.spaces))
        component_to = sum_dims.to_.spaces[to_index]
        to_data_start = sum_dims.to_._space_cumdims[to_index]
        to_data_stop = sum_dims.to_._space_cumdims[to_index + 1]
    else:
        _check_bounds(to_index, 0, 1)
        component_to = sum_dims.to_
        to_data_start = 0
        to_data_stop = sum_dims.to_.size

    if isinstance(sum_dims.from_, SumSpace):
        _check_bounds(from_index, 0, len(sum_dims.from_.spaces))
        component_from = sum_dims.from_.spaces[from_index]
        from_data_start = sum_dims.from_._space_cumdims[from_index]
        from_data_stop = sum_dims.from_._space_cumdims[from_index + 1]
    else:
        _check_bounds(from_index, 0, 1)
        component_from = sum_dims.from_
        from_data_start = 0
        from_data_stop = sum_dims.from_.size

    return (component_to, to_data_start, to_data_stop,
            component_from, from_data_start, from_data_stop)
