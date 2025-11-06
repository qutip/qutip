from . import data as _data
from .cy.qobjevo import QobjEvo
from .dimensions import Dimensions, Field, SumSpace
from .qobj import Qobj
from .superoperator import operator_to_vector

from .. import settings
from ..typing import DimensionLike, LayerType

from functools import partial
from numbers import Number
from typing import Any, Union, overload

import numpy as np


QobjLike = Union[Number, Qobj, QobjEvo]


__all__ = ['direct_sum', 'direct_sum_sparse']


def _is_like_ket(qobj: Any) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'ket']
        ))


def _is_like_bra(qobj: Any) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'bra']
        ))


def _is_like_oper(qobj: Any) -> bool:
    return (
        qobj is None
        or isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'oper', 'ket', 'bra']
        ))


def _is_like_operator_ket(qobj: Any) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'oper', 'operator-ket']
        ))


def _is_like_operator_bra(qobj: Any) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'oper', 'operator-bra']
        ))


def _is_like_super(qobj: Any) -> bool:
    return (
        qobj is None
        or isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in [
                'scalar', 'super', 'operator-ket', 'operator-bra']
        ))


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


@overload
def direct_sum(
    qobjs: list[Number | Qobj] | list[list[Number | Qobj]],
    dtype: LayerType = None
) -> Qobj:
    ...


@overload
def direct_sum(
    qobjs: list[QobjLike] | list[list[QobjLike]],
    dtype: LayerType = None
) -> QobjEvo:
    ...


def direct_sum(qobjs, dtype=None):
    # TODO: docstring
    # TODO everywhere to_ from_

    if len(qobjs) == 0:
        raise ValueError("No Qobjs provided for direct sum.")

    # Step 1: check type of all provided qobj and make (row, col) -> qobj dict

    linear = isinstance(qobjs[0], QobjLike)
    if not linear and len(qobjs[0]) == 0:
        raise ValueError("No Qobjs provided for direct sum.")
    if not linear and not all(len(row) == len(qobjs[0]) for row in qobjs):
        raise ValueError("Matrix of Qobjs in direct sum must be square.")

    if linear and all(_is_like_ket(qobj) for qobj in qobjs):
        num_rows = len(qobjs)
        num_columns = 1
        qobj_dict = {(row, 0): qobj for row, qobj in enumerate(qobjs)}

    elif linear and all(_is_like_operator_ket(qobj) for qobj in qobjs):
        # for convenience, we call operator_to_vector on operators provided
        # in matrix form
        def _ensure_vector(qobj):
            if isinstance(qobj, (Qobj, QobjEvo)) and qobj.type == 'oper':
                return operator_to_vector(qobj)
            return qobj

        num_rows = len(qobjs)
        num_columns = 1
        qobj_dict = {(row, 0): _ensure_vector(qobj)
                     for row, qobj in enumerate(qobjs)}

    elif linear and all(_is_like_bra(qobj) for qobj in qobjs):
        num_rows = 1
        num_columns = len(qobjs)
        qobj_dict = {(0, col): qobj for col, qobj in enumerate(qobjs)}

    elif linear and all(_is_like_operator_bra(qobj) for qobj in qobjs):
        # for convenience, we call operator_to_vector on operators provided
        # in matrix form
        def _ensure_vector(qobj):
            if isinstance(qobj, (Qobj, QobjEvo)) and qobj.type == 'oper':
                return operator_to_vector(qobj).dag()
            return qobj

        num_rows = 1
        num_columns = len(qobjs)
        qobj_dict = {(0, col): _ensure_vector(qobj)
                     for col, qobj in enumerate(qobjs)}

    elif linear:
        raise ValueError("Invalid combination of Qobj types for direct sum.")

    else:
        allsuper = all(_is_like_super(qobj) for row in qobjs for qobj in row)
        alloper = all(_is_like_oper(qobj) for row in qobjs for qobj in row)
        if not (allsuper or alloper):
            raise ValueError(
                "Invalid combination of Qobj types for direct sum.")

        num_rows = len(qobjs)
        num_columns = len(qobjs[0])
        qobj_dict = {(row, col): qobjs[row][col]
                     for row in range(num_rows) for col in range(num_columns)
                     if qobjs[row][col] is not None}

    # Step 2: determine dimensions of sum space

    to_spaces = [None] * num_rows
    from_spaces = [None] * num_columns

    for (row, col), qobj in qobj_dict.items():
        to_space = _qobj_dims(qobj)[0]
        if to_spaces[row] is None:
            to_spaces[row] = to_space
        elif to_spaces[row] != to_space:
            raise ValueError(
                "Direct sum: inconsistent dimensions in row"
                f" {row + 1}. Expected {to_spaces[row].as_list()},"
                f" got {to_space.as_list()}.")

        from_space = _qobj_dims(qobj)[1]
        if from_spaces[col] is None:
            from_spaces[col] = from_space
        elif from_spaces[col] != from_space:
            raise ValueError(
                "Direct sum: inconsistent dimensions in column"
                f" {col + 1}. Expected {from_spaces[col].as_list()},"
                f" got {from_space.as_list()}.")

    for row, space in enumerate(to_spaces):
        if space is None:
            raise ValueError(f"Direct sum: empty row {row + 1}.")
    for col, space in enumerate(from_spaces):
        if space is None:
            raise ValueError(f"Direct sum: empty column {col + 1}.")

    sum_dimension = Dimensions(SumSpace(*from_spaces), SumSpace(*to_spaces))

    # Step 3: create direct sum

    return _do_direct_sum(qobj_dict, sum_dimension, dtype)


@overload
def direct_sum_sparse(
    qobjs: dict[tuple[int, int], Number | Qobj],
    sum_dimensions: DimensionLike,
    dtype: LayerType = None
) -> Qobj:
    ...


@overload
def direct_sum_sparse(
    qobjs: dict[tuple[int, int], QobjLike],
    sum_dimensions: DimensionLike,
    dtype: LayerType = None
) -> QobjEvo:
    ...


def direct_sum_sparse(qobjs, sum_dimensions, dtype=None):
    # TODO: docstring

    sum_dimensions = Dimensions(sum_dimensions)
    to_spaces, from_spaces = _spaces_from_dims(sum_dimensions)

    for (row, col), qobj in qobjs.items():
        _check_bounds(row, 0, len(to_spaces))
        _check_bounds(col, 0, len(from_spaces))
        if (
            _qobj_dims(qobj).to_ != to_spaces[row] or
            _qobj_dims(qobj).from_ != from_spaces[col]
        ):
            raise ValueError("Direct sum: dimension mismatch for component at"
                             f" ({row}, {col}).")

    return _do_direct_sum(dict(sorted(qobjs.items())), sum_dimensions, dtype)


def _do_direct_sum(
        qobjs: dict[tuple[int, int], QobjLike],
        sum_dimensions: Dimensions,
        dtype: LayerType = None
):
    # qobjs is assumed to be sorted and type checked
    # sum_dimensions is assumed to be between sumspaces
    # the blocks are assumed to fit
    # TODO: delete this comment

    # TODO: on "bofin-direct-sum" branch, I had a version of this where the
    # QobjEvo are first decomposed into their elements, and grouped by
    # coefficients --> check if that would be better for performance

    # TODO: trim down implementation note in dimensions.py

    if settings.core["default_dtype_scope"] == "full":
        dtype = _data._parse_default_dtype(dtype, "sparse")
    else:
        dtype = _data._parse_default_dtype(dtype or "core", "sparse")

    nonevo_count = sum(
        not isinstance(qobj, QobjEvo) for qobj in qobjs.values()
    )
    block_rows = np.empty((nonevo_count,), dtype=_data.base.idxint_dtype)
    block_cols = np.empty((nonevo_count,), dtype=_data.base.idxint_dtype)
    blocks = np.empty((nonevo_count,), dtype=_data.Data)

    i = 0
    qobjevos = []
    for (row, col), qobj in qobjs.items():
        if not isinstance(qobj, QobjEvo):
            block_rows[i] = row
            block_cols[i] = col
            blocks[i] = _qobj_data(qobj, dtype)
            i += 1
        else:
            qobjevos.append((row, col, qobj))

    if isinstance(sum_dimensions[0], SumSpace):
        block_heights = np.array(
            sum_dimensions[0]._space_dims, dtype=_data.base.idxint_dtype)
    else:
        block_heights = np.full(
            (1,), sum_dimensions[0].size, dtype=_data.base.idxint_dtype)
    if isinstance(sum_dimensions[1], SumSpace):
        block_widths = np.array(
            sum_dimensions[1]._space_dims, dtype=_data.base.idxint_dtype)
    else:
        block_widths = np.full(
            (1,), sum_dimensions[1].size, dtype=_data.base.idxint_dtype)

    result = Qobj(
        _data.block_build(
            block_rows, block_cols, blocks,
            block_heights, block_widths, dtype=dtype
        ), dims=sum_dimensions, copy=False
    )

    for row, col, qobjevo in qobjevos:
        result += qobjevo.linear_map(
            partial(_blow_up_qobj, sum_dimensions=sum_dimensions,
                    row=row, col=col, dtype=dtype),
            _skip_check=True
        )
    if qobjevos:
        result.compress()
    return result


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

    if settings.core["default_dtype_scope"] == "full":
        dtype = settings.core["default_dtype"] or sum_qobj.dtype
    else:
        dtype = sum_qobj.dtype

    if isinstance(sum_qobj, QobjEvo):
        result = sum_qobj.linear_map(lambda x: component(x, *index),
                                     _skip_check=True)
        result.compress()
        return result(0) if result.isconstant else result

    to_index, from_index = _check_component_index(sum_qobj._dims, index)
    component_dims, row_start, row_stop, col_start, col_stop =\
        _component_info(sum_qobj._dims, to_index, from_index)

    out_data = _data.block_extract(
        sum_qobj.data, row_start, row_stop, col_start, col_stop, dtype=dtype
    )
    return Qobj(out_data, dims=component_dims, copy=False)


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
    component_dims, row_start, _, col_start, _ =\
        _component_info(sum_qobj._dims, to_index, from_index)

    if component is not None and _qobj_dims(component) != component_dims:
        raise ValueError("Canot set component of direct sum: dimension"
                         f" mismatch. Expected: {component_dims.as_list()},"
                         f" got: {_qobj_dims(component).as_list()}.")

    if not (isinstance(sum_qobj, QobjEvo) or isinstance(component, QobjEvo)):
        component_data = (
            _qobj_data(component, dtype)
            or _data.zeros[dtype](*component_dims.shape)
        )
        out_data = _data.block_overwrite(sum_qobj.data, component_data,
                                         row_start, col_start, dtype=dtype)
        return Qobj(out_data, dims=sum_qobj._dims, copy=False)

    # If QobjEvo is involved, the result is the sum of two parts:
    # - the sum_qobj with the component zeroed
    # - the new component embedded into a large zero matrix
    # Note that all "set_component" below act on simple Qobj

    if isinstance(sum_qobj, QobjEvo):
        zeroed = sum_qobj.linear_map(
            lambda x: set_component(x, None, *index, dtype=dtype),
            _skip_check=True)
        zeroed.compress()
        zeroed = zeroed(0) if zeroed.isconstant else zeroed
    else:
        zeroed = set_component(sum_qobj, None, *index, dtype=dtype)

    blow_up_func = partial(_blow_up_qobj, sum_dimensions=sum_qobj._dims,
                           row=to_index, col=from_index, dtype=dtype)
    if isinstance(component, QobjEvo):
        blow_up = component.linear_map(blow_up_func, _skip_check=True)
    else:
        blow_up = blow_up_func(component)

    return zeroed + blow_up


def _blow_up_qobj(x: Qobj, *, sum_dimensions, row, col, dtype):
    # given small Qobj x, makes big Qobj with given dimensions
    # resulting big Qobj is zero except for the block x starting at row, col

    if isinstance(sum_dimensions[0], SumSpace):
        data_row = sum_dimensions[0]._space_cumdim(row)
    else:
        data_row = 0

    if isinstance(sum_dimensions[1], SumSpace):
        data_col = sum_dimensions[1]._space_cumdim(col)
    else:
        data_col = 0

    return Qobj(
        _data.block_overwrite(
            _data.zeros[dtype](*sum_dimensions.shape),
            x.data, data_row, data_col, dtype=dtype
        ), dims=sum_dimensions, copy=False
    )


def _check_bounds(given, min, max):
    if not (min <= given < max):
        raise ValueError(f"Index ({given}) out of bounds ({min}, {max-1})"
                         " for component of direct sum.")


def _spaces_from_dims(sum_dimensions):
    if isinstance(sum_dimensions[0], SumSpace):
        to_spaces = sum_dimensions[0].spaces
    else:
        to_spaces = [sum_dimensions[0]]
    if isinstance(sum_dimensions[1], SumSpace):
        from_spaces = sum_dimensions[1].spaces
    else:
        from_spaces = [sum_dimensions[1]]
    return to_spaces, from_spaces


def _check_component_index(sum_dimensions, index):
    """
    Check that an appropriate number of indices is provided for the given
    dimensions. 2 indices is always okay, 1 index is also okay if only one of
    the spaces is a sum. Returns tuple (to_index, from_index) of 2 indices.
    """
    to_spaces, from_spaces = _spaces_from_dims(sum_dimensions)
    is_to_sum = len(to_spaces) > 1
    is_from_sum = len(from_spaces) > 1

    if is_to_sum and is_from_sum:
        if len(index) != 2:
            raise ValueError("Invalid number of indices provided for component"
                             " of direct sum (two indices required).")
    else:
        if len(index) == 1 and is_to_sum:
            index = (index[0], 0)
        elif len(index) == 1:
            index = (0, index[0])
        if len(index) != 2:
            raise ValueError("Invalid number of indices provided for component"
                             " of direct sum (one or two indices required).")

    return index


def _component_info(sum_dimensions, to_index, from_index):
    """
    For the component (block) indicated by the indices, return
    * the Dimensions of the block
    * the index of the first row of the block in the full data array
    * the index of the row below the block in the full data array
    * the index of the first column of the block in the full data array
    * the index of the column after the block in the full data array
    """
    if isinstance(sum_dimensions[0], SumSpace):
        _check_bounds(to_index, 0, len(sum_dimensions[0].spaces))
        component_to = sum_dimensions[0].spaces[to_index]
        data_row_start = sum_dimensions[0]._space_cumdim(to_index)
        data_row_stop = sum_dimensions[0]._space_cumdim(to_index + 1)
    else:
        _check_bounds(to_index, 0, 1)
        component_to = sum_dimensions[0]
        data_row_start = 0
        data_row_stop = sum_dimensions[0].size

    if isinstance(sum_dimensions[1], SumSpace):
        _check_bounds(from_index, 0, len(sum_dimensions[1].spaces))
        component_from = sum_dimensions[1].spaces[from_index]
        data_col_start = sum_dimensions[1]._space_cumdim(from_index)
        data_col_stop = sum_dimensions[1]._space_cumdim(from_index + 1)
    else:
        _check_bounds(from_index, 0, 1)
        component_from = sum_dimensions[1]
        data_col_start = 0
        data_col_stop = sum_dimensions[1].size

    return (Dimensions(component_from, component_to),
            data_row_start, data_row_stop,
            data_col_start, data_col_stop)
