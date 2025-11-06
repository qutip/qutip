# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

from . import data as _data
from .cy.qobjevo import QobjEvo
from .dimensions import Dimensions, Field, SumSpace
from .operators import qzero
from .qobj import Qobj
from .superoperator import operator_to_vector

from .. import settings
from ..typing import DimensionLike, LayerType

from functools import partial
from numbers import Number
from typing import Any, Union, overload

import numpy as np


__all__ = ['direct_sum', 'direct_sum_sparse',
           'direct_component', 'set_direct_component']


QobjLike = Union[Number, Qobj, QobjEvo]


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
    """
    Construct the direct sum of the provided component quantum objects.
    The matrix representing the returned direct sum is a block matrix, where
    the matrices of the components are all concatenated.

    This function accepts either a 1D list of components or a 2D square matrix
    (list of lists) of components. A 1D list is interpreted as a column or row
    block vector, depending on the element types:

    * If all elements are kets or scalars, the output will be a ket.
    * If all elements are bras or scalars, the output will be a bra.
    * If all elements are operator-kets, operators, or scalars, the operators
      will be converted to operator-kets automatically, and the output will be
      an operator-ket.
    * If all elements are operator-bras, operators, or scalars, the operators
      will be converted to operator-bras automatically, and the output will be
      an operator-bra.

    Python numbers are always accepted as scalars.

    A 2D square matrix may contain any types of Qobj, but the horizontal and
    vertical dimensions must match in each block row and block column. In a 2D
    square matrix, entries may be ``None`` to indicate zero blocks. However,
    each block row and each block column must contain at least one not-``None``
    entry. For direct sums with all-zero rows or columns, consider using
    :func:`.direct_sum_sparse`.

    Parameters
    ----------
    qobjs : list
        1D list or 2D square matrix of components. The components may be Python
        numbers, :class:`.Qobj`, or :class:`.QobjEvo`.
    dtype : type or str, optional
        Storage representation for the output ``Qobj``. Any data-layer known
        to ``qutip.data.to`` is accepted.

    Returns
    -------
    direct_sum: :class:`.Qobj` or :class:`.QobjEvo`
        The assembled direct-sum object. Returns a time-dependent object
        (:class:`.QobjEvo`) if any input component is time-dependent.
    """

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
    r"""
    Construct the direct sum of the provided component quantum objects.
    This is a variant of :func:`.direct_sum` suitable for large, sparse block
    matrices. The caller must provide the dimensions of the direct sum, for
    example in list form as in :code:`[([2], [3]), ([2], [3])]` for an operator
    on :math:`\mathbb C^2 \oplus \mathbb C^3`.

    Parameters
    ----------
    qobjs : dict
        Mapping that assigns a component to the given (row, col) location in
        the block matrix. Missing locations are filled with zero blocks. The
        components may be Python numbers, :class:`.Qobj`, or :class:`.QobjEvo`.
    sum_dimensions : list of tuples or lists
        Dimensions of the resulting Qobj, like in the creation of a Qobj.
    dtype : type or str, optional
        Storage representation for the output ``Qobj``. Any data-layer known
        to ``qutip.data.to`` is accepted.

    Returns
    -------
    direct_sum: :class:`.Qobj` or :class:`.QobjEvo`
        The assembled direct-sum object. Returns a time-dependent object
        (:class:`.QobjEvo`) if any input component is time-dependent.
    """

    sum_dimensions = Dimensions(sum_dimensions)
    to_spaces, from_spaces = _spaces_from_dims(sum_dimensions)

    for (row, col), qobj in qobjs.items():
        _check_bounds(row, 0, len(to_spaces))
        _check_bounds(col, 0, len(from_spaces))
        if (
            _qobj_dims(qobj)[0] != to_spaces[row] or
            _qobj_dims(qobj)[1] != from_spaces[col]
        ):
            raise ValueError("Direct sum: dimension mismatch for component at"
                             f" ({row}, {col}).")

    return _do_direct_sum(dict(sorted(qobjs.items())), sum_dimensions, dtype)


def _do_direct_sum(
        qobjs: dict[tuple[int, int], QobjLike],
        sum_dimensions: Dimensions,
        dtype: LayerType = None
):
    """Assumes `qobjs` is sorted and performs no dimensions checks"""

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
def direct_component(sum_qobj: Qobj, *index: int) -> Qobj:
    ...


@overload
def direct_component(sum_qobj: QobjEvo, *index: int) -> Qobj | QobjEvo:
    ...


def direct_component(sum_qobj, *index):
    """
    Extract the component (block) at the given index from a direct sum
    ``Qobj``.

    Parameters
    ----------
    sum_qobj : :class:`.Qobj` or :class:`.QobjEvo`
        A direct sum object from which to extract a component.
    index : list of int
        One or two indices identifying the component to extract. If both the
        row and column spaces are sums, two indices (``row``, ``col``) are
        required. If only one side is a sum, a single index is accepted and
        interpreted as the index into the summed side.

    Returns
    -------
    component: :class:`.Qobj` or :class:`.QobjEvo`
        The extracted component.
    """

    if settings.core["default_dtype_scope"] == "full":
        dtype = settings.core["default_dtype"] or sum_qobj.dtype
    else:
        dtype = sum_qobj.dtype

    if isinstance(sum_qobj, QobjEvo):
        result = sum_qobj.linear_map(lambda x: direct_component(x, *index),
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
def set_direct_component(
    sum_qobj: Qobj,
    component: Qobj,
    *index: int,
    dtype: LayerType = None
) -> Qobj:
    ...


@overload
def set_direct_component(
    sum_qobj: Qobj | QobjEvo,
    component: Qobj | QobjEvo,
    *index: int,
    dtype: LayerType = None
) -> QobjEvo:
    ...


def set_direct_component(sum_qobj, component, *index):
    """
    Set (replace) a component in a direct sum ``Qobj``. The function returns a
    new object where the component at the given index is replaced with
    ``component``.

    Parameters
    ----------
    sum_qobj : :class:`.Qobj` or :class:`.QobjEvo`
        The direct sum object whose component will be replaced.
    component : :class:`.Qobj` or :class:`.QobjEvo`
        The new component to insert. ``None`` sets the component to zero.
    index : list of int
        One or two indices identifying the component to extract. If both the
        row and column spaces are sums, two indices (``row``, ``col``) are
        required. If only one side is a sum, a single index is accepted and
        interpreted as the index into the summed side.

    Returns
    -------
    updated: :class:`.Qobj` or :class:`.QobjEvo`
        The resulting direct sum object with the component set.
    """

    if settings.core["default_dtype_scope"] == "full":
        dtype = settings.core["default_dtype"] or sum_qobj.dtype
    else:
        dtype = sum_qobj.dtype

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
            lambda x: set_direct_component(x, None, *index),
            _skip_check=True)
        zeroed.compress()
        zeroed = zeroed(0) if zeroed.isconstant else zeroed
    else:
        zeroed = set_direct_component(sum_qobj, None, *index)

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

    if x is None:
        return qzero(sum_dimensions[0], sum_dimensions[1])

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
        raise IndexError(f"Index ({given}) out of bounds ({min}, {max-1})"
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
            raise IndexError("Invalid number of indices provided for component"
                             " of direct sum (two indices required).")
    else:
        if len(index) == 1 and is_to_sum:
            index = (index[0], 0)
        elif len(index) == 1:
            index = (0, index[0])
        if len(index) != 2:
            raise IndexError("Invalid number of indices provided for component"
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
