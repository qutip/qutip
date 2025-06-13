#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from . import base, convert, csr
from . cimport base, CSR, Data, Dense

import numpy as np
cimport numpy as cnp


__all__ = [
    'concat_dense', 'spconcat_dense', 'slice_dense', 'insert_dense',
    'concat_csr', 'spconcat_csr', 'slice_csr', 'insert_csr',
    'concat', 'spconcat', 'slice', 'insert',
]


cdef base.idxint[:] _cumsum(base.idxint[:] array):
    """Cumulative sum of array entries, starting with zero."""
    cdef base.idxint i, n = len(array)
    cdef base.idxint[:] out = np.empty(n + 1, dtype=base.idxint_dtype)
    out[0] = 0
    for i in range(n):
        out[i + 1] = out[i] + array[i]
    return out


cpdef Dense concat_dense(
    Data[:,:] data_array,
    base.idxint[:] block_widths,
    base.idxint[:] block_heights
):
    cdef base.idxint row, column
    cdef Data op

    if len(block_widths) == 0 or len(block_heights) == 0:
        raise ValueError("Cannot concatenate empty data array.")

    if (data_array.shape[0] != len(block_heights) or
            data_array.shape[1] != len(block_widths)):
        raise ValueError("Wrong number of widths or heights supplied "
                         "for data concatenation.")

    blocks = [[None] * len(block_widths) for _ in range(len(block_heights))]
    for row in range(len(block_heights)):
        for column in range(len(block_widths)):
            op = data_array[row][column]
            if op is None:
                blocks[row][column] = np.zeros(
                    (block_heights[row], block_widths[column])
                )
            elif type(op) is Dense:
                blocks[row][column] = op.as_ndarray()
            else:
                blocks[row][column] = op.to_array()

    return Dense(np.block(blocks), copy=None)


cdef class _getitem:
    # Using `_getitem(data_array)` instead of `lambda r, c: data_array[r, c]`
    # in `concat_csr` below, because the lambda crashes the cython compiler
    cdef readonly Data[:,:] array

    def __init__(self, Data[:,:] array):
        self.array = array

    def __call__(self, base.idxint row, base.idxint column):
        return self.array[row, column]


cpdef CSR concat_csr(
    Data[:,:] data_array,
    base.idxint[:] block_widths,
    base.idxint[:] block_heights
):
    if len(block_widths) == 0 or len(block_heights) == 0:
        raise ValueError("Cannot concatenate empty data array.")

    if (data_array.shape[0] != len(block_heights) or
            data_array.shape[1] != len(block_widths)):
        raise ValueError("Wrong number of widths or heights supplied "
                         "for data concatenation.")

    # delegate to spconcat_csr
    block_rows = []
    block_cols = []

    cdef base.idxint row, column
    for row in range(len(block_heights)):
        for column in range(len(block_widths)):
            if data_array[row][column] is not None:
                block_rows.append(row)
                block_cols.append(column)

    return spconcat_csr(
        np.array(block_rows, dtype=base.idxint_dtype),
        np.array(block_cols, dtype=base.idxint_dtype),
        _getitem(data_array),
        block_widths, block_heights
    )


cpdef Dense spconcat_dense(
    base.idxint[:] block_rows, base.idxint[:] block_cols, block_generator,
    base.idxint[:] block_widths, base.idxint[:] block_heights
):
    if len(block_rows) != len(block_cols):
        raise ValueError(
            "The arrays block_rows and block_cols must have the same length."
        )

    cdef base.idxint[:] row_pos = _cumsum(block_heights)
    cdef base.idxint[:] col_pos = _cumsum(block_widths)
    cdef base.idxint shape1 = row_pos[len(block_heights)]
    cdef base.idxint shape2 = col_pos[len(block_widths)]

    if shape1 == 0 or shape2 == 0:
        raise ValueError("Cannot concatenate empty data array.")

    cdef base.idxint i, block_height, block_width, above, before
    cdef Data block
    cdef cnp.ndarray block_array
    cdef cnp.ndarray result = np.zeros((shape1, shape2), dtype=complex)

    for i in range(len(block_rows)):
        above = row_pos[block_rows[i]]
        before = col_pos[block_cols[i]]
        block = block_generator(block_rows[i], block_cols[i])
        block_height, block_width = block.shape
        if type(block) is Dense:
            block_array = block.as_ndarray()
        else:
            block_array = block.to_array()
        result[above:(above+block_height),
               before:(before+block_width)] = block_array

    return Dense(result, copy=False)


cpdef CSR spconcat_csr(
    base.idxint[:] block_rows, base.idxint[:] block_cols, block_generator,
    base.idxint[:] block_widths, base.idxint[:] block_heights
):
    if len(block_rows) != len(block_cols):
        raise ValueError(
            "The arrays block_rows and block_cols must have the same length."
        )

    cdef base.idxint num_ops = len(block_rows)
    cdef base.idxint[:] row_pos = _cumsum(block_heights)
    cdef base.idxint[:] col_pos = _cumsum(block_widths)
    cdef base.idxint shape1 = row_pos[len(block_heights)]
    cdef base.idxint shape2 = col_pos[len(block_widths)]

    if shape1 == 0 or shape2 == 0:
        raise ValueError("Cannot concatenate empty data array.")

    if num_ops == 0:
        return csr.zeros(shape1, shape2)

    cdef base.idxint idx, row, column, nnz = 0
    cdef cnp.ndarray ops = np.empty((num_ops,), dtype=CSR)
    cdef Data op

    for idx in range(num_ops):
        row = block_rows[idx]
        column = block_cols[idx]
        # check ops are ordered by (row, column)
        if idx > 0 and (
            row < block_rows[idx - 1] or
            (row == block_rows[idx - 1] and column < block_cols[idx - 1])
        ):
            raise ValueError("The arrays block_rows and block_cols must be "
                             "sorted by (row, column).")

        # check op shape, convert to CSR if needed, calculate nnz
        op = block_generator(row, column)
        if op.shape != (block_heights[row], block_widths[column]):
            raise ValueError(
                "Block operators do not have the correct shape."
            )

        if type(op) is not CSR:
            op = convert.to(CSR, op)
        ops[idx] = <CSR>op
        nnz += csr.nnz(op)

    if nnz == 0:
        return csr.zeros(shape1, shape2)

    cdef CSR out = csr.empty(shape1, shape2, nnz)
    cdef CSR csr_op

    idx = 0
    cdef base.idxint prev_idx, counter, end = 0
    cdef base.idxint op_row, op_row_start, op_row_end, op_row_len

    out.row_index[0] = 0

    for row in range(len(block_heights)):
        prev_idx = idx
        while idx < num_ops:
            if block_rows[idx] != row:
                break
            idx += 1
        # now the operators in the current row have ids (prev_idx, ..., idx-1)

        for op_row in range(block_heights[row]):
            for i in range(prev_idx, idx):
                csr_op = ops[i]
                if csr.nnz(op) == 0:
                    # empty CSR matrices have uninitialized row_index entries.
                    # it's unclear whether users should ever see such matrixes
                    # but we support them here anyway.
                    continue

                column = block_cols[i]
                op_row_start = csr_op.row_index[op_row]
                op_row_end = csr_op.row_index[op_row + 1]
                op_row_len = op_row_end - op_row_start
                for counter in range(op_row_len):
                    out.col_index[end + counter] = (
                        csr_op.col_index[op_row_start + counter] +
                        col_pos[column]
                    )
                    out.data[end + counter] =\
                        csr_op.data[op_row_start + counter]
                end += op_row_len
            out.row_index[row_pos[row] + op_row + 1] = end

    return out


cpdef Dense slice_dense(Dense data,
                        base.idxint row_start, base.idxint row_stop,
                        base.idxint col_start, base.idxint col_stop):
    cdef cnp.ndarray array = data.as_ndarray()
    return Dense(array[row_start:row_stop, col_start:col_stop], copy=True)


cpdef CSR slice_csr(CSR data,
                    base.idxint row_start, base.idxint row_stop,
                    base.idxint col_start, base.idxint col_stop):
    scipy = data.as_scipy()
    return CSR(scipy[row_start:row_stop, col_start:col_stop], copy=True)


cpdef Dense insert_dense(Data data, Dense block,
                         base.idxint above, base.idxint before):
    cdef base.idxint data_height, data_width, block_height, block_width

    data_height, data_width = data.shape
    block_height, block_width = block.shape
    if above + block_height > data_height or before + block_width > data_width:
        raise IndexError("Cannot insert block into data: doesn't fit.")

    cdef cnp.ndarray data_array = data.to_array()  # copies
    cdef cnp.ndarray block_array = block.as_ndarray()  # doesn't copy

    data_array[above:(above+block_height),
               before:(before+block_width)] = block_array  # copies block
    return Dense(data_array, copy=False)


cpdef CSR insert_csr(CSR data, CSR block,
                     base.idxint above, base.idxint before):
    cdef base.idxint data_height, data_width, block_height, block_width

    data_height, data_width = data.shape
    block_height, block_width = block.shape
    if above + block_height > data_height or before + block_width > data_width:
        raise IndexError("Cannot insert block into data: doesn't fit.")

    data_scipy = data.as_scipy().copy()
    block_scipy = block.as_scipy()

    import warnings, scipy
    with warnings.catch_warnings():
        # TODO find a better way
        warnings.filterwarnings(
            "ignore", category=scipy.sparse._base.SparseEfficiencyWarning)
        data_scipy[above:(above+block_height),
                before:(before+block_width)] = block_scipy
    return CSR(data_scipy, copy=False)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect


concat = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('data_array',
                           _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('block_widths',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('block_heights',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='concat',
    module=__name__,
    inputs=(),
    out=True,
)
concat.__doc__ =\
    """
    Concatenates a 2D array of data blocks into a block matrix.

    Parameters
    ----------
    data_array : Data[:,:]
        A 2D array of data blocks to concatenate.
        Within each row of blocks, the heights of the blocks must be the same.
        Within each column of blocks, the widths of the blocks must be the
        same.

    block_widths : int[:]
        Array containing the block widths. The length of this array must be
        equal to the number of columns, i.e., ``data_array.shape[1]``.

    block_heights : int[:]
        Array containing the block heights. The length of this array must be
        equal to the number of rows, i.e., ``data_array.shape[0]``.
    """
concat.add_specialisations([
    (Dense, concat_dense),
    (CSR, concat_csr),
], _defer=True)


spconcat = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('block_rows',
                           _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('block_cols',
                           _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('block_generator',
                           _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('block_widths',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('block_heights',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='spconcat',
    module=__name__,
    inputs=(),
    out=True,
)
spconcat.__doc__ =\
    """
    Concatenates data blocks into a block matrix.
    In contrast to ``concat``, this function is intended for large block
    matrices where most of the blocks are empty. The coordinates of the
    non-empty blocks are given by the arrays ``block_rows`` and ``block_cols``,
    and the blocks themselves are generated by the ``block_generator``
    function.

    The generator function takes two arguments, ``row`` and ``column``, and
    returns a ``Data`` object representing the block at that position. The
    shape of the block should be
    ``(block_heights[row], ``block_widths[column])``. The shape of the output
    will be ``(sum(block_heights), sum(block_widths))``.

    Parameters
    ----------
    block_rows : int[:]
        The block row for each operator. The block row should be in
        ``range(0, len(block_heights))``.

    block_cols : int[:]
        The block column for each operator. The block column should be in
        ``range(0, len(block_widths))``.

    generator : Callable[[int, int] -> Data]
        Function generating the blocks. Will only be called with arguments
        ``(row, column)`` that are in ``zip(block_rows, block_cols)``.

    block_widths : int[:]
        Array containing the block widths.

    block_heights : int[:]
        Array containing the block heights.
    """
spconcat.add_specialisations([
    (CSR, spconcat_csr),
    (Dense, spconcat_dense),
], _defer=True)


slice = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('data', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('row_start',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('row_stop',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('col_start',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('col_stop',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='slice',
    module=__name__,
    inputs=('data',),
    out=False,
)
slice.__doc__ =\
    """
    Extracts a block of data from a large matrix.
    """
slice.add_specialisations([
    (Dense, slice_dense),
    (CSR, slice_csr),
], _defer=True)


insert = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('data', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('block', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('above', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('before', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='insert',
    module=__name__,
    inputs=('data', 'block'),
    out=True,
)
insert.__doc__ =\
    """
    Inserts a block of data into a large matrix.
    """
insert.add_specialisations([
    (Data, Dense, Dense, insert_dense),
    (CSR, CSR, CSR, insert_csr),
], _defer=True)


del _inspect, _Dispatcher
