#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memcpy

from . import base, convert
from . cimport base, csr, CSR, Data, Dense

import numpy as np
cimport numpy as cnp


__all__ = [
    'concat_blocks_dense', 'slice_dense', 'insert_block_dense',
    'concat_blocks_csr', 'slice_csr', 'insert_block_csr',
    'concat_blocks', 'slice', 'insert_block',
]


cdef base.idxint[:] _cumsum(base.idxint[:] array):
    """Cumulative sum of array entries, starting with zero."""
    cdef base.idxint i, n = len(array)
    cdef base.idxint[:] out = np.empty(n + 1, dtype=base.idxint_dtype)
    out[0] = 0
    for i in range(n):
        out[i + 1] = out[i] + array[i]
    return out


cpdef Dense concat_blocks_dense(
    base.idxint[:] block_rows, base.idxint[:] block_cols, Data[:] blocks,
    base.idxint[:] block_widths, base.idxint[:] block_heights
):
    if len(block_rows) != len(block_cols) or len(block_rows) != len(blocks):
        raise ValueError("The arrays block_rows, block_cols and blocks must"
                         " have the same length.")

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
        block = blocks[i]
        block_height, block_width = block.shape
        if (
            block_height != block_heights[block_rows[i]]
            or block_width != block_widths[block_cols[i]]
        ):
            raise ValueError(
                f"Block operator {i} does not have the correct shape."
            )
        if type(block) is Dense:
            block_array = block.as_ndarray()
        else:
            block_array = block.to_array()
        result[above:(above+block_height),
               before:(before+block_width)] = block_array

    return Dense(result, copy=False)


cpdef CSR concat_blocks_csr(
    base.idxint[:] block_rows, base.idxint[:] block_cols, Data[:] blocks,
    base.idxint[:] block_widths, base.idxint[:] block_heights
):
    # check arrays are the same length
    if len(block_rows) != len(block_cols) or len(block_rows) != len(blocks):
        raise ValueError("The arrays block_rows, block_cols and blocks must"
                         " have the same length.")

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
    cdef Data block

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

        # check block shape, convert to CSR if needed, calculate nnz
        block = blocks[idx]
        if (
            block.shape[0] != block_heights[row]
            or block.shape[1] != block_widths[column]
        ):
            raise ValueError(
                f"Block operator does not have the correct shape at row={row},"
                f" column={column}."
            )

        if type(block) is not CSR:
            block = <Data>convert.to(CSR, block)
            blocks[idx] = block
        nnz += csr.nnz(<CSR>block)

    if nnz == 0:
        return csr.zeros(shape1, shape2)

    cdef CSR out = csr.empty(shape1, shape2, nnz)
    cdef CSR op

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
                op = <CSR>blocks[i]
                if csr.nnz(op) == 0:
                    # empty CSR matrices have uninitialized row_index entries.
                    # it's unclear whether users should ever see such matrixes
                    # but we support them here anyway.
                    continue

                column = block_cols[i]
                op_row_start = op.row_index[op_row]
                op_row_end = op.row_index[op_row + 1]
                op_row_len = op_row_end - op_row_start
                for counter in range(op_row_len):
                    out.col_index[end + counter] = (
                        op.col_index[op_row_start + counter] +
                        col_pos[column]
                    )
                    out.data[end + counter] =\
                        op.data[op_row_start + counter]
                end += op_row_len
            out.row_index[row_pos[row] + op_row + 1] = end

    return out


cpdef Dense slice_dense(Dense data,
                        base.idxint row_start, base.idxint row_stop,
                        base.idxint col_start, base.idxint col_stop):
    if (
        row_start < 0 or col_start < 0
        or row_stop > data.shape[0] or col_stop > data.shape[1]
        or row_start >= row_stop or col_start >= col_stop
    ):
        raise IndexError("Slice indices are out of bounds.")
    cdef cnp.ndarray array = data.as_ndarray()
    return Dense(array[row_start:row_stop, col_start:col_stop], copy=True)


cpdef CSR slice_csr(CSR data,
                    base.idxint row_start, base.idxint row_stop,
                    base.idxint col_start, base.idxint col_stop):
    if (
        row_start < 0 or col_start < 0
        or row_stop > data.shape[0] or col_stop > data.shape[1]
        or row_start >= row_stop or col_start >= col_stop
    ):
        raise IndexError("Slice indices are out of bounds.")
    scipy = data.as_scipy()
    return CSR(scipy[row_start:row_stop, col_start:col_stop], copy=True)


cpdef Dense insert_block_dense(Data data, Dense block,
                         base.idxint above, base.idxint before):
    cdef base.idxint data_height, data_width, block_height, block_width

    data_height, data_width = data.shape
    block_height, block_width = block.shape
    if (
        above < 0 or before < 0
        or above + block_height > data_height
        or before + block_width > data_width
    ):
        raise IndexError("Cannot insert block into data: doesn't fit.")

    cdef cnp.ndarray data_array = data.to_array()  # copies
    cdef cnp.ndarray block_array = block.as_ndarray()  # doesn't copy

    data_array[above:(above+block_height),
               before:(before+block_width)] = block_array  # copies block
    return Dense(data_array, copy=False)


cdef void _memcpy_idxs(base.idxint* target, base.idxint target_start,
                       base.idxint* source, base.idxint source_start,
                       base.idxint length):
    memcpy(&target[target_start], &source[source_start],
           length * sizeof(base.idxint))


cdef void _memcpy_data(double complex* target, base.idxint target_start,
                       double complex* source, base.idxint source_start,
                       base.idxint length):
    memcpy(&target[target_start], &source[source_start],
           length * sizeof(double complex))


cpdef CSR insert_block_csr(CSR data, CSR block,
                     base.idxint above, base.idxint before):
    cdef base.idxint data_height, data_width, block_height, block_width

    data_height, data_width = data.shape
    block_height, block_width = block.shape
    if (
        above < 0 or before < 0
        or above + block_height > data_height
        or before + block_width > data_width
    ):
        raise IndexError("Cannot insert block into data: doesn't fit.")

    cdef base.idxint row, idx, idx_data, idx_block

    # the resulting nnz is the sum of the nnz of data and the nnz of block,
    # minus the number of non-zero elements in data that are overwritten
    cdef base.idxint nnz = csr.nnz(data) + csr.nnz(block)
    for row in range(above, above + block_height):
        for idx_data in range(data.row_index[row], data.row_index[row + 1]):
            if data.col_index[idx_data] < before:
                continue
            if data.col_index[idx_data] >= before + block_width:
                break
            nnz -= 1

    if nnz == 0:
        return csr.zeros(data_height, data_width)

    cdef CSR out = csr.empty(data_height, data_width, nnz)
    # copy rows above inserted 
    _memcpy_idxs(out.row_index, 0, data.row_index, 0, above + 1)
    idx = data.row_index[above]
    _memcpy_idxs(out.col_index, 0, data.col_index, 0, idx)
    _memcpy_data(out.data, 0, data.data, 0, idx)

    for row in range(above, above + block_height):
        # copy data before the inserted block
        # and increase idx_data until after the inserted block
        for idx_data in range(data.row_index[row], data.row_index[row + 1]):
            if data.col_index[idx_data] < before:
                out.col_index[idx] = data.col_index[idx_data]
                out.data[idx] = data.data[idx_data]
                idx += 1
            elif data.col_index[idx_data] >= before + block_width:
                break
        else:  # there are no more entries after the inserted block
            idx_data = data.row_index[row + 1]

        # copy the inserted block row
        for idx_block in range(
            block.row_index[row - above],
            block.row_index[row - above + 1]
        ):
            out.col_index[idx] = block.col_index[idx_block] + before
            out.data[idx] = block.data[idx_block]
            idx += 1

        # copy data after the inserted block
        nnz = data.row_index[row + 1] - idx_data
        _memcpy_idxs(out.col_index, idx, data.col_index, idx_data, nnz)
        _memcpy_data(out.data, idx, data.data, idx_data, nnz)
        idx += nnz

        out.row_index[row + 1] = idx

    # rows below inserted block
    idx_data = data.row_index[above + block_height]
    for row in range(above + block_height, data_height):
        out.row_index[row + 1] = data.row_index[row + 1] + idx - idx_data
    _memcpy_idxs(
        out.col_index, idx, data.col_index, idx_data, csr.nnz(data) - idx_data
    )
    _memcpy_data(
        out.data, idx, data.data, idx_data, csr.nnz(data) - idx_data
    )

    return out


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect


concat_blocks = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('block_rows',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('block_cols',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('blocks',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('block_widths',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('block_heights',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='concat_blocks',
    module=__name__,
    inputs=(),
    out=True,
)
concat_blocks.__doc__ =\
    """
    Concatenates data blocks into a block matrix.
    
    The coordinates of the non-empty blocks are given by the arrays
    ``block_rows`` and ``block_cols``, and the blocks themselves in the
    ``blocks`` array. All three arrays ``block_rows``, ``block_cols``, and
    ``blocks`` should have the same length.
    
    The shape of the i-th block should be
    ``(block_heights[block_rows[i]], ``block_widths[block_cols[i]])``. The
    shape of the output will be ``(sum(block_heights), sum(block_widths))``.

    Parameters
    ----------
    block_rows : int[:]
        The block row for each data block. The block row should be in
        ``range(0, len(block_heights))``.
    block_cols : int[:]
        The block column for each data block. The block column should be in
        ``range(0, len(block_widths))``.
    blocks : Data[:]
        The data blocks themselves. For performance reasons, implementations of
        ``concat_blocks`` are allowed to modify this array in place.
    block_widths : int[:]
        Array containing the block widths.
    block_heights : int[:]
        Array containing the block heights.
    """
concat_blocks.add_specialisations([
    (CSR, concat_blocks_csr),
    (Dense, concat_blocks_dense),
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
    out=True,
)
slice.__doc__ =\
    """
    Extracts a block of data from a large matrix. The output of this function
    is the slice ``[row_start:row_stop, col_start:col_stop]``. Returns a copy,
    not a view.
    """
slice.add_specialisations([
    (Dense, Dense, slice_dense),
    (CSR, CSR, slice_csr),
], _defer=True)


insert_block = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('data', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('block', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('above', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('before', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='insert_block',
    module=__name__,
    inputs=('data', 'block'),
    out=True,
)
insert_block.__doc__ =\
    """
    Inserts a block of data into a large matrix. The slice
    ``[above:(above+block_height), before:(before+block_width)]``
    of the ``data`` matrix is replaced by the ``block`` matrix, where
    ``block_height, block_width = block.shape``. The data objects ``data``
    and ``block`` are not modified, a new data object is returned.
    """
insert_block.add_specialisations([
    (Data, Dense, Dense, insert_block_dense),
    (CSR, CSR, CSR, insert_block_csr),
], _defer=True)


del _inspect, _Dispatcher