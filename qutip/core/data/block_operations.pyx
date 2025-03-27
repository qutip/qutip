#cython: language_level=3

from . import base, convert, csr
from . cimport base, CSR, Data, Dense

import numpy as np
cimport numpy as cnp

is_numpy1 = np.lib.NumpyVersion(np.__version__) < '2.0.0b1'


__all__ = [
    'concat_dense', 'slice_dense', 'insert_dense',
    'concat_csr', 'slice_csr', 'insert_csr',
    'concat', 'slice', 'insert',
]


cpdef Dense concat_dense(
    Data[:,:] data_array,
    base.idxint[:] block_widths = None,
    base.idxint[:] block_heights = None
):
    cdef base.idxint row, column
    cdef Data op

    if data_array.shape[0] == 0 or data_array.shape[1] == 0:
        raise ValueError("Cannot concatenate empty data array.")

    if block_widths is None or block_heights is None:
        block_widths, block_heights = _check_widths_heights(data_array)

    blocks = [[None] * len(block_widths) for _ in range(len(block_heights))]
    for row in range(len(block_heights)):
        for column in range(len(block_widths)):
            op = data_array[row][column]
            if op is None:
                blocks[row][column] = np.zeros(
                    (block_heights[row], block_widths[column])
                )
            else:
                blocks[row][column] = op.to_array()

    return Dense(np.block(blocks), copy=(False if is_numpy1 else None))


cpdef CSR concat_csr(
    Data[:,:] data_array,
    base.idxint[:] block_widths = None,
    base.idxint[:] block_heights = None
):
    if data_array.shape[0] == 0 or data_array.shape[1] == 0:
        raise ValueError("Cannot concatenate empty data array.")

    if block_widths is None or block_heights is None:
        block_widths, block_heights = _check_widths_heights(data_array)

    cdef cnp.ndarray[base.idxint] row_pos = np.zeros(
        len(block_heights) + 1, dtype=base.idxint_dtype)
    row_pos[1:] = np.cumsum(block_heights, dtype=base.idxint_dtype)
    cdef cnp.ndarray[base.idxint] col_pos = np.zeros(
        len(block_heights) + 1, dtype=base.idxint_dtype)
    col_pos[1:] = np.cumsum(block_widths, dtype=base.idxint_dtype)

    cdef base.idxint row, columm
    cdef base.idxint num_rows = len(block_heights)
    cdef base.idxint num_cols = len(block_widths)
    cdef base.idxint shape1 = row_pos[num_rows]
    cdef base.idxint shape2 = col_pos[num_cols]
    cdef base.idxint nnz_ = 0
    cdef Data op

    for row in range(num_rows):
        for column in range(num_cols):
            op = data_array[row][column]
            if op is None:
                continue
            if type(op) is not CSR:
                op = convert.to(CSR, op)
                data_array[row][column] = op
            nnz_ += csr.nnz(op)

    if nnz_ == 0:
        return csr.zeros(shape1, shape2)

    cdef CSR out = csr.empty(shape1, shape2, nnz_)
    cdef base.idxint op_row, op_row_start, op_row_end, op_row_len, i
    cdef base.idxint end = 0
    cdef CSR csr_op

    out.row_index[0] = 0

    for row in range(num_rows):
        for op_row in range(block_heights[row]):
            for column in range(num_cols):
                csr_op = data_array[row][column]
                if csr_op is None or csr.nnz(csr_op) == 0:
                    continue
                op_row_start = csr_op.row_index[op_row]
                op_row_end = csr_op.row_index[op_row + 1]
                op_row_len = op_row_end - op_row_start
                for i in range(op_row_len):
                    out.col_index[end + i] =\
                        csr_op.col_index[op_row_start + i] + col_pos[column]
                    out.data[end + i] = csr_op.data[op_row_start + i]
                end += op_row_len
            out.row_index[row_pos[row] + op_row + 1] = end

    return out


cdef _check_widths_heights(Data[:,:] data_array):
    cdef base.idxint num_rows = data_array.shape[0]
    cdef base.idxint num_columns = data_array.shape[1]
    cdef base.idxint row, column

    # determine block widths
    cdef base.idxint[:] block_widths =\
        np.full(num_columns, -1, dtype=base.idxint_dtype)
    for column in range(num_columns):
        for row in range(num_rows):
            if data_array[row][column] is None:
                continue
            if block_widths[column] == -1:
                block_widths[column] = data_array[row][column].shape[1]
            elif block_widths[column] != data_array[row][column].shape[1]:
                raise ValueError(
                    "Cannot concatenate data array: inconsistent block"
                    f" widths in column {column + 1}.")

    # determine block heights
    cdef base.idxint[:] block_heights =\
        np.full(num_rows, -1, dtype=base.idxint_dtype)
    for row in range(num_rows):
        for column in range(num_columns):
            if data_array[row][column] is None:
                continue
            if block_heights[row] == -1:
                block_heights[row] = data_array[row][column].shape[0]
            elif block_heights[row] != data_array[row][column].shape[0]:
                raise ValueError(
                    "Cannot concatenate data array: inconsistent block"
                    f" heights in row {row + 1}.")

    return block_widths, block_heights


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

    data_scipy[above:(above+block_height),
               before:(before+block_width)] = block_scipy
    return CSR(data_scipy, copy=False)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

concat = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('data_array', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('block_widths',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=None),
        _inspect.Parameter('block_heights',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=None),
    ]),
    name='concat',
    module=__name__,
    inputs=(),
    out=True,
)
concat.__doc__ =\
    """
    Concatenates a 2D array of data blocks into a single large matrix.
    """
concat.add_specialisations([
    (Dense, concat_dense),
    (CSR, concat_csr),
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
