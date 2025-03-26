#cython: language_level=3

from .base cimport Data
from .dense cimport Dense

from .base import Data
from .dense import Dense

import numpy as np

is_numpy1 = np.lib.NumpyVersion(np.__version__) < '2.0.0b1'


__all__ = [
    'concat_dense', 'slice_dense', 'insert_dense',
    'concat', 'slice', 'insert',
]


cpdef Dense concat_dense(
    list[list[Data]] data_array,
    list[object] block_widths = None,
    list[object] block_heights = None
):
    cdef int columm, row

    if len(data_array) == 0 or len(data_array[0]) == 0:
        raise ValueError("Cannot concatenate empty data array.")

    if block_widths is None or block_heights is None:
        block_widths, block_heights = _check_widths_heights(data_array)

    for row in range(len(block_heights)):
        for column in range(len(block_widths)):
            if data_array[row][column] is None:
                data_array[row][column] = np.zeros(
                    (block_heights[row], block_widths[column])
                )
            else:
                data_array[row][column] = data_array[row][column].to_array()

    return Dense(np.block(data_array),
                 copy=(False if is_numpy1 else None))


cdef _check_widths_heights(list[list[Data]] data_array):
    cdef int num_columns, num_rows
    cdef int columm, row

    num_columns = len(data_array[0])
    num_rows = len(data_array)

    # determine block widths
    block_widths = [-1] * num_columns
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
    block_heights = [-1] * num_rows
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


cpdef Dense slice_dense(Dense data, row_start, row_stop, col_start, col_stop):
    array = data.as_ndarray()
    return Dense(array[row_start:row_stop, col_start:col_stop], copy=True)


cpdef Dense insert_dense(Data data, Dense block, above, before):
    cdef int data_height, data_width, block_height, block_width

    data_array = data.to_array()  # copies
    block_array = block.as_ndarray()  # doesn't copy

    data_height, data_width = data.shape
    block_height, block_width = block.shape
    if above + block_height > data_height or before + block_width > data_width:
        raise IndexError("Cannot insert block into data: doesn't fit.")

    data_array[above:(above+block_height),
               before:(before+block_width)] = block_array  # copies block
    return Dense(data_array, copy=False)


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
    (Dense, concat_dense)
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
], _defer=True)


del _inspect, _Dispatcher
