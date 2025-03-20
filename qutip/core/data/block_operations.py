from .base import Data
from .dense import Dense
from .dispatch import Dispatcher as _Dispatcher

import inspect as _inspect

import numpy


__all__ = [
    'concat_data', 'slice_dense', 'slice', 'insert_dense', 'insert'
]


def _invalid_shapes():
    raise ValueError(
        "The given matrices have incompatible shapes and cannot be"
        " concatenated."
    )

def _to_array(data: Data | numpy.ndarray) -> numpy.ndarray:
    if isinstance(data, Data):
        return data.to_array()
    return data

def concat_data(
    data_array: list[list[Data | numpy.ndarray]],
    _skip_checks: bool = False
) -> Dense:
    """
    Concatenates blocks of data into a block matrix
    """

    if not _skip_checks:
        if len(data_array) == 0 or len(data_array[0]) == 0:
            _invalid_shapes()

        column_widths = [data.shape[1] for data in data_array[0]]
        for row in data_array:
            if len(row) != len(column_widths):
                _invalid_shapes()

            row_height = row[0].shape[0]
            for data, column_width in zip(row, column_widths):
                if data.shape != (row_height, column_width):
                    _invalid_shapes()

    nd_arrays = [[_to_array(data) for data in row] for row in data_array]
    return Dense(numpy.block(nd_arrays), copy=None)
    # copy=None means: only copy if numpy cannot avoid copying


# TODO other dtypes, dispatch function


def slice_dense(data: Data,
                row_start: int, row_stop: int,
                col_start: int, col_stop: int) -> Dense:
    array = data.to_array()
    return Dense(array[row_start:row_stop, col_start:col_stop], copy=True)

slice = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('data', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('row_start', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('row_stop', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('col_start', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('col_stop', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='slice',
    module=__name__,
    inputs=('data',),
    out=True,
)
slice.__doc__ =\
    """
    Extracts a block of data from a large matrix.
    """
slice.add_specialisations([
    (Data, Dense, slice_dense),
], _defer=True)


def insert_dense(data: Data, block: Data,
                 above: int, before: int) -> Dense:
    data_array = data.to_array()
    data_height, data_width = data.shape
    block_array = block.to_array()
    block_height, block_width = block.shape
    if above + block_height > data_height or before + block_width > data_width:
        raise IndexError("Cannot insert block into data: doesn't fit.")
    data_array[above:(above+block_height), before:(before+block_width)] =\
        block_array
    return Dense(data_array, copy=False)

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
insert.__doc__ = """TODO"""
insert.add_specialisations([
    (Data, Data, Dense, insert_dense),
], _defer=True)