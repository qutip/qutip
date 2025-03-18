from .base import Data
from .csr import CSR
from .dense import Dense

import numpy


__all__ = [
    'concat_data',
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
    TODO: docstring
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


# TODO other dtypes
def concat_csr(data_array: list[list[CSR]]) -> CSR:
    ...
