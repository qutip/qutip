from . import dense, csr, Dense, CSR
from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect


def get_dense(matrix, copy=True):
    if copy:
        return matrix.to_array()
    else:
        return matrix.as_ndarray()


def get_csr(matrix, copy=True):
    data = matrix.as_scipy()
    if copy:
        data = data.copy()
    return data


get = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter(
            'copy', _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=True
        )
    ]),
    name='get',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
get.__doc__ =\
    """
    Return the common representation of the data layer object: scipy's
    ``csr_matrix`` for ``CSR``, numpy array for ``Dense``, Jax's ``Array`` for
    ``JaxArray``, etc.

    Parameters
    ----------
    matrix : int
        The matrix to convert to common type.

    copy : bool, default: True
        Whether to pass a copy of the object.
    """
get.add_specialisations([
    (CSR, get_csr),
    (Dense, get_dense),
], _defer=True)


del _Dispatcher, _inspect
