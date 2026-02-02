from . import Dense, CSR, Dia
from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect
try:
    from scipy.sparse import csr_array
except ImportError:
    csr_array = None
from scipy.sparse import csr_matrix


__all__ = ["extract"]


def extract_dense(matrix, format=None, copy=True):
    """
    Return an array representation of the Dense data object.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to the given format.

    format : str {"ndarray"}, default="ndarray"
        Type of the output.

    copy : bool, default: True
        Whether to return a copy of the data. If False,
        a view of the data is returned when possible.
    """
    if format not in [None, "ndarray"]:
        raise ValueError(
            "Dense can only be extracted to 'ndarray'"
        )
    if copy:
        return matrix.to_array()
    else:
        return matrix.as_ndarray()


def extract_csr(matrix, format=None, copy=True):
    """
    Return the scipy's object ``csr_matrix``.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.

    format : str, {"csr_matrix"}
        Type of the output.

    copy : bool, default: True
        Whether to pass a copy of the object or not.
    """
    if format not in [None, "scipy_csr", "csr_matrix"]:
        raise ValueError(
            "CSR can only be extracted to 'csr_matrix'"
        )
    csr_mat = matrix.as_scipy()
    if copy:
        csr_mat = csr_mat.copy()
    return csr_mat


def extract_dia(matrix, format=None, copy=True):
    """
    Return the scipy's object ``dia_matrix``.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.

    format : str, {"dia_matrix"}
        Type of the output.

    copy : bool, default: True
        Whether to pass a copy of the object or not.
    """
    if format not in [None, "scipy_dia", "dia_matrix"]:
        raise ValueError(
            "Dia can only be extracted to 'dia_matrix'"
        )
    dia_mat = matrix.as_scipy()
    if copy:
        dia_mat = dia_mat.copy()
    return dia_mat


extract = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter(
            'format', _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
        ),
        _inspect.Parameter(
            'copy', _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=True
        )
    ]),
    name='extract',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
extract.__doc__ =\
    """
    Return the common representation of the data layer object: scipy's
    ``csr_matrix`` for ``CSR``, numpy array for ``Dense``, Jax's ``Array`` for
    ``JaxArray``, etc.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.

    format : str, default: None
        Type of the output, "ndarray" for ``Dense``, "csr_array" for ``CSR``.
        A ValueError will be raised if the format is not supported.

    copy : bool, default: True
        Whether to pass a copy of the object.
    """
extract.add_specialisations([
    (CSR, extract_csr),
    (Dia, extract_dia),
    (Dense, extract_dense),
], _defer=True)


del _Dispatcher, _inspect
