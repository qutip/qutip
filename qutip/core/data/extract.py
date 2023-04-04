from . import dense, csr, Dense, CSR
from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect
try:
    from scipy.sparse import csr_array
except ImportError:
    csr_array = None
from scipy.sparse import csr_matrix


def get_dense(matrix, format=None, copy=True):
    """
    Return the scipy's object ``csr_array``.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.

    format : str, {"csr_array", "csr_matrix"}
        Type of the output.
        "csr_array" is available with scipy >= 1.8

    copy : bool, default: True
        Whether to pass a copy of the object or not.
    """
    if format not in [None, "ndarray"]:
        raise ValueError(
            "Valid format for Dense is 'ndarray'"
        )
    if copy:
        return matrix.to_array()
    else:
        return matrix.as_ndarray()


def get_csr(matrix, format=None, copy=True):
    """
    Return the scipy's object ``csr_array``.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.

    format : str, {"csr_array", "csr_matrix"}
        Type of the output.
        "csr_array" is available with scipy >= 1.8

    copy : bool, default: True
        Whether to pass a copy of the object or not.
    """
    # The requirements do not require scipy>=1.8 so csr_array may not be
    # available. But as the newest options, it is the default when available.
    # We could simplify this when support for old version is dropped.
    if format in "csr_array" and csr_array is not None:
        base = csr_array
    elif format in "csr_matrix":
        base = csr_matrix
    elif format in [None, "scipy_csr"]:
        base = csr_array or csr_matrix
    else:
        raise ValueError(
            "Valid format for CSR are 'csr_matrix' and 'csr_array'"
        )

    csr_mat = matrix.as_scipy()
    out = base(csr_mat)
    if not copy:
        # The __init__ from scipy does a copy.
        out.data = csr_mat.data
        out.indices = csr_mat.indices
        out.indptr = csr_mat.indptr

    return data


get = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter(
            'format', _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
        )
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
    matrix : Data
        The matrix to convert to common type.

    format : str, default: None
        Type of the output, "ndarray" for ``Dense``, "csr_array" for ``CSR``.
        A ValueError will be raised if the format is not supported.

    copy : bool, default: True
        Whether to pass a copy of the object.
    """
get.add_specialisations([
    (CSR, get_csr),
    (Dense, get_dense),
], _defer=True)


del _Dispatcher, _inspect
