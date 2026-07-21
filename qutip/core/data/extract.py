from . import Dense, CSR, Dia
from .dispatch import Dispatcher as _Dispatcher
from ._scipy_sparse import (
    csr_as_matrix, dia_as_matrix, _warn_if_legacy_scipy_input,
)
import inspect as _inspect


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
    Return the scipy CSR representation of the data object.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.

    format : str or None, default: None
        The scipy container to return.  ``"csr_array"`` returns a ``csr_array``;
        ``"csr_matrix"`` returns the legacy ``csr_matrix``.  ``None`` (or
        ``"scipy_csr"``) returns the native backing, which is always a
        ``csr_array``.

    copy : bool, default: True
        Whether to pass a copy of the object or not.
    """
    if format not in [None, "scipy_csr", "csr_matrix", "csr_array"]:
        raise ValueError(
            "CSR can only be extracted to 'csr_array' or 'csr_matrix'"
        )
    scipy_csr = matrix.as_scipy()
    # TODO: remove upon SciPy deprecation of csr_matrix
    if format == "csr_matrix":
        #_warn_if_legacy_scipy_input("csr")
        scipy_csr = csr_as_matrix(scipy_csr)
    if copy:
        scipy_csr = scipy_csr.copy()
    return scipy_csr


def extract_dia(matrix, format=None, copy=True):
    """
    Return the scipy DIA representation of the data object.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.

    format : str or None, default: None
        The scipy container to return.  ``"dia_array"`` returns a ``dia_array``;
        ``"dia_matrix"`` returns the legacy ``dia_matrix``.  ``None`` (or
        ``"scipy_dia"``) returns the native backing, which is always a
        ``dia_array``.

    copy : bool, default: True
        Whether to pass a copy of the object or not.
    """
    if format not in [None, "scipy_dia", "dia_matrix", "dia_array"]:
        raise ValueError(
            "Dia can only be extracted to 'dia_array' or 'dia_matrix'"
        )
    scipy_dia = matrix.as_scipy()
    # TODO: remove upon SciPy deprecation of dia_matrix
    if format == "dia_matrix":
        #_warn_if_legacy_scipy_input("dia")
        scipy_dia = dia_as_matrix(scipy_dia)
    if copy:
        scipy_dia = scipy_dia.copy()
    return scipy_dia


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

extract.__doc__ = \
    """
    Return the common representation of the data layer object: scipy's
    ``csr_array`` or ``csr_matrix`` (soon-to-be-deprecated; see
    https://docs.scipy.org/doc/scipy/dev/roadmap-detailed.html#sparse) for
    ``CSR``, numpy array for ``Dense``, Jax's ``Array`` for ``JaxArray``,
    etc.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.
    format : str, default: None
        Type of the output: "ndarray" for ``Dense``; "csr_array"/
        "csr_matrix" for ``CSR``; "dia_array"/"dia_matrix" for ``Dia``.
        When ``None``, sparse types return their native ``sparray``
        backing. A ValueError is raised if the format is not supported.
    copy : bool, default: True
        Whether to pass a copy of the object.
    """
extract.add_specialisations([
    (CSR, extract_csr),
    (Dia, extract_dia),
    (Dense, extract_dense),
], _defer=True)


del _Dispatcher, _inspect
