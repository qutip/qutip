import numpy as np
import scipy.sparse.linalg
import scipy.linalg

from .dense import Dense
from .csr import CSR
from . import csr
from .make import diag
from .add import add_dense, add_csr
from .properties import isdiag_csr, iszero_csr, iszero_dense
from qutip.settings import settings
from .base import idxint_dtype

__all__ = [
    'expm', 'expm_csr', 'expm_csr_dense', 'expm_dense',
    'logm', 'logm_csr', 'logm_dense',
]


def expm_csr(matrix: CSR) -> CSR:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only exponentiate square matrix")
    if isdiag_csr(matrix):
        matrix_sci = matrix.as_scipy()
        data = np.ones(matrix.shape[0], dtype=np.complex128)
        data[matrix_sci.indices] += np.expm1(matrix_sci.data)
        return CSR(
            (
                data,
                np.arange(matrix.shape[0], dtype=idxint_dtype),
                np.arange(matrix.shape[0] + 1, dtype=idxint_dtype),
            ),
            shape=matrix.shape,
            copy=False,
        )
    # The scipy solvers for the Pade approximant are more efficient with the
    # CSC format than the CSR one.
    csc = matrix.as_scipy().tocsc()
    return CSR(scipy.sparse.linalg.expm(csc).tocsr(),
               tidyup=settings.core['auto_tidyup'])


def expm_csr_dense(matrix: CSR) -> Dense:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only exponentiate square matrix")
    return Dense(scipy.sparse.linalg.expm(matrix.to_array()))


def expm_dense(matrix: Dense) -> Dense:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only exponentiate square matrix")
    return Dense(scipy.linalg.expm(matrix.as_ndarray()))


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect


expm = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='expm',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
expm.__doc__ = """Matrix exponential `e**A` for a matrix `A`."""
expm.add_specialisations([
    (CSR, CSR, expm_csr),
    (CSR, Dense, expm_csr_dense),
    (Dense, Dense, expm_dense),
], _defer=True)


def logm_csr(matrix: CSR) -> CSR:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only compute logarithm of square matrix")
    if iszero_csr(matrix):
        data = np.ones(matrix.shape[0]) * 1e-300
        shift = diag(data, 0, dtype=CSR)
        matrix = add_csr(matrix, shift)
    if isdiag_csr(matrix):
        matrix_sci = matrix.as_scipy()
        data = np.zeros(matrix.shape[0], dtype=np.complex128)
        data[matrix_sci.indices] += np.log(matrix_sci.data)
        return CSR(
            (
                data,
                np.arange(matrix.shape[0], dtype=idxint_dtype),
                np.arange(matrix.shape[0] + 1, dtype=idxint_dtype),
            ),
            shape=matrix.shape,
            copy=False,
        )
    else:
        return csr.from_dense(Dense(scipy.linalg.logm(matrix.to_array())))


def logm_dense(matrix: Dense) -> Dense:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only compute logarithm square matrix")
    if iszero_dense(matrix):
        data = np.ones(matrix.shape[0]) * 1e-300
        shift = diag(data, 0, dtype=Dense)
        matrix = add_dense(matrix, shift)
    return Dense(scipy.linalg.logm(matrix.as_ndarray()))


logm = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='logm',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
logm.__doc__ = """Matrix logarithm `ln(A)` for a matrix `A`."""
logm.add_specialisations([
    (CSR, CSR, logm_csr),
    (Dense, Dense, logm_dense),
], _defer=True)

del _inspect, _Dispatcher
