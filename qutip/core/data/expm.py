import numpy as np
import scipy.sparse.linalg
import scipy.linalg

from .dense import Dense
from .csr import CSR
from .properties import isdiag_csr
from qutip.settings import settings
from .base import idxint_dtype

__all__ = [
    'expm', 'expm_csr', 'expm_csr_dense', 'expm_dense',
    'logm', 'logm_dense',
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
    return Dense(scipy.linalg.expm(matrix.as_ndarray()), copy=False)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect


expm = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
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


def logm_dense(matrix: Dense) -> Dense:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only compute logarithm square matrix")
    return Dense(scipy.linalg.logm(matrix.as_ndarray()), copy=False)


logm = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='logm',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
logm.__doc__ = """Matrix logarithm `ln(A)` for a matrix `A`."""
logm.add_specialisations([
    (Dense, Dense, logm_dense),
], _defer=True)

del _inspect, _Dispatcher
