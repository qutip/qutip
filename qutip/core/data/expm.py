import numpy as np
import scipy.sparse.linalg

from .dense import Dense
from .csr import CSR
from .properties import isdiag_csr

__all__ = [
    'expm', 'expm_csr', 'expm_csr_dense',
]


def expm_csr(matrix: CSR) -> CSR:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only exponentiate square matrix")
    if isdiag_csr(matrix):
        sci = matrix.as_scipy()
        # Note that expm1 is an elemetwise operation that maps 0->0 which
        # involves no operation at all. Since we know that the input is
        # diagonal, elementwise expm1 in the whole matrix has the same
        # computational cost as doing only a expm1 in the diagonal
        # (elementwise).
        sci = sci.expm1() + scipy.sparse.diags(np.ones(matrix.shape[0]))
        return CSR(sci)
    # The scipy solvers for the Pade approximant are more efficient with the
    # CSC format than the CSR one.
    csc = matrix.as_scipy().tocsc()
    return CSR(scipy.sparse.linalg.expm(csc).tocsr())


def expm_csr_dense(matrix: CSR) -> Dense:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only exponentiate square matrix")
    return Dense(scipy.sparse.linalg.expm(matrix.to_array()))


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
], _defer=True)

del _inspect, _Dispatcher
