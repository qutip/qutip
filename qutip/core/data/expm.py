import numpy as np
import scipy.sparse.linalg

from .dense import Dense
from .csr import CSR
from .properties import isdiag_csr


def expm_csr(matrix: CSR) -> CSR:
    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError("can only exponentiate square matrix")
    if isdiag_csr(matrix):
        out = matrix.copy()
        sci = out.as_scipy()
        sci.data[:] = np.exp(sci.data)
        return out
    # The scipy solvers for the Pade approximant are more efficient with the
    # CSC format than the CSR one.
    csc = matrix.as_scipy().tocsc()
    return CSR(scipy.sparse.linalg.expm(csc).tocsr())


def expm_csr_dense(matrix: CSR) -> Dense:
    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError("can only exponentiate square matrix")
    return Dense(scipy.sparse.linalg.expm(matrix.to_array()))
