#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython

from qutip.core.data cimport csr
from qutip.core.data.csr cimport CSR
from qutip.core.data.matmul cimport matmul_csr

@cython.nonecheck(False)
@cython.cdivision(True)
cpdef CSR pow_csr(CSR matrix, unsigned long long n):
    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError("matrix power only works with square matrices")
    if n == 0:
        return csr.identity(matrix.shape[0])
    if n == 1:
        return matrix.copy()
    # We do the matrix power in terms of powers of two, so we can do it
    # ceil(lg(n)) + bits(n) - 1 matrix mulitplications, where `bits` is the
    # number of set bits in the input.
    #
    # We don't have to do matrix.copy() or pow.copy() here, because we've
    # guaranteed that we won't be returning without at least one matrix
    # multiplcation, which will allocate a new matrix.
    cdef CSR pow = matrix
    cdef CSR out = pow if n & 1 else None
    n >>= 1
    while n:
        pow = matmul_csr(pow, pow)
        if n & 1:
            out = pow if out is None else matmul_csr(out, pow)
        n >>= 1
    return out
