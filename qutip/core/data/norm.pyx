#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.stdlib cimport calloc, free
from libc cimport math

from scipy.linalg cimport cython_blas as blas

from qutip.core.data cimport csr, dense, CSR, Dense

from qutip.core.data.adjoint cimport adjoint_csr, adjoint_dense
from qutip.core.data.matmul cimport matmul_csr

from qutip.core.data import eigs_csr

cdef extern from "<complex>" namespace "std" nogil:
    # abs is templated such that Cython treats std::abs as complex->complex
    double abs(double complex x)

cdef double abssq(double complex x) nogil:
    return x.real*x.real + x.imag*x.imag

# We always use BLAS routines where possible because architecture-specific
# libraries will typically apply vectorised operations for us.

# This module is meant to be accessed by dot-access (e.g. norm.one_csr).
__all__ = []

cpdef double one_csr(CSR matrix) nogil except -1:
    cdef int n=matrix.shape[1], inc=1
    cdef size_t ptr
    cdef double *col = <double *>calloc(matrix.shape[1], sizeof(double))
    try:
        for ptr in range(csr.nnz(matrix)):
            col[matrix.col_index[ptr]] += abs(matrix.data[ptr])
        # BLAS is a Fortran library, so it's one-indexed of course...
        return col[blas.idamax(&n, col, &inc) - 1]
    finally:
        free(col)

cpdef double trace_csr(CSR matrix, sparse=False, tol=0, maxiter=None) except -1:
    # We use the general eigenvalue solver which involves a Python call, so
    # there's no point attempting to release the GIL.
    cdef CSR op = matmul_csr(matrix, adjoint_csr(matrix))
    cdef size_t i
    cdef double [::1] eigs = eigs_csr(op, isherm=True, vecs=False,
                                      sparse=sparse, tol=tol, maxiter=maxiter)
    cdef double total = 0
    for i in range(matrix.shape[0]):
        # The abs is technically not part of the definition, but since all
        # eigenvalues _should_ be > 0 (as X @ X.adjoint() is Hermitian), any
        # which are lower will just be ~1e-15 due to numerical approximations.
        total += math.sqrt(abs(eigs[i]))
    return total

cpdef double max_csr(CSR matrix) nogil:
    cdef size_t ptr
    cdef double total=0, cur
    for ptr in range(csr.nnz(matrix)):
        # The positive square root is monotonic over positive reals, so we can
        # find the maximum value by considering the abs squared (which doesn't
        # require a sqrt) rather than the abs(which does), and then perform
        # only a single sqrt at the end.
        cur = abssq(matrix.data[ptr])
        total = cur if cur > total else total
    return math.sqrt(total)

cpdef double frobenius_csr(CSR matrix) nogil:
    # The Frobenius norm is effectively the same as the L2 norm when
    # considering the non-zero elements as a vector.
    cdef int n=csr.nnz(matrix), inc=1
    return blas.dznrm2(&n, &matrix.data[0], &inc)

cpdef double l2_csr(CSR matrix) nogil except -1:
    if matrix.shape[0] != 1 and matrix.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_csr(matrix)

cpdef double frobenius_dense(Dense matrix) nogil:
    cdef int n = matrix.shape[0] * matrix.shape[1]
    cdef int inc = 1
    return blas.dznrm2(&n, matrix.data, &inc)


cpdef double l2_dense(Dense matrix) nogil except -1:
    if matrix.shape[0] != 1 and matrix.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_dense(matrix)
