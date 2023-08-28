#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc cimport math

from cpython cimport mem

from scipy.linalg cimport cython_blas as blas
import scipy
import numpy as np

from qutip.core.data cimport CSR, Dense, csr, Data, Dia

from qutip.core.data.adjoint cimport adjoint_csr, adjoint_dense
from qutip.core.data.matmul cimport matmul_csr

from qutip.core.data import eigs_csr, eigs_dense

cdef extern from *:
    # Not included in Cython for some reason?
    void *PyMem_Calloc(size_t n, size_t elsize)

cdef extern from "<complex>" namespace "std" nogil:
    # abs is templated such that Cython treats std::abs as complex->complex
    double abs(double complex x)

cdef double abssq(double complex x) nogil:
    return x.real*x.real + x.imag*x.imag

# We always use BLAS routines where possible because architecture-specific
# libraries will typically apply vectorised operations for us.

# This module is meant to be accessed by dot-access (e.g. norm.one_csr).
__all__ = []

cpdef double one_csr(CSR matrix) except -1:
    cdef int n=matrix.shape[1], inc=1
    cdef size_t ptr
    cdef double *col = <double *> PyMem_Calloc(matrix.shape[1], sizeof(double))
    try:
        for ptr in range(csr.nnz(matrix)):
            col[matrix.col_index[ptr]] += abs(matrix.data[ptr])
        # BLAS is a Fortran library, so it's one-indexed of course...
        return col[blas.idamax(&n, col, &inc) - 1]
    finally:
        mem.PyMem_Free(col)


cpdef double trace_dense(Dense matrix) except -1:
    """Compute the trace norm relaying scipy for dense operations."""
    return scipy.linalg.norm(matrix.as_ndarray(), 'nuc')


cpdef double trace_csr(CSR matrix, tol=0, maxiter=None) except -1:
    """Compute the trace norm using only sparse operations. These consist
    of determining the eigenvalues of `matrix @ matrix.adjoint()` and summing
    their square roots."""
    # For column and row vectors we simply use the l2 norm as it is equivalent
    # to the trace norm.
    if matrix.shape[0]==1 or matrix.shape[1]==1:
        return l2_csr(matrix)

    cdef CSR op = matmul_csr(matrix, adjoint_csr(matrix))
    cdef size_t i
    cdef double [::1] eigs

    eigs = eigs_csr(op, isherm=True, vecs=False, tol=tol, maxiter=maxiter)

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

cpdef double l2_csr(CSR matrix) except -1 nogil:
    if matrix.shape[0] != 1 and matrix.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_csr(matrix)

cpdef double one_dense(Dense matrix) nogil:
    cdef size_t ptr, col, row, col_stride, row_stride
    cdef double out=0, cur
    col_stride = matrix.shape[0] if matrix.fortran else 1
    row_stride = 1 if matrix.fortran else matrix.shape[1]
    for col in range(matrix.shape[1]):
        ptr = col * col_stride
        cur = 0
        for row in range(matrix.shape[0]):
            cur += abs(matrix.data[ptr])
            ptr += row_stride
        out = cur if cur > out else out
    return out

cpdef double max_dense(Dense matrix) nogil:
    cdef size_t ptr
    cdef double total=0, cur
    for ptr in range(matrix.shape[0] * matrix.shape[1]):
        # The positive square root is monotonic over positive reals, so we can
        # find the maximum value by considering the abs squared (which doesn't
        # require a sqrt) rather than the abs(which does), and then perform
        # only a single sqrt at the end.
        cur = abssq(matrix.data[ptr])
        total = cur if cur > total else total
    return math.sqrt(total)

cpdef double frobenius_dense(Dense matrix) nogil:
    cdef int n = matrix.shape[0] * matrix.shape[1]
    cdef int inc = 1
    return blas.dznrm2(&n, matrix.data, &inc)

cpdef double l2_dense(Dense matrix) except -1 nogil:
    if matrix.shape[0] != 1 and matrix.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_dense(matrix)

cpdef double frobenius_dia(Dia matrix) nogil:
    cdef int offset, diag, start, end, col=1
    cdef double total=0, cur
    for diag in range(matrix.num_diag):
        offset = matrix.offsets[diag]
        start = int_max(0, offset)
        end = min(matrix.shape[1], matrix.shape[0] + offset)
        for col in range(start, end):
            total += abssq(matrix.data[diag * matrix.shape[1] + col])
    return math.sqrt(total)

cpdef double l2_dia(Dia matrix) except -1 nogil:
    if matrix.shape[0] != 1 and matrix.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_dia(matrix)

cpdef double max_dia(Dia matrix) nogil:
    cdef int offset, diag, start, end, col=1
    cdef double total=0, cur
    for diag in range(matrix.num_diag):
        offset = matrix.offsets[diag]
        start = int_max(0, offset)
        end = min(matrix.shape[1], matrix.shape[0] + offset)
        for col in range(start, end):
            cur = abssq(matrix.data[diag * matrix.shape[1] + col])
            total = cur if cur > total else total
    return math.sqrt(total)

cpdef double one_dia(Dia matrix) except -1:
    cdef int offset, diag, start, end, col=1
    cols_one = np.zeros(matrix.shape[1], dtype=float)
    for diag in range(matrix.num_diag):
        offset = matrix.offsets[diag]
        start = int_max(0, offset)
        end = min(matrix.shape[1], matrix.shape[0] + offset)
        for col in range(start, end):
            cols_one[col] += abs(matrix.data[diag * matrix.shape[1] + col])
    return np.max(cols_one)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

l2 = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('vector', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='l2',
    module=__name__,
    inputs=('vector',),
)
l2.__doc__ =\
    """
    Compute the L2 (Euclidean) norm of a bra or ket vector.  This is equal to
        sqrt(|v[0]|**2 + |v[1]|**2 + ...)
    This is only defined for vectors, but see `norm.frobenius` for the similar
    norm defined on all matrices.
    """
l2.add_specialisations([
    (Dense, l2_dense),
    (Dia, l2_dia),
    (CSR, l2_csr),
], _defer=True)

_norm_signature = _inspect.Signature([
    _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
])

frobenius = _Dispatcher(_norm_signature, name='frobenius', module=__name__, inputs=('matrix',))
frobenius.__doc__ =\
    """
    Compute the Frobenius (Hilbert-Schmidt) norm of a matrix.  This is defined
    as the
        sqrt(sum_i sum_j |matrix[i, j]|**2)
    and is similar to an extension of the vector L2 norm to all matrices.
    """
frobenius.add_specialisations([
    (Dense, frobenius_dense),
    (Dia, frobenius_dia),
    (CSR, frobenius_csr),
], _defer=True)

max = _Dispatcher(_norm_signature, name='max', module=__name__, inputs=('matrix',))
max.__doc__ =\
    """
    Compute the max norm of a matrix.  This is the largest absolute value of an
    entry in the matrix, or mathematically
        max_{i,j} |matrix[i, j]|
    """
max.add_specialisations([
    (Dense, max_dense),
    (Dia, max_dia),
    (CSR, max_csr),
], _defer=True)

one = _Dispatcher(_norm_signature, name='one', module=__name__, inputs=('matrix',))
one.__doc__ =\
    """
    Compute the one-norm (L1--L1) norm of a matrix.  This is the value of the
    largest L1 norm of a column in the matrix, where the L1 norm of a vector is
    the sum of the absolute values.
    """
one.add_specialisations([
    (Dense, one_dense),
    (Dia, one_dia),
    (CSR, one_csr),
], _defer=True)


trace = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    inputs=('matrix',),
    name='trace',
    module=__name__,
    out=False,
)
trace.__doc__ =\
    """
    Compute the trace-norm of a matrix.  This is the sum of the singular values
    of the matrix, or equivalently
        Tr(sqrt(A @ A.adjoint()))
    """
trace.add_specialisations([
    (CSR, trace_csr),
    (Dense, trace_dense),
], _defer=True)


cpdef double frobenius_data(Data state) except -1:
    if type(state) is Dense:
        return frobenius_dense(state)
    elif type(state) is CSR:
        return frobenius_csr(state)
    else:
        return frobenius(state)
