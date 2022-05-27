#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
from cpython cimport mem

from qutip.settings import settings

from qutip.core.data.base cimport idxint
from qutip.core.data cimport csr, dense, CSR, Dense
from qutip.core.data.adjoint cimport transpose_csr

cdef extern from *:
    # Not defined in cpython.mem for some reason, but is in pymem.h.
    void *PyMem_Calloc(size_t nelem, size_t elsize)

__all__ = [
    'isherm', 'isherm_csr',
    'isdiag', 'isdiag_csr',
    'iszero', 'iszero_csr', 'iszero_dense',
]

cdef inline bint _conj_feq(double complex a, double complex b, double tol) nogil:
    """Check whether a == conj(b) up to an absolute tolerance."""
    cdef double re = a.real - b.real
    cdef double im = a.imag + b.imag
    # Comparing the squares should be fine---there is possible precision loss
    # in the addition, but as re*re and im*im are both positive semidefinite,
    # the floating point result is strictly not greater than the true value.
    # Save the cycles: don't sqrt.
    return re*re + im*im < tol*tol

cdef inline bint _feq_zero(double complex a, double tol) nogil:
    return a.real*a.real + a.imag*a.imag < tol*tol

cdef inline double _abssq(double complex x) nogil:
    return x.real*x.real + x.imag*x.imag


cdef bint _isherm_csr_full(CSR matrix, double tol) except 2:
    """
    Full, structural test for Hermicity of a matrix.  We assume that the input
    matrix has already had its shape tested (must be square).

    This test is only necessary when there is at least one value which is less
    than the tolerance, and needed to be compared to an implicit zero.  In
    general it is less efficient than the other test, and allocates more
    memory.
    """
    cdef CSR transpose = transpose_csr(matrix)
    cdef idxint row, ptr_a, ptr_b, col_a, col_b
    for row in range(matrix.shape[0]):
        ptr_a, ptr_a_end = matrix.row_index[row], matrix.row_index[row + 1]
        ptr_b, ptr_b_end = transpose.row_index[row], transpose.row_index[row + 1]
        while ptr_a < ptr_a_end and ptr_b < ptr_b_end:
            # Doing this on every loop actually involves a few more
            # de-references than are strictly necessary, but just
            # simplifies the logic checking for the end of the row.
            col_a = matrix.col_index[ptr_a]
            col_b = transpose.col_index[ptr_b]
            if col_a == col_b:
                if not _conj_feq(matrix.data[ptr_a], transpose.data[ptr_b], tol):
                    return False
                ptr_a += 1
                ptr_b += 1
            elif col_a < col_b:
                if not _feq_zero(matrix.data[ptr_a], tol):
                    return False
                ptr_a += 1
            else:
                if not _feq_zero(transpose.data[ptr_b], tol):
                    return False
                ptr_b += 1
        for ptr_a in range(ptr_a, ptr_a_end):
            if not _feq_zero(matrix.data[ptr_a], tol):
                return False
        for ptr_b in range(ptr_b, ptr_b_end):
            if not _feq_zero(transpose.data[ptr_b], tol):
                return False
    return True

cpdef bint isherm_csr(CSR matrix, double tol=-1):
    """
    Determine whether an input CSR matrix is Hermitian up to a given
    floating-point tolerance.

    Parameters
    ----------
    matrix : CSR
        Input matrix to test
    tol : double, optional
        Absolute tolerance value to use.  Defaults to
        :obj:`settings.core['atol']`.

    Returns
    -------
    bint
        Boolean True if it is Hermitian, False if not.

    Notes
    -----
    The implementation is effectively just taking the adjoint, but rather than
    actually allocating and creating a new matrix, we just check whether the
    output would match the input matrix.  If we cannot be certain of Hermicity
    because the sizes of some elements are within tolerance of 0, we have to
    resort to a complete adjoint calculation.
    """
    tol = tol if tol >= 0 else settings.core["atol"]
    cdef size_t row, col, ptr, ptr_t, nrows=matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        return False
    cdef idxint *out_row_index = <idxint *>PyMem_Calloc(nrows + 1, sizeof(idxint))
    if out_row_index == NULL:
        raise MemoryError
    matrix.sort_indices()
    try:
        for row in range(nrows):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr] + 1
                out_row_index[col] += 1
        for row in range(nrows):
            out_row_index[row+1] += out_row_index[row]
            if out_row_index[row + 1] != matrix.row_index[row + 1]:
                # Structures are not the same, but it could still be Hermitian
                # if any value is less than the tolerance.  That is the
                # worst-case scenario, so we sacrifice its speed in favour of
                # returning faster for the more common failure cases.
                for ptr in range(matrix.row_index[nrows]):
                    if _conj_feq(matrix.data[ptr], 0, tol):
                        return _isherm_csr_full(matrix, tol)
                return False
        for row in range(nrows):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr]
                # Pointer into the "transposed" matrix.
                ptr_t = out_row_index[col]
                out_row_index[col] += 1
                if row != matrix.col_index[ptr_t]:
                    return _isherm_csr_full(matrix, tol)
                if not _conj_feq(matrix.data[ptr], matrix.data[ptr_t], tol):
                    return False
        return True
    finally:
        mem.PyMem_Free(out_row_index)


cpdef bint isdiag_csr(CSR matrix) nogil:
    cdef size_t row, ptr_start, ptr_end=matrix.row_index[0]
    for row in range(matrix.shape[0]):
        ptr_start, ptr_end = ptr_end, matrix.row_index[row + 1]
        if ptr_end - ptr_start > 1:
            return False
        if ptr_end - ptr_start == 1:
            if matrix.col_index[ptr_start] != row:
                return False
    return True


cpdef bint iszero_csr(CSR matrix, double tol=-1) nogil:
    cdef size_t ptr
    if tol < 0:
        with gil:
            tol = settings.core["atol"]
    tolsq = tol*tol
    for ptr in range(csr.nnz(matrix)):
        if _abssq(matrix.data[ptr]) > tolsq:
            return False
    return True

cpdef bint iszero_dense(Dense matrix, double tol=-1) nogil:
    cdef size_t ptr
    if tol < 0:
        with gil:
            tol = settings.core["atol"]
    tolsq = tol*tol
    for ptr in range(matrix.shape[0]*matrix.shape[1]):
        if _abssq(matrix.data[ptr]) > tolsq:
            return False
    return True


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

isherm = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('tol', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=-1),
    ]),
    name='isherm',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
isherm.__doc__ =\
    """
    Check if the matrix is Hermitian up to a optional element-wise absolute
    tolerance.  If the tolerance given is less than zero, the global settings
    value `qutip.settings.atol` will be used instead.

    Only square matrices can possibly be Hermitian.

    Arguments
    ---------
    matrix : Data
        The matrix to test for Hermicity.

    tol : real, optional
        If given, the absolute tolerance used to compare two values for
        equality.  If not given, or given and negative, the value of
        `qutip.settings.atol` is used instead.
    """
isherm.add_specialisations([
    (CSR, isherm_csr),
], _defer=True)

isdiag = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='isdiag',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
isdiag.__doc__ =\
    """
    Check if the matrix is diagonal.  The matrix need not be square to test.

    Arguments
    ---------
    matrix : Data
        The matrix to test for diagonality.
    """
isdiag.add_specialisations([
    (CSR, isdiag_csr),
], _defer=True)

iszero = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('tol', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=-1),
    ]),
    name='iszero',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
iszero.__doc__ =\
    """
    Test if this matrix is the zero matrix, up to a certain absolute tolerance.

    Arguments
    ---------
    matrix : Data
        The matrix to test.
    tol : real, optional
        The absolute tolerance to use when comparing to zero.  If not given, or
        less than 0, use the core setting `atol`.

    Returns
    -------
    bool
        Whether the matrix is equivalent to 0 under the given absolute
        tolerance.
    """
iszero.add_specialisations([
    (CSR, iszero_csr),
    (Dense, iszero_dense),
], _defer=True)

del _inspect, _Dispatcher
