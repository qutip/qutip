#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
from cpython cimport mem

from qutip.settings import settings

from qutip.core.data.base cimport idxint
from qutip.core.data cimport csr, dense, dia, CSR, Dense, Dia
from qutip.core.data.adjoint cimport transpose_csr
import numpy as np

cdef extern from *:
    # Not defined in cpython.mem for some reason, but is in pymem.h.
    void *PyMem_Calloc(size_t nelem, size_t elsize)

__all__ = [
    'isherm', 'isherm_csr', 'isherm_dense', 'isherm_dia',
    'isdiag', 'isdiag_csr', 'isdiag_dense', 'isdiag_dia',
    'iszero', 'iszero_csr', 'iszero_dense', 'iszero_dia',
    'isequal', 'isequal_csr', 'isequal_dense', 'isequal_dia',
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

cdef inline bint _feq(double complex a, double complex b, double atol, double rtol) nogil:
    """
    Follow numpy.allclose tolerance equation:
        |a - b| <= (atol + rtol * |b|)
    Avoid slow sqrt.
    """
    cdef double diff = (a.real - b.real)**2 + (a.imag - b.imag)**2 - atol * atol
    if diff <= 0:
        # Early exit if under atol.
        # |a - b|**2 <= atol**2
        return True
    cdef double normb_sq = b.real * b.real + b.imag * b.imag
    if normb_sq == 0. or rtol == 0.:
        # No rtol term, the previous computation was final.
        return False
    diff -= rtol * rtol * normb_sq
    if diff <= 0:
        # Early exit if under atol + rtol without cross term.
        # |a - b|**2 <= atol**2 + (rtol * |b|)**2
        return True
    # Full computation
    # (|a - b|**2 - atol**2 * (rtol * |b|)**2)**2 <= (2* atol * rtol * |b|)**2
    return diff**2 <= 4 * atol * atol * rtol * rtol * normb_sq


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


cpdef bint isherm_dia(Dia matrix, double tol=-1) nogil:
    cdef double complex val, valT
    cdef size_t diag, other_diag, col, start, end, other_start
    if tol < 0:
        with gil:
            tol = settings.core["atol"]
    if matrix.shape[0] != matrix.shape[1]:
        return False
    for diag in range(matrix.num_diag):
        if matrix.offsets[diag] == 0:
            for col in range(matrix.shape[1]):
                val = valT = matrix.data[diag * matrix.shape[1] + col]
                if not _conj_feq(val, valT, tol):
                    return False
            continue
        other_diag = 0
        while other_diag < matrix.num_diag:
            if matrix.offsets[diag] == -matrix.offsets[other_diag]:
                break
            other_diag += 1

        if other_diag < diag:
            continue

        start = max(0, matrix.offsets[diag])
        end = min(matrix.shape[1], matrix.shape[0] + matrix.offsets[diag])

        if other_diag == matrix.num_diag:
            # No matching diag, should be 0
            for col in range(start, end):
                val = matrix.data[diag * matrix.shape[1] + col]
                if not _feq_zero(val, tol):
                    return False
            continue

        other_start = max(0, matrix.offsets[other_diag])
        for col in range(end - start):
            val = matrix.data[diag * matrix.shape[1] + col + start]
            valT = matrix.data[other_diag * matrix.shape[1] + col + other_start]
            if not _conj_feq(val, valT, tol):
                return False
    return True


cpdef bint isherm_dense(Dense matrix, double tol=-1):
    """
    Determine whether an input Dense matrix is Hermitian up to a given
    floating-point tolerance.

    Parameters
    ----------
    matrix : Dense
        Input matrix to test
    tol : double, optional
        Absolute tolerance value to use.  Defaults to
        :obj:`settings.core['atol']`.

    Returns
    -------
    bint
        Boolean True if it is Hermitian, False if not.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    tol = tol if tol >= 0 else settings.core["atol"]
    cdef size_t row, col, size=matrix.shape[0]
    for row in range(size):
        for col in range(row + 1):
            if not _conj_feq(
                matrix.data[col*size+row],
                matrix.data[row*size+col],
                tol
            ):
                return False
    return True


cpdef bint isdiag_dia(Dia matrix, double tol=-1) nogil:
    cdef size_t diag, start, end, col
    if tol < 0:
        with gil:
            tol = settings.core["atol"]
    cdef double tolsq = tol*tol
    for diag in range(matrix.num_diag):
        if matrix.offsets[diag] == 0:
            continue
        start = max(0, matrix.offsets[diag])
        end = min(matrix.shape[1], matrix.shape[0] + matrix.offsets[diag])
        for col in range(start, end):
            if _abssq(matrix.data[diag * matrix.shape[1] + col]) > tolsq:
                return False
    return True


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


cpdef bint isdiag_dense(Dense matrix) nogil:
    cdef size_t row, row_stride = 1 if matrix.fortran else matrix.shape[1]
    cdef size_t col, col_stride = matrix.shape[0] if matrix.fortran else 1
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if (col != row) and matrix.data[col * col_stride + row * row_stride] != 0.:
                return False
    return True


cpdef bint iszero_dia(Dia matrix, double tol=-1) nogil:
    cdef size_t diag, start, end, col
    if tol < 0:
        with gil:
            tol = settings.core["atol"]
    cdef double tolsq = tol*tol
    for diag in range(matrix.num_diag):
        start = max(0, matrix.offsets[diag])
        end = min(matrix.shape[1], matrix.shape[0] + matrix.offsets[diag])
        for col in range(start, end):
            if _abssq(matrix.data[diag * matrix.shape[1] + col]) > tolsq:
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


cpdef bint isequal_dia(Dia A, Dia B, double atol=-1, double rtol=-1):
    if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]:
        return False
    if atol < 0:
        atol = settings.core["atol"]
    if rtol < 0:
        rtol = settings.core["rtol"]

    cdef idxint diag_a=0, diag_b=0
    cdef double complex *ptr_a
    cdef double complex *ptr_b
    cdef idxint size=A.shape[1]

    # TODO:
    # Works only for a sorted offsets list.
    # We don't have a check for whether it's already sorted, but it should be
    # in most cases. Could be improved by tracking whether it is or not.
    A = dia.clean_dia(A)
    B = dia.clean_dia(B)

    ptr_a = A.data
    ptr_b = B.data

    with nogil:
        while diag_a < A.num_diag and diag_b < B.num_diag:
            if A.offsets[diag_a] == B.offsets[diag_b]:
                for i in range(size):
                    if not _feq(ptr_a[i], ptr_b[i], atol, rtol):
                        return False
                ptr_a += size
                diag_a += 1
                ptr_b += size
                diag_b += 1
            elif A.offsets[diag_a] <= B.offsets[diag_b]:
                for i in range(size):
                    if not _feq(ptr_a[i], 0., atol, rtol):
                        return False
                ptr_a += size
                diag_a += 1
            else:
                for i in range(size):
                    if not _feq(0., ptr_b[i], atol, rtol):
                        return False
                ptr_b += size
                diag_b += 1
    return True


cpdef bint isequal_dense(Dense A, Dense B, double atol=-1, double rtol=-1):
    if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]:
        return False
    if atol < 0:
        atol = settings.core["atol"]
    if rtol < 0:
        rtol = settings.core["rtol"]
    return np.allclose(A.as_ndarray(), B.as_ndarray(), rtol, atol)


cpdef bint isequal_csr(CSR A, CSR B, double atol=-1, double rtol=-1):
    if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]:
        return False
    if atol < 0:
        atol = settings.core["atol"]
    if rtol < 0:
        rtol = settings.core["rtol"]

    cdef idxint row, ptr_a, ptr_b, ptr_a_max, ptr_b_max, col_a, col_b
    cdef idxint ncols = A.shape[1], prev_col_a, prev_col_b

    # TODO:
    # Works only for sorted indices.
    # We don't have a check for whether it's already sorted, but it should be
    # in most cases.
    A = A.sort_indices()
    B = B.sort_indices()

    with nogil:
        ptr_a_max = ptr_b_max = 0
        for row in range(A.shape[0]):
            ptr_a = ptr_a_max
            ptr_a_max = A.row_index[row + 1]
            ptr_b = ptr_b_max
            ptr_b_max = B.row_index[row + 1]
            col_a = A.col_index[ptr_a] if ptr_a < ptr_a_max else ncols + 1
            col_b = B.col_index[ptr_b] if ptr_b < ptr_b_max else ncols + 1
            prev_col_a = -1
            prev_col_b = -1
            while ptr_a < ptr_a_max or ptr_b < ptr_b_max:

                if col_a == col_b:
                    if not _feq(A.data[ptr_a], B.data[ptr_b], atol, rtol):
                        return False
                    ptr_a += 1
                    ptr_b += 1
                    col_a = A.col_index[ptr_a] if ptr_a < ptr_a_max else ncols + 1
                    col_b = B.col_index[ptr_b] if ptr_b < ptr_b_max else ncols + 1
                elif col_a < col_b:
                    if not _feq(A.data[ptr_a], 0., atol, rtol):
                        return False
                    ptr_a += 1
                    col_a = A.col_index[ptr_a] if ptr_a < ptr_a_max else ncols + 1
                else:
                    if not _feq(0., B.data[ptr_b], atol, rtol):
                        return False
                    ptr_b += 1
                    col_b = B.col_index[ptr_b] if ptr_b < ptr_b_max else ncols + 1

    return True


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

isherm = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
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

    Parameters
    ----------
    matrix : Data
        The matrix to test for Hermicity.

    tol : real, optional
        If given, the absolute tolerance used to compare two values for
        equality.  If not given, or given and negative, the value of
        `qutip.settings.atol` is used instead.
    """
isherm.add_specialisations([
    (Dense, isherm_dense),
    (Dia, isherm_dia),
    (CSR, isherm_csr),
], _defer=True)

isdiag = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='isdiag',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
isdiag.__doc__ =\
    """
    Check if the matrix is diagonal.  The matrix need not be square to test.

    Parameters
    ----------
    matrix : Data
        The matrix to test for diagonality.
    """
isdiag.add_specialisations([
    (Dense, isdiag_dense),
    (Dia, isdiag_dia),
    (CSR, isdiag_csr),
], _defer=True)

iszero = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
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

    Parameters
    ----------
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
    (Dia, iszero_dia),
    (Dense, iszero_dense),
], _defer=True)

isequal = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('A', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('B', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('atol', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=-1),
        _inspect.Parameter('rtol', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=-1),
    ]),
    name='isequal',
    module=__name__,
    inputs=('A', 'B',),
    out=False,
)
isequal.__doc__ =\
    """
    Test if two matrices are equal up to absolute and relative tolerance:

        |A - B| <= atol +  rtol * |b|

    Similar to ``numpy.allclose``.

    Parameters
    ----------
    A, B : Data
        Matrices to compare.
    atol : real, optional
        The absolute tolerance to use. If not given, or
        less than 0, use the core setting `atol`.
    rtol : real, optional
        The relative tolerance to use. If not given, or
        less than 0, use the core setting `atol`.

    Returns
    -------
    bool
        Whether the matrix are equal.
    """
isequal.add_specialisations([
    (CSR, CSR, isequal_csr),
    (Dia, Dia, isequal_dia),
    (Dense, Dense, isequal_dense),
], _defer=True)

del _inspect, _Dispatcher
