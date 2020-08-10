#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
from cpython cimport mem

import qutip.settings

from qutip.core.data.base cimport idxint
from qutip.core.data.csr cimport CSR
from qutip.core.data cimport csr

cdef extern from *:
    # Not defined in cpython.mem for some reason, but is in pymem.h.
    void *PyMem_Calloc(size_t nelem, size_t elsize)


cdef inline bint _conj_feq(double complex a, double complex b, double tol) nogil:
    """Check whether a == conj(b) up to an absolute tolerance."""
    cdef double re = a.real - b.real
    cdef double im = a.imag + b.imag
    # Comparing the squares should be fine---there is possible precision loss
    # in the addition, but as re*re and im*im are both positive semidefinite,
    # the floating point result is strictly not greater than the true value.
    # Save the cycles: don't sqrt.
    return re*re + im*im < tol*tol


cpdef bint isherm_csr(CSR matrix, double tol=qutip.settings.atol):
    """
    Determine whether an input CSR matrix is Hermitian up to a given
    floating-point tolerance.

    Parameters
    ----------
    matrix : CSR
        Input matrix to test
    tol : double, optional
        Absolute tolerance value to use.  Defaults to
        :obj:`qutip.settings.atol`.

    Returns
    -------
    bint
        Boolean True if it is Hermitian, False if not.

    Notes
    -----
    The implementation is effectively just taking the adjoint, but rather than
    actually allocating and creating a new matrix, we just check whether the
    output would match the input matrix.
    """
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

        for row in range(nrows):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr]
                # Pointer into the "transposed" matrix.
                ptr_t = out_row_index[col]
                out_row_index[col] += 1
                if not (
                    matrix.col_index[ptr_t] == row
                    and _conj_feq(matrix.data[ptr], matrix.data[ptr_t], tol)
                ):
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
