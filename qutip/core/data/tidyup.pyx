#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.math cimport fabs

cimport numpy as cnp

from qutip.core.data cimport csr
from qutip.core.data.csr cimport CSR

cdef extern from "<complex>" namespace "std" nogil:
    # abs is templated such that Cython treats std::abs as complex->complex
    double abs(double complex x)

cpdef CSR tidyup_csr(CSR matrix, double tol, bint inplace=True):
    cdef bint re, im
    cdef size_t row, ptr, ptr_start, ptr_end=0, nnz_new, nnz_orig
    cdef double complex value
    cdef CSR out = matrix if inplace else matrix.copy()
    nnz_new = 0
    nnz_orig = csr.nnz(matrix)
    out.row_index[0] = 0
    for row in range(matrix.shape[0]):
        ptr_start, ptr_end = ptr_end, matrix.row_index[row + 1]
        for ptr in range(ptr_start, ptr_end):
            re = im = False
            value = matrix.data[ptr]
            if fabs(value.real) < tol:
                re = True
                value.real = 0
            if fabs(value.imag) < tol:
                im = True
                value.imag = 0
            if not (re & im):
                out.data[nnz_new] = value
                nnz_new += 1
        out.row_index[row + 1] = nnz_new
    if nnz_new == nnz_orig:
        return out
    return out
