#cython: language_level=3

import numpy as np
cimport numpy as cnp

from . cimport base

cdef class CSR(base.Data):
    cdef double complex [::1] data
    cdef base.idxint [::1] col_index
    cdef base.idxint [::1] row_index
    cdef object _scipy
    cpdef CSR copy(CSR self)

cdef void sort_indices(CSR matrix) nogil
cpdef base.idxint nnz(CSR matrix) nogil
cpdef CSR empty(base.idxint rows, base.idxint cols, base.idxint size)
cpdef CSR zeroes(base.idxint rows, base.idxint cols)
cpdef CSR identity(base.idxint dimension, double complex scale=*)
