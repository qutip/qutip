#cython: language_level=3

import numpy as np
cimport numpy as cnp

from . cimport base

cdef class CSR(base.Data):
    cdef double complex [::1] data
    cdef base.idxint [::1] col_index
    cdef base.idxint [::1] row_index
    cdef object _scipy

cpdef CSR empty((base.idxint, base.idxint) shape, base.idxint size)
cpdef CSR zeroes((base.idxint, base.idxint) shape)
cpdef CSR identity(base.idxint dimension, double complex scale=*)
