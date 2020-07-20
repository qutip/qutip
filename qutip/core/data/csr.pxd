#cython: language_level=3

import numpy as np
cimport numpy as cnp

from . cimport base

cdef class CSR(base.Data):
    cdef double complex [::1] data
    cdef base.idxint [::1] col_index
    cdef base.idxint [::1] row_index
    cdef object _scipy
    cdef bint _deallocate
    cpdef CSR copy(CSR self)
    cpdef object as_scipy(CSR self, bint full=*)
    cpdef CSR sort_indices(CSR self)

# Internal structure for sorting pairs of elements.  Not actually meant to be
# used in external code.
cdef struct _data_col:
    double complex data
    base.idxint col

cdef class Sorter:
    cdef size_t size
    cdef base.idxint **argsort
    cdef _data_col *sort

    cdef void inplace(Sorter self, CSR matrix, base.idxint ptr, size_t size) nogil
    cdef void copy(Sorter self,
                   double complex *dest_data, base.idxint *dest_cols,
                   double complex *src_data, base.idxint *src_cols,
                   size_t size) nogil

cpdef CSR copy_structure(CSR matrix)
cpdef CSR sorted(CSR matrix)
cpdef base.idxint nnz(CSR matrix) nogil
cpdef CSR empty(base.idxint rows, base.idxint cols, base.idxint size)
cpdef CSR zeros(base.idxint rows, base.idxint cols)
cpdef CSR identity(base.idxint dimension, double complex scale=*)
