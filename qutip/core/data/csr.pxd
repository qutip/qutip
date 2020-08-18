#cython: language_level=3

import numpy as np
cimport numpy as cnp

from qutip.core.data cimport base
from qutip.core.data.dense cimport Dense

cdef class CSR(base.Data):
    cdef double complex *data
    cdef base.idxint *col_index
    cdef base.idxint *row_index
    cdef size_t size
    cdef object _scipy
    cdef bint _deallocate
    cpdef CSR copy(CSR self)
    cpdef object as_scipy(CSR self, bint full=*)
    cpdef CSR sort_indices(CSR self)
    cpdef double complex trace(CSR self)
    cpdef CSR adjoint(CSR self)
    cpdef CSR conj(CSR self)
    cpdef CSR transpose(CSR self)


cdef class Accumulator:
    cdef double complex *values
    cdef size_t *modified
    cdef base.idxint *nonzero
    cdef size_t _cur_row
    cdef size_t nnz, size
    cdef void scatter(Accumulator self, double complex value, base.idxint position)
    cdef base.idxint gather(Accumulator self, double complex *values, base.idxint *indices)
    cdef void reset(Accumulator self)


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


cpdef CSR fast_from_scipy(object sci)
cpdef CSR copy_structure(CSR matrix)
cpdef CSR sorted(CSR matrix)
cpdef base.idxint nnz(CSR matrix) nogil
cpdef CSR empty(base.idxint rows, base.idxint cols, base.idxint size)
cpdef CSR empty_like(CSR other)
cpdef CSR zeros(base.idxint rows, base.idxint cols)
cpdef CSR identity(base.idxint dimension, double complex scale=*)

cpdef CSR from_dense(Dense matrix)
cdef CSR from_coo_pointers(base.idxint *rows, base.idxint *cols, double complex *data,
                           base.idxint n_rows, base.idxint n_cols, base.idxint nnz)
