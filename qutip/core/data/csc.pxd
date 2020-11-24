#cython: language_level=3

import numpy as np
cimport numpy as cnp

from qutip.core.data cimport base
from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense

cdef class CSC(base.Data):
    cdef double complex *data
    cdef base.idxint *col_index
    cdef base.idxint *row_index
    cdef size_t size
    cdef object _scipy
    cdef bint _deallocate
    cpdef CSC copy(CSC self)
    cpdef object as_scipy(CSC self, bint full=*)
    cpdef CSC sort_indices(CSC self)
    cpdef double complex trace(CSC self)
    cpdef CSC adjoint(CSC self)
    cpdef CSC conj(CSC self)
    cpdef CSC transpose(CSC self)

cpdef CSC copy_structure(CSC matrix)
cpdef CSC sorted(CSC matrix)
cpdef base.idxint nnz(CSC matrix) nogil
cpdef CSC empty(base.idxint rows, base.idxint cols, base.idxint size)
cpdef CSC empty_like(CSC Rother)
cpdef CSC zeros(base.idxint rows, base.idxint cols)
cpdef CSC identity(base.idxint dimension, double complex scale=*)

cpdef CSC fast_from_scipy(object sci)
cpdef CSC from_csr(CSR matrix)
cpdef CSC from_dense(Dense matrix)
cpdef CSR as_tr_csr(CSC matrix, bint copy=*)
cpdef CSC from_tr_csr(CSR matrix, bint copy=*)
cpdef CSR to_csr(CSC matrix)
cpdef Dense to_dense(CSC matrix)
