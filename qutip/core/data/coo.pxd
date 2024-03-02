#cython: language_level=3

cdef extern from *:
    void *PyMem_Calloc(size_t n, size_t elsize)

import numpy as np
cimport numpy as cnp

from qutip.core.data cimport base
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR

cdef class COO(base.Data):
    cdef double complex *data
    cdef base.idxint *col_index
    cdef base.idxint *row_index
    cdef size_t size
    cdef base.idxint _nnz
    cdef object _scipy
    cdef bint _deallocate
    cpdef COO copy(COO self)
    cpdef object as_scipy(COO self, bint full=*)
    cpdef double complex trace(COO self)
    cpdef COO adjoint(COO self)
    cpdef COO conj(COO self)
    cpdef COO transpose(COO self)


cpdef COO fast_from_scipy(object sci)
cpdef COO copy_structure(COO matrix)
cpdef COO sorted(COO matrix)
cpdef base.idxint nnz(COO matrix) nogil
cpdef COO empty(base.idxint rows, base.idxint cols, base.idxint size)
cpdef COO empty_like(COO other)
cpdef COO expand(COO matrix, base.idxint size)
cpdef COO zeros(base.idxint rows, base.idxint cols)
cpdef COO identity(base.idxint dimension, double complex scale=*)

cpdef COO from_dense(Dense matrix)
cpdef COO from_csr(CSR matrix)
