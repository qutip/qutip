#cython: language_level=3

# from cpython cimport mem
# from libcpp.algorithm cimport sort
# from libc.math cimport fabs

# cdef extern from *:
#     void *PyMem_Calloc(size_t n, size_t elsize)

# import numpy as np
# cimport numpy as cnp

from qutip.core.data cimport base
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR

cdef class Diag(base.Data):
    cdef double complex *data
    cdef base.idxint *offsets
    cdef readonly size_t num_diag, _max_diag
    cdef object _scipy
    cdef bint _deallocate
    cpdef Diag copy(Diag self)
    cpdef object as_scipy(Diag self, bint full=*)
    cpdef double complex trace(Diag self)
    cpdef Diag adjoint(Diag self)
    cpdef Diag conj(Diag self)
    cpdef Diag transpose(Diag self)

cpdef Diag fast_from_scipy(object sci)
cpdef Diag empty(base.idxint rows, base.idxint cols, base.idxint num_diag)
cpdef Diag empty_like(Diag other)
cpdef Diag zeros(base.idxint rows, base.idxint cols)
cpdef Diag identity(base.idxint dimension, double complex scale=*)
cpdef Diag from_dense(Dense matrix)
cpdef Diag from_csr(CSR matrix)
cpdef Diag clean_diag(Diag matrix, bint inplace=*)
