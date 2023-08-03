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

cdef class Dia(base.Data):
    cdef double complex *data
    cdef base.idxint *offsets
    cdef readonly size_t num_diag, _max_diag
    cdef object _scipy
    cdef bint _deallocate
    cpdef Dia copy(Dia self)
    cpdef object as_scipy(Dia self, bint full=*)
    cpdef double complex trace(Dia self)
    cpdef Dia adjoint(Dia self)
    cpdef Dia conj(Dia self)
    cpdef Dia transpose(Dia self)

cpdef Dia fast_from_scipy(object sci)
cpdef Dia empty(base.idxint rows, base.idxint cols, base.idxint num_diag)
cpdef Dia empty_like(Dia other)
cpdef Dia zeros(base.idxint rows, base.idxint cols)
cpdef Dia identity(base.idxint dimension, double complex scale=*)
cpdef Dia from_dense(Dense matrix)
cpdef Dia from_csr(CSR matrix)
cpdef Dia clean_dia(Dia matrix, bint inplace=*)
