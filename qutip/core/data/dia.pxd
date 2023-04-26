#cython: language_level=3

# from cpython cimport mem
# from libcpp.algorithm cimport sort
# from libc.math cimport fabs

# cdef extern from *:
#     void *PyMem_Calloc(size_t n, size_t elsize)

# import numpy as np
# cimport numpy as cnp

from qutip.core.data cimport base
# from qutip.core.data.dense cimport Dense

cdef class Diag(base.Data):
    cdef double complex *data
    cdef base.idxint *offsets
    cdef readonly size_t num_diag, size
    cdef object _scipy
    cdef bint _deallocate
    cpdef Diag copy(Diag self)
    cpdef object as_scipy(Diag self)
    #cpdef double complex trace(Diag self)
    #cpdef Diag adjoint(Diag self)
    #cpdef Diag conj(Diag self)
    #cpdef Diag transpose(Diag self)
