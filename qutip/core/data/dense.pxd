#cython: language_level=3

cimport numpy as cnp

from . cimport base

cdef class Dense(base.Data):
    cdef double complex [:, ::1] data
    cdef object _np
    cpdef Dense copy(self)
    cpdef object as_ndarray(Dense self)
    cpdef object to_array(Dense self)

cpdef Dense empty(base.idxint rows, base.idxint cols)
cpdef Dense zeros(base.idxint rows, base.idxint cols)
cpdef Dense identity(base.idxint dimension, double complex scale=*)
