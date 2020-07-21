#cython: language_level=3

cimport numpy as cnp

from . cimport base

cdef class Dense(base.Data):
    cdef double complex *data
    cdef bint fortran
    cdef object _np
    cdef void _fix_flags(Dense self, object array)
    cpdef Dense reorder(Dense self, int fortran=*)
    cpdef Dense copy(Dense self)
    cpdef object as_ndarray(Dense self)
    cpdef object to_array(Dense self)

cpdef Dense empty(base.idxint rows, base.idxint cols, bint fortran=*)
cpdef Dense empty_like(Dense other, int fortran=*)
cpdef Dense zeros(base.idxint rows, base.idxint cols, bint fortran=*)
cpdef Dense identity(base.idxint dimension, double complex scale=*,
                     bint fortran=*)
