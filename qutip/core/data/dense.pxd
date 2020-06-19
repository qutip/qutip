#cython: language_level=3

cimport numpy as cnp

from . cimport base

cdef class Dense(base.Data):
    cdef double complex [:, ::1] data
    cdef object _np
