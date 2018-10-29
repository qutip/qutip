#!python
#cython: language_level=3

cdef class coeffFunc:
    cdef int N_ops
    cdef void _call_core(self, double t, complex * coeff)

cdef class str_coeff(coeffFunc):
    cdef object args
