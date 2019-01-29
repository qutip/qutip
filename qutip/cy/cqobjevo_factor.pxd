#!python
#cython: language_level=3

cdef class CoeffFunc:
    cdef int num_ops
    cdef void _call_core(self, double t, complex* coeff)

cdef class StrCoeff(CoeffFunc):
    cdef object args
