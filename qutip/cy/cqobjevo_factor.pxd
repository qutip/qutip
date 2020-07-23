#cython: language_level=3
cimport numpy as np

cdef class CoeffFunc:
    cdef int num_ops
    cdef list coeffs
    cdef list dyn_args_list
    cdef dict args

    cdef void _call_core(self, double t, complex* coeff)
    cdef void _dyn_args(self, double t, complex* state, int[::1] shape)

cdef class StrCoeff(CoeffFunc):
    cdef int _num_expect
    cdef int[2] _mat_shape
    cdef list _expect_op
    cdef complex[::1] _expect_vec
    cdef complex[::1] _vec
