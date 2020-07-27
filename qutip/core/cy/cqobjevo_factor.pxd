#cython: language_level=3

from qutip.core.data cimport Data

cdef class CoeffFunc:
    cdef dict _args
    cdef int _num_ops
    cdef void _call_core(self, double t, double complex *coeff)
    cdef void _dyn_args(self, double t, Data state)

cdef class StrCoeff(CoeffFunc):
    cdef list _dyn_args_list
    cdef int _num_expect
    cdef int[2] _mat_shape
    cdef list _expect_op
    cdef complex[::1] _expect_vec
    cdef complex[::1] _vec
